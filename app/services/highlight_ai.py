"""
Context:
- This module analyzes YouTube transcript segments for short-form clip detection
- Transcript input format: [{"start": float, "duration": float, "text": str}]

Editorial Philosophy:
- Produce fewer, higher-quality clips (15-30s ideal)
- Prioritize: clarity, strong claim, standalone value
- Clips must center around a strong editorial anchor
- The strongest statement should appear early in the clip (within first 25%)

Scoring Architecture (3-domain system + learning layer):
1. Core Editorial Score (MANDATORY - must pass for any clip)
   - Strong declarative claims
   - Standalone statements
   - Assertive/instructional tone
   
2. Support Score (bonus signals)
   - Emotional resonance
   - Emphasis markers
   - First-person credibility

3. Penalty Score (disqualifiers)
   - Filler-heavy segments
   - Context-dependent statements
   - Questions without answers
   - Storytelling without takeaway

4. Editorial Learning Layer (ADDITIVE - biases scores toward historical success)
   - Adapts from accept/reject feedback
   - Deterministic, no ML/AI
   - Capped at ±4 points total bias
   - Never overrides core gates

Output:
- List of highlight candidates with: {start, end, score, category, reason}

Constraints:
- NO AI APIs, NO external dependencies
- Fully deterministic, heuristic-based
- Readable, testable, maintainable
"""
import re
from typing import List, Dict, Tuple, Optional

from app.services.editorial_memory import (
  load_editorial_bias,
  apply_editorial_bias,
  EditorialBiasProfile,
  EditorialLearningProfile
)


def _clean_transcript_text(text: str) -> str:
  """Clean transcript artifacts from auto-caption systems.
  
  Removes:
  - Timestamp patterns: <00:02:02.000>
  - HTML-like tags: <c>, </c>, <font>, etc.
  - Excessive whitespace
  - Duplicated consecutive words (caption overlap artifacts)
  
  Returns clean text ready for analysis.
  """
  if not text:
    return ""
  
  # Remove timestamp patterns: <HH:MM:SS.mmm> or <MM:SS.mmm> or <SS.mmm>
  import re
  text = re.sub(r'<\d{1,2}:\d{2}:\d{2}\.\d{3}>', '', text)
  text = re.sub(r'<\d{1,2}:\d{2}\.\d{3}>', '', text)
  text = re.sub(r'<\d{1,2}\.\d{3}>', '', text)
  
  # Remove HTML-like tags: <c>, </c>, <font>, <i>, etc.
  text = re.sub(r'</?[a-z]+[^>]*>', '', text, flags=re.IGNORECASE)
  
  # Normalize whitespace
  text = ' '.join(text.split())
  
  # Deduplicate immediate repeated words (caption overlap)
  # Example: "fokus fokus pada" -> "fokus pada"
  words = text.split()
  deduped = []
  prev = None
  for word in words:
    word_lower = word.lower()
    if word_lower != prev:
      deduped.append(word)
      prev = word_lower
  
  return ' '.join(deduped).strip()


def _tokenize(text: str) -> List[str]:
  # Simple, library-free tokenizer
  cleaned = []
  for ch in text.lower():
    if ch.isalnum() or ch in {" ", "-"}:
      cleaned.append(ch)
    else:
      cleaned.append(" ")
  return [w for w in "".join(cleaned).split() if w]


def _filler_and_stopwords() -> Tuple[set[str], set[str]]:
  # Small, hand-curated Indonesian filler/stopword lists.
  # Keep them short and high-signal to stay deterministic and maintainable.
  filler = {
    "eee", "ee", "eh", "em", "umm", "hmm", "nah", "oke", "ok", "ya", "yah",
    "anu", "gitu", "gini", "kayak", "kaya", "nih", "sih", "deh", "dong", "kan",
    "aja", "lah", "loh", "lho", "tau", "tahu", "maksudnya",
  }
  stop = {
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "pada", "itu", "ini", "nya",
    "saya", "aku", "kita", "kami", "kamu", "lu", "gue", "dia", "mereka",
    "jadi", "terus", "lalu", "kemudian", "nah", "oke", "karena", "sebab", "biar",
    "dengan", "dalam", "sebagai", "adalah", "ialah", "akan", "sudah", "udah", "belum",
    "bisa", "nggak", "tidak", "ga", "gak", "pun", "kok", "juga", "aja", "lagi",
  }
  return filler, stop


def _is_punchline(text: str) -> bool:
  """Enhanced punchline detector - focus on assertive, standalone statements.

  Editorial intent: 
  - Identify clip-ready statements (advice, strong opinion, emotional spike)
  - NOT strict on word count (allow 6-18 words if structure is strong)
  - Prioritize declarative, assertive tone over length
  
  Returns True if text is a strong standalone statement suitable as clip anchor.
  """
  t = (text or "").strip().lower()
  if not t:
    return False

  words = _tokenize(t)
  n = len(words)
  
  # Editorial: allow more flexibility (6-18 words) instead of rigid 5-12
  # Strong structure matters more than exact length
  if n < 6 or n > 18:
    return False

  # Assertive openers - signal a definitive claim
  strong_openers = (
    "masalahnya", "kenyataannya", "faktanya", "intinya", "kuncinya",
    "sebenarnya", "artinya", "poinnya", "kesalahannya", "solusinya",
  )
  
  # Absolute/definitive language - signals strong claim
  absolute_language = {
    "selalu", "kebanyakan", "semua", "pasti", "cuma", "hanya", "satu-satunya",
    "never", "nggak", "tidak", "gak", "ga", "bukan", "tanpa",
  }
  
  polar_phrases = (
    "tidak pernah", "nggak pernah", "gak pernah", "gak ada", "tidak ada",
  )
  
  # Imperative/instructional verbs - signal actionable advice
  punch_verbs = {"harus", "wajib", "jangan", "berhenti", "fokus", "ingat", "pastikan", "hindari"}

  has_opener = t.startswith(strong_openers)
  has_absolute = any(w in absolute_language for w in words) or any(p in t for p in polar_phrases)
  has_take = any(w in punch_verbs for w in words)
  has_emotion = _emotional_signal_score(t)[0] > 0

  # Require at least ONE strong signal for a statement to qualify as punchline
  signal_count = sum([has_opener, has_absolute, has_take, has_emotion])
  
  # If 2+ signals present, it's definitely a punchline
  # If 1 signal + reasonably short (6-12 words), also qualifies
  return signal_count >= 2 or (signal_count >= 1 and 6 <= n <= 12)


def _quantile(values: List[int], q: float) -> float:
  """Deterministic quantile for small lists (no external deps)."""
  if not values:
    return 0.0
  xs = sorted(values)
  if len(xs) == 1:
    return float(xs[0])
  q = max(0.0, min(1.0, q))
  pos = (len(xs) - 1) * q
  lo = int(pos)
  hi = min(len(xs) - 1, lo + 1)
  frac = pos - lo
  return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _adaptive_peak_threshold(seg_scores: List[int]) -> int:
  """Adaptive threshold so flat transcripts still yield highlights, while spiky ones remain selective."""
  if not seg_scores:
    return 2
  p75 = _quantile(seg_scores, 0.75)
  p50 = _quantile(seg_scores, 0.50)
  # Base threshold near upper quartile, but not too strict.
  thr = int(round(max(2.0, min(6.0, p75))))
  # If distribution is flat (p75 close to median), lower slightly.
  if (p75 - p50) < 1.0:
    thr = max(2, thr - 1)
  return thr


def _compress_reasons(reasons: List[str], limit: int = 3) -> List[str]:
  """Deduplicate and keep the most meaningful reasons.

  Why: editorial output should be readable (2–3 concise reasons).
  """
  if not reasons:
    return []
  seen = set()
  deduped: List[str] = []
  for r in reasons:
    r = r.strip()
    if not r:
      continue
    key = r.lower()
    if key in seen:
      continue
    seen.add(key)
    deduped.append(r)

  priority = [
    "Punchline berkualitas",
    "Punchline",
    "Hook di awal (ideal)",
    "Pernyataan deklaratif kuat",
    "Nada instruktif/tegas",
    "Mengandung kata kunci editorial",
    "Meaning density tinggi",
    "Sinyal urgensi/risiko",
    "Sinyal gagal/frustrasi",
    "Sinyal sukses/lega",
    "Awalan penekanan",
    "Claim+justifikasi",
    "Durasi optimal",
  ]

  picked: List[str] = []
  for p in priority:
    for r in deduped:
      if r == p or r.startswith(p):
        if r not in picked:
          picked.append(r)
          if len(picked) >= limit:
            return picked

  for r in deduped:
    if len(picked) >= limit:
      break
    if r not in picked:
      picked.append(r)
  return picked


def _is_hanging_start(text: str) -> bool:
  """Check if text starts with a conjunction or hanging word.
  
  These are incomplete sentence starts that need preceding context.
  """
  if not text:
    return True
  
  text_lower = text.strip().lower()
  
  # Conjunctions and connectors that signal incomplete thought
  hanging_starts = (
    "jadi", "karena", "kalau", "terus", "lalu", "kemudian",
    "makanya", "soalnya", "tapi", "namun", "atau", "dan",
    "sebab", "maka", "sehingga", "nah", "oke",
  )
  
  return text_lower.startswith(hanging_starts)


def _is_incomplete_end(text: str) -> bool:
  """Check if text ends with an unresolved phrase.
  
  These endings signal the thought continues in the next segment.
  """
  if not text:
    return True
  
  text_lower = text.strip().lower()
  
  # Endings that signal continuation
  incomplete_endings = (
    "karena", "kalau", "jadi", "makanya", "yaitu", "seperti",
    "yaitu", "adalah", "itu", "ini", "artinya", "berarti",
  )
  
  # Questions without answers signal incompleteness
  question_markers = ("hah", "pertanyaannya", "kenapa", "bagaimana")
  if any(text_lower.endswith(q) for q in question_markers):
    return True
  if "?" in text:
    return True
  
  words = _tokenize(text_lower)
  if not words:
    return True
  
  last_word = words[-1]
  return last_word in incomplete_endings


def _is_educational_content(text: str) -> bool:
  """Detect if text contains educational/explanatory content.
  
  Educational content includes:
  - Numbers and units (gram, liter, persen, kali, bagi)
  - Mathematical operations
  - Cause-effect reasoning
  - Conversions and calculations
  
  These clips require longer duration and complete reasoning chains.
  """
  if not text:
    return False
  
  text_lower = text.strip().lower()
  words = _tokenize(text_lower)
  
  # Check for numbers (digits in text)
  has_numbers = any(char.isdigit() for char in text)
  
  # Units and measurements
  units = {
    "gram", "kilogram", "kg", "liter", "meter", "km", "cm",
    "persen", "percent", "%", "kali", "bagi", "dikali", "dibagi",
    "ribu", "juta", "miliar", "rupiah", "dollar",
  }
  has_units = any(u in words for u in units)
  
  # Mathematical/reasoning phrases
  reasoning_phrases = [
    r"\bkalau\s+.+\s+maka\b",
    r"\bjika\s+.+\s+maka\b",
    r"\bartinya\b",
    r"\bberarti\b",
    r"\bsetara\s+dengan\b",
    r"\bsama\s+dengan\b",
  ]
  has_reasoning = any(re.search(p, text_lower) for p in reasoning_phrases)
  
  # Educational signal: numbers + units OR numbers + reasoning
  is_educational = (has_numbers and has_units) or (has_numbers and has_reasoning)
  
  return is_educational


def _check_informational_completeness(window_segments: List[str]) -> Tuple[bool, str]:
  """Check if a clip window contains complete information.
  
  A complete clip must have:
  1. Premise (context/setup)
  2. Transformation (reasoning/calculation/explanation)
  3. Conclusion (result/implication/takeaway)
  
  Incomplete patterns to reject:
  - Only numeric result without context
  - Conversion without explanation
  - Question without answer
  - Premise without conclusion
  
  Returns: (is_complete, reason)
  """
  if not window_segments:
    return False, "Empty window"
  
  combined_text = " ".join(window_segments).lower()
  
  # Check if this is educational content
  is_edu = _is_educational_content(combined_text)
  
  if is_edu:
    # Educational clips need premise + transformation + conclusion
    
    # Check for premise (setup, context)
    premise_markers = {
      "kalau", "jika", "misalnya", "contoh", "kita", "ada", "punya",
      "awalnya", "pertama", "mulai",
    }
    has_premise = any(w in _tokenize(combined_text) for w in premise_markers)
    
    # Check for transformation (calculation, reasoning)
    transformation_markers = {
      "kali", "bagi", "dikali", "dibagi", "tambah", "kurang",
      "artinya", "berarti", "jadi", "setara", "sama",
    }
    has_transformation = any(w in _tokenize(combined_text) for w in transformation_markers)
    
    # Check for conclusion (result, implication)
    conclusion_markers = {
      "jadi", "hasilnya", "kesimpulannya", "artinya", "berarti",
      "itulah", "makanya", "sehingga", "maka",
    }
    conclusion_patterns = [
      r"\bjadi\s+.+\s+(adalah|itu)\b",
      r"\bartinya\s+.+\b",
      r"\bberarti\s+.+\b",
    ]
    has_conclusion = (any(w in _tokenize(combined_text) for w in conclusion_markers) or
                     any(re.search(p, combined_text) for p in conclusion_patterns))
    
    # Educational content must have at least transformation + conclusion
    # Premise is helpful but not always explicitly stated
    if not has_transformation:
      return False, "Educational content lacks transformation/reasoning"
    if not has_conclusion:
      return False, "Educational content lacks conclusion"
    
    # Additional check: don't end on a question
    if combined_text.strip().endswith("?") or "hah?" in combined_text:
      # Check if question is followed by answer
      segments_after_question = []
      found_q = False
      for seg in window_segments:
        if "?" in seg or "hah" in seg.lower():
          found_q = True
        elif found_q:
          segments_after_question.append(seg)
      
      if found_q and len(segments_after_question) < 1:
        return False, "Question without answer"
  
  # For non-educational clips, check basic completeness
  else:
    # Must not end with incomplete marker
    if _is_incomplete_end(combined_text):
      return False, "Incomplete ending"
    
    # Must have reasonable length (not just a fragment)
    if len(window_segments) < 2:
      return False, "Single segment insufficient"
  
  return True, "Complete information"


def _insight_components(text: str) -> Tuple[bool, bool, bool]:
  """Detect insight components (deterministic, no external deps).

  Insight components:
  1) CLAIM: declarative / punchline / assertive instruction
  2) REASON (WHY): causal markers (karena, sebab, artinya, berarti, jadi, makanya)
  3) IMPLICATION/TAKEAWAY: markers (hasilnya, pelajarannya, dampaknya, sehingga, maka)
  """
  t = (text or "").strip()
  if not t:
    return False, False, False

  t_lower = t.lower()

  # CLAIM: reuse existing core/punchline heuristics to preserve architecture
  core_pass, core_score = _calculate_core_editorial_pass(t)
  claim_present = _is_punchline(t) or core_pass or core_score >= 4

  reason_markers = (
    "karena", "sebab", "artinya", "berarti", "jadi", "makanya",
  )
  implication_markers = (
    "hasilnya", "pelajarannya", "dampaknya", "sehingga", "maka",
  )

  reason_present = any(re.search(rf"\b{re.escape(m)}\b", t_lower) for m in reason_markers)
  implication_present = any(re.search(rf"\b{re.escape(m)}\b", t_lower) for m in implication_markers)

  return claim_present, reason_present, implication_present


def _has_insight_structure(text: str) -> Tuple[bool, List[str]]:
  """Insight Completion Layer.

  Passes if it contains at least TWO of the THREE components:
  CLAIM, REASON, IMPLICATION.

  Returns: (passes_insight, reasons)
  """
  claim_present, reason_present, implication_present = _insight_components(text)
  components = int(claim_present) + int(reason_present) + int(implication_present)
  passes = components >= 2

  reasons: List[str] = []

  if claim_present and reason_present and implication_present:
    reasons.append("Insight lengkap (claim, alasan, dampak)")
    return True, reasons

  if claim_present and reason_present:
    reasons.append("Claim + reasoning jelas")
  elif claim_present and implication_present:
    reasons.append("Claim dengan implikasi kuat")
  elif reason_present and implication_present:
    reasons.append("Alasan + dampak (tanpa claim eksplisit)")
  else:
    # Single-component cases: explain the failure
    if claim_present:
      reasons.append("Punchline tanpa alasan/dampak")
    elif reason_present:
      reasons.append("Alasan tanpa claim/dampak")
    elif implication_present:
      reasons.append("Dampak tanpa claim/alasan")
    else:
      reasons.append("Tidak ada struktur insight")

  return passes, reasons


def _detect_educational_sequences(transcript: List[Dict], min_length: int = 3) -> List[int]:
  """Detect educational sequences where multiple segments form a complete explanation.
  
  Educational content often has low-scoring individual segments (context-dependent,
  hanging starts, numeric results without context). But when combined, they form
  valuable explanatory content with premise + reasoning + conclusion.
  
  Returns list of anchor indices (typically the first educational segment in each sequence).
  """
  if len(transcript) < min_length:
    return []
  
  n = len(transcript)
  sequences = []
  i = 0
  
  while i < n:
    # Check if current segment starts an educational sequence
    seg_text = str(transcript[i].get("text", ""))
    
    if _is_educational_content(seg_text):
      # Found potential start - check next few segments
      sequence_texts = [seg_text]
      j = i + 1
      
      # Collect consecutive educational or related segments (max 8 segments, ~30s)
      while j < n and j < i + 8:
        next_text = str(transcript[j].get("text", ""))
        
        # Continue sequence if:
        # 1. Also educational, OR
        # 2. Contains reasoning/conclusion markers, OR
        # 3. Short connector segment (< 5 words)
        is_edu = _is_educational_content(next_text)
        has_reasoning = any(m in next_text.lower() for m in ["jadi", "artinya", "berarti", "hasilnya", "itulah"])
        is_short_connector = len(_tokenize(next_text)) < 5
        
        if is_edu or has_reasoning or is_short_connector:
          sequence_texts.append(next_text)
          j += 1
        else:
          break
      
      # Check if sequence is long enough and has complete structure
      if len(sequence_texts) >= min_length:
        is_complete, _ = _check_informational_completeness(sequence_texts)
        if is_complete:
          sequences.append(i)  # Mark start of sequence as anchor
          i = j  # Skip past this sequence
          continue
    
    i += 1
  
  return sequences


def _dynamic_window_profile(anchor_text: str, learning_profile: Optional[EditorialLearningProfile] = None) -> Tuple[Dict[str, float], List[str]]:
  """Pick window sizing based on the anchor's category and punchiness.

  Editorial intent:
  - Punchline: short & tight (8-18s) - ONLY exception to duration rules
  - Educational: LONG (18-35s MINIMUM) - needs complete premise + reasoning + conclusion
  - Hard advice: medium (12-25s) - actionable and crisp
  - Warning: medium (15-30s) - need context for risk/consequence
  - Lesson learned: medium-long (15-35s) - story needs breathing room
  - Insight: medium (12-30s) - explanation with clarity
  - Motivational: medium-short (12-25s) - punchy inspiration
  
  CRITICAL: min_duration is HARD MINIMUM - never produce shorter clips
  All clips capped at 60s max (enforced elsewhere).
  """
  profile_reasons: List[str] = []

  if _is_punchline(anchor_text):
    prof = {
      "min_duration": 8.0,  # ONLY category allowed to be under 10s
      "max_duration": 18.0,
      "pre_roll": 0.5,  # Minimal pre-roll for punchlines
      "post_roll": 1.0,
      "max_pre_context": 4.0,   # Aggressively trim before anchor
      "max_post_context": 6.0,  # Allow extension for complete thought
    }
    category_key = "punchline"
    # Apply bounded learning adjustments
    if learning_profile is not None and not learning_profile.is_zero():
      dmin = float(learning_profile.min_duration_delta_by_category.get(category_key, 0.0))
      dmax = float(learning_profile.max_duration_delta_by_category.get(category_key, 0.0))
      dpre = float(learning_profile.max_pre_context_delta_by_category.get(category_key, 0.0))
      dpost = float(learning_profile.max_post_context_delta_by_category.get(category_key, 0.0))

      if dmin > 0:
        prof["min_duration"] = min(prof["max_duration"], prof["min_duration"] + dmin)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dmax < 0:
        prof["max_duration"] = max(prof["min_duration"], prof["max_duration"] + dmax)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dpre != 0:
        prof["max_pre_context"] = max(1.0, prof["max_pre_context"] + dpre)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
      if dpost != 0:
        prof["max_post_context"] = max(3.0, prof["max_post_context"] + dpost)
        profile_reasons.append("Disesuaikan dari feedback pengguna")

    return prof, _compress_reasons(profile_reasons, limit=2)

  cat = _classify(anchor_text)
  
  # EDUCATIONAL clips need longer duration for informational completeness
  # SHORT EDUCATIONAL CLIPS ARE FORBIDDEN - they confuse rather than educate
  if cat == "educational":
    prof = {
      "min_duration": 18.0,  # HARD MINIMUM - never allow shorter (preferred 20s+)
      "max_duration": 35.0,  # Allow longer for complete explanations
      "pre_roll": 0.8,  # Need context (premise)
      "post_roll": 1.0,
      "max_pre_context": 12.0,  # More context allowed
      "max_post_context": 20.0,  # Strongly favor post-context (explanation/resolution)
    }
    category_key = "educational"
    if learning_profile is not None and not learning_profile.is_zero():
      dmin = float(learning_profile.min_duration_delta_by_category.get(category_key, 0.0))
      dmax = float(learning_profile.max_duration_delta_by_category.get(category_key, 0.0))
      dpre = float(learning_profile.max_pre_context_delta_by_category.get(category_key, 0.0))
      dpost = float(learning_profile.max_post_context_delta_by_category.get(category_key, 0.0))
      if dmin > 0:
        prof["min_duration"] = min(prof["max_duration"], prof["min_duration"] + dmin)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dmax < 0:
        prof["max_duration"] = max(prof["min_duration"], prof["max_duration"] + dmax)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dpre != 0:
        prof["max_pre_context"] = max(3.0, prof["max_pre_context"] + dpre)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
      if dpost != 0:
        prof["max_post_context"] = max(8.0, prof["max_post_context"] + dpost)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
    return prof, _compress_reasons(profile_reasons, limit=2)
  
  if cat == "hard_advice":
    prof = {
      "min_duration": 12.0,
      "max_duration": 25.0,
      "pre_roll": 0.5,  # Reduced pre-roll
      "post_roll": 1.0,
      "max_pre_context": 5.0,
      "max_post_context": 10.0,  # Increased for complete thoughts
    }
    category_key = "hard_advice"
    if learning_profile is not None and not learning_profile.is_zero():
      dmin = float(learning_profile.min_duration_delta_by_category.get(category_key, 0.0))
      dmax = float(learning_profile.max_duration_delta_by_category.get(category_key, 0.0))
      dpre = float(learning_profile.max_pre_context_delta_by_category.get(category_key, 0.0))
      dpost = float(learning_profile.max_post_context_delta_by_category.get(category_key, 0.0))
      if dmin > 0:
        prof["min_duration"] = min(prof["max_duration"], prof["min_duration"] + dmin)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dmax < 0:
        prof["max_duration"] = max(prof["min_duration"], prof["max_duration"] + dmax)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dpre != 0:
        prof["max_pre_context"] = max(1.0, prof["max_pre_context"] + dpre)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
      if dpost != 0:
        prof["max_post_context"] = max(6.0, prof["max_post_context"] + dpost)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
    return prof, _compress_reasons(profile_reasons, limit=2)
  
  if cat == "warning":
    prof = {
      "min_duration": 15.0,  # HARD MINIMUM - need full context
      "max_duration": 30.0,
      "pre_roll": 0.5,  # Reduced pre-roll
      "post_roll": 1.0,
      "max_pre_context": 8.0,
      "max_post_context": 15.0,  # Favor consequences/resolution
    }
    category_key = "warning"
    if learning_profile is not None and not learning_profile.is_zero():
      dmin = float(learning_profile.min_duration_delta_by_category.get(category_key, 0.0))
      dmax = float(learning_profile.max_duration_delta_by_category.get(category_key, 0.0))
      dpre = float(learning_profile.max_pre_context_delta_by_category.get(category_key, 0.0))
      dpost = float(learning_profile.max_post_context_delta_by_category.get(category_key, 0.0))
      if dmin > 0:
        prof["min_duration"] = min(prof["max_duration"], prof["min_duration"] + dmin)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dmax < 0:
        prof["max_duration"] = max(prof["min_duration"], prof["max_duration"] + dmax)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dpre != 0:
        prof["max_pre_context"] = max(2.0, prof["max_pre_context"] + dpre)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
      if dpost != 0:
        prof["max_post_context"] = max(8.0, prof["max_post_context"] + dpost)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
    return prof, _compress_reasons(profile_reasons, limit=2)
  
  if cat == "lesson_learned":
    prof = {
      "min_duration": 15.0,  # HARD MINIMUM - need story arc
      "max_duration": 35.0,
      "pre_roll": 0.8,
      "post_roll": 1.0,
      "max_pre_context": 10.0,
      "max_post_context": 18.0,  # Strongly favor resolution/lesson
    }
    category_key = "lesson_learned"
    if learning_profile is not None and not learning_profile.is_zero():
      dmin = float(learning_profile.min_duration_delta_by_category.get(category_key, 0.0))
      dmax = float(learning_profile.max_duration_delta_by_category.get(category_key, 0.0))
      dpre = float(learning_profile.max_pre_context_delta_by_category.get(category_key, 0.0))
      dpost = float(learning_profile.max_post_context_delta_by_category.get(category_key, 0.0))
      if dmin > 0:
        prof["min_duration"] = min(prof["max_duration"], prof["min_duration"] + dmin)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dmax < 0:
        prof["max_duration"] = max(prof["min_duration"], prof["max_duration"] + dmax)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dpre != 0:
        prof["max_pre_context"] = max(3.0, prof["max_pre_context"] + dpre)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
      if dpost != 0:
        prof["max_post_context"] = max(10.0, prof["max_post_context"] + dpost)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
    return prof, _compress_reasons(profile_reasons, limit=2)
  
  if cat == "motivational":
    prof = {
      "min_duration": 12.0,  # HARD MINIMUM
      "max_duration": 25.0,
      "pre_roll": 0.5,  # Reduced pre-roll
      "post_roll": 1.0,
      "max_pre_context": 5.0,
      "max_post_context": 12.0,  # Favor motivational payoff
    }
    category_key = "motivational"
    if learning_profile is not None and not learning_profile.is_zero():
      dmin = float(learning_profile.min_duration_delta_by_category.get(category_key, 0.0))
      dmax = float(learning_profile.max_duration_delta_by_category.get(category_key, 0.0))
      dpre = float(learning_profile.max_pre_context_delta_by_category.get(category_key, 0.0))
      dpost = float(learning_profile.max_post_context_delta_by_category.get(category_key, 0.0))
      if dmin > 0:
        prof["min_duration"] = min(prof["max_duration"], prof["min_duration"] + dmin)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dmax < 0:
        prof["max_duration"] = max(prof["min_duration"], prof["max_duration"] + dmax)
        profile_reasons.append("Sesuai preferensi durasi pengguna")
      if dpre != 0:
        prof["max_pre_context"] = max(1.0, prof["max_pre_context"] + dpre)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
      if dpost != 0:
        prof["max_post_context"] = max(6.0, prof["max_post_context"] + dpost)
        profile_reasons.append("Disesuaikan dari feedback pengguna")
    return prof, _compress_reasons(profile_reasons, limit=2)
  
  # Default: insight
  prof = {
    "min_duration": 12.0,  # HARD MINIMUM - need complete explanation
    "max_duration": 30.0,
    "pre_roll": 0.5,  # Reduced pre-roll
    "post_roll": 1.0,
    "max_pre_context": 8.0,
    "max_post_context": 15.0,  # Favor explanation/reasoning
  }

  category_key = cat
  if learning_profile is not None and not learning_profile.is_zero():
    dmin = float(learning_profile.min_duration_delta_by_category.get(category_key, 0.0))
    dmax = float(learning_profile.max_duration_delta_by_category.get(category_key, 0.0))
    dpre = float(learning_profile.max_pre_context_delta_by_category.get(category_key, 0.0))
    dpost = float(learning_profile.max_post_context_delta_by_category.get(category_key, 0.0))
    if dmin > 0:
      prof["min_duration"] = min(prof["max_duration"], prof["min_duration"] + dmin)
      profile_reasons.append("Sesuai preferensi durasi pengguna")
    if dmax < 0:
      prof["max_duration"] = max(prof["min_duration"], prof["max_duration"] + dmax)
      profile_reasons.append("Sesuai preferensi durasi pengguna")
    if dpre != 0:
      prof["max_pre_context"] = max(2.0, prof["max_pre_context"] + dpre)
      profile_reasons.append("Disesuaikan dari feedback pengguna")
    if dpost != 0:
      prof["max_post_context"] = max(8.0, prof["max_post_context"] + dpost)
      profile_reasons.append("Disesuaikan dari feedback pengguna")

  return prof, _compress_reasons(profile_reasons, limit=2)


def _emotional_signal_score(text: str) -> Tuple[int, List[str]]:
  """Detect emotional signals in text.
  
  Editorial: Emotional content often makes compelling clips.
  Keep lexicon small and high-precision.
  """
  words = _tokenize(text)
  if not words:
    return 0, []

  # Emotion lexicon: keep small, high-precision.
  fear_urgency = {
    "takut", "khawatir", "cemas", "panik", "bahaya", "ancam", "darurat", "segera", "urgent", "krusial",
    "parah", "fatal", "berisiko", "resiko", "hancur",
  }
  success_relief = {
    "berhasil", "sukses", "menang", "naik", "tembus", "lega", "akhirnya", "tenang", "selamat",
  }
  failure_frustration = {
    "gagal", "kecewa", "frustrasi", "capek", "lelah", "stress", "stres", "marah", "kesal", "kacau",
    "susah", "sulit", "berantakan", "nyesel", "menyerah",
  }

  score = 0
  reasons: List[str] = []

  if any(w in fear_urgency for w in words):
    score += 2
    reasons.append("Sinyal urgensi/risiko")
  if any(w in success_relief for w in words):
    score += 1
    reasons.append("Sinyal sukses/lega")
  if any(w in failure_frustration for w in words):
    score += 1
    reasons.append("Sinyal gagal/frustrasi")

  # Exclamation often indicates emphasis (but keep small)
  if "!" in text:
    score += 1
    reasons.append("Penekanan emosional")

  return score, reasons


def _score_text(text: str) -> Tuple[int, List[str]]:
  """Score text using 3-domain architecture.
  
  Editorial Philosophy:
  - Core score is MANDATORY - measures editorial value (claim strength, standalone quality)
  - Support score provides bonus signals (emotion, emphasis, credibility)
  - Penalty score disqualifies weak content (filler, context-dependence, questions)
  
  A clip must pass core editorial checks to be viable.
  """
  words = _tokenize(text)
  n = len(words)
  
  core_score = 0
  support_score = 0
  penalty_score = 0
  reasons: List[str] = []

  is_punch = _is_punchline(text)

  # ============================================================
  # DOMAIN 1: CORE EDITORIAL SCORE (mandatory for any clip)
  # ============================================================
  # Measures: claim strength, declarative structure, standalone value
  
  # Declarative claim markers - strongest signal
  declarative_patterns = [
    r"\bkunci(nya)?\s+(itu\s+)?adalah\b",
    r"\bintinya\s+(itu\s+)?\b",
    r"\bartinya\s+\b",
    r"\byang\s+terjadi\s+(itu\s+)?adalah\b",
    r"\bpoin(nya)?\s+(itu\s+)?\b",
    r"\bpelajaran(nya)?\s+(itu\s+)?\b",
    r"\bmasalah\s+utamanya\s+(itu\s+)?\b",
    r"\bsolusi(nya)?\s+(itu\s+)?\b",
    r"\bfakta(nya)?\s+(itu\s+)?\b",
    r"\bkenyataan(nya)?\s+(itu\s+)?\b",
  ]
  is_declarative = any(re.search(pat, text.lower()) for pat in declarative_patterns)
  if is_declarative:
    core_score += 4  # Strong core signal
    reasons.append("Pernyataan deklaratif kuat")

  # Core keywords - signal valuable insight/advice
  core_keywords = {
    "penting", "kunci", "kuncinya", "masalah", "solusi", "kesalahan",
    "intinya", "sebenarnya", "faktanya", "kenyataannya", "artinya",
  }
  if any(w in core_keywords for w in words):
    core_score += 3
    reasons.append("Mengandung kata kunci editorial")

  # Assertive/instructional tone - signals actionable advice
  imperative_markers = {"harus", "wajib", "jangan", "pastikan", "ingat", "fokus", "hindari"}
  if any(w in imperative_markers for w in words):
    core_score += 3
    reasons.append("Nada instruktif/tegas")

  # Punchline bonus - editorial gold
  if is_punch:
    core_score += 3
    reasons.append("Punchline berkualitas")
  
  # Educational content bonus - complete explanations with premise + reasoning + conclusion
  # Educational clips are valuable when they contain full information flow
  if _is_educational_content(text):
    # Check for reasoning markers (indicates complete explanation, not just numbers)
    reasoning_markers = {"artinya", "berarti", "jadi", "makanya", "hasilnya", "itulah", "begitulah"}
    has_reasoning = any(w in reasoning_markers for w in words)
    
    # Check for sufficient length (complete explanations need context)
    if has_reasoning and n >= 15:
      core_score += 5  # Strong educational value
      reasons.append("Konten edukasi lengkap")
    elif n >= 10:
      core_score += 3  # Moderate educational value
      reasons.append("Konten edukasi")

  # Meaning density - information-rich content
  filler, stop = _filler_and_stopwords()
  if words:
    content_words = [w for w in words if (w not in filler and w not in stop)]
    content_ratio = len(content_words) / len(words)
    if content_ratio >= 0.55 and len(content_words) >= 5:
      core_score += 2
      reasons.append("Meaning density tinggi")

  # ============================================================
  # DOMAIN 2: SUPPORT SCORE (bonus signals, not mandatory)
  # ============================================================
  # Measures: emotion, emphasis, credibility markers
  
  # Emotional signals
  em_score, em_reasons = _emotional_signal_score(text)
  if em_score > 0:
    support_score += min(em_score, 2)  # Cap emotional bonus
    reasons.extend(em_reasons[:1])  # Keep concise

  # Emphasis starters - signal important point
  emphasis_starts = [
    "jadi", "intinya", "yang paling", "ingat", "sebenarnya", "singkatnya",
    "poin", "kunci", "masalahnya",
  ]
  if any(text.strip().lower().startswith(prefix) for prefix in emphasis_starts):
    support_score += 1
    reasons.append("Awalan penekanan")

  # First-person credibility
  first_person = {"saya", "aku", "kita", "kami", "gue"}
  if any(w in first_person for w in words):
    support_score += 1
    reasons.append("Insight orang pertama")

  # Claim + support structure
  claim_words = {"kunci", "penting", "masalah", "solusi", "harus", "jangan", "intinya"}
  support_words = {"karena", "sebab", "soalnya", "makanya", "jadi", "biar", "supaya"}
  has_claim = any(w in claim_words for w in words)
  has_support = any(w in support_words for w in words)
  if has_claim and has_support:
    support_score += 1
    reasons.append("Claim+justifikasi")

  # ============================================================
  # DOMAIN 3: PENALTY SCORE (disqualifiers)
  # ============================================================
  # Measures: filler density, context-dependence, weak structure
  
  # Filler-heavy content penalty
  if words:
    filler_count = sum(1 for w in words if w in filler)
    filler_ratio = filler_count / len(words)
    if filler_ratio >= 0.40:
      penalty_score -= 4
      reasons.append("Terlalu banyak filler")
    elif filler_ratio >= 0.30:
      penalty_score -= 2
      reasons.append("Cukup banyak filler")

  # Question penalty (unless part of rhetorical pattern)
  lowered = text.strip().lower()
  question_starts = {"apa", "mengapa", "kenapa", "bagaimana", "kapan", "dimana", "berapa"}
  is_question = "?" in text or any(lowered.startswith(q) for q in question_starts)
  if is_question:
    # Allow rhetorical questions if followed by answer signal
    answer_markers = {"karena", "jawabannya", "ternyata", "sebenarnya", "faktanya"}
    has_answer = any(w in answer_markers for w in words)
    if not has_answer:
      penalty_score -= 3
      reasons.append("Pertanyaan tanpa jawaban")

  # Storytelling without claim penalty
  storytelling_markers = {
    "waktu", "dulu", "pas", "kemarin", "terus", "lalu", "kemudian", "abis", "habis",
  }
  has_story = any(w in storytelling_markers for w in words)
  # Only penalize if it's storytelling WITHOUT a takeaway
  if has_story and not has_claim and not is_declarative and n >= 8:
    penalty_score -= 3
    reasons.append("Storytelling tanpa takeaway")

  # Length-based penalty (but flexible for punchlines)
  if not is_punch:
    if n < 5:
      penalty_score -= 3
      reasons.append("Terlalu pendek")
    elif n > 30:
      penalty_score -= 2
      reasons.append("Terlalu panjang")

  # Stopword dominance (context-dependent, weak standalone value)
  if words:
    stop_count = sum(1 for w in words if w in stop)
    if stop_count / len(words) >= 0.70 and core_score <= 2:
      penalty_score -= 2
      reasons.append("Terlalu banyak stopword")

  # ============================================================
  # FINAL SCORE CALCULATION
  # ============================================================
  # Core score must be positive for clip to be viable
  # Support and penalty modify final result
  
  total_score = core_score + support_score + penalty_score
  
  # Compress reasons for readability
  reasons = _compress_reasons(reasons, limit=3)
  
  return total_score, reasons


def _classify(text: str) -> str:
  """Enhanced classification based on intent, not just keywords.
  
  Categories:
  - educational: explanations with numbers, units, reasoning (needs longer duration)
  - insight: reveals understanding, explains "why" or "what really matters"
  - hard_advice: actionable instructions, imperative, "do this / don't do that"
  - warning: caution, risk, consequences of not doing something
  - lesson_learned: retrospective wisdom, "I learned", past experience
  - motivational: encouragement, belief, inspiration
  
  Editorial: Classification helps determine optimal clip duration and window sizing.
  """
  t = text.strip().lower()
  w = set(_tokenize(t))
  
  # EDUCATIONAL - explanations, calculations, reasoning chains
  # Signals: numbers, units, conversions, cause-effect
  # These REQUIRE longer duration to maintain informational completeness
  if _is_educational_content(t):
    return "educational"
  
  # HARD ADVICE - imperative, instructional
  # Signals: must, should, don't, focus, remember
  imperative = {"harus", "wajib", "jangan", "pastikan", "ingat", "fokus", "hindari", "lakukan"}
  if w & imperative:
    return "hard_advice"
  
  # WARNING - risk, danger, consequences
  # Signals: danger, risk, problem, mistake, avoid
  warning_signals = {
    "bahaya", "risiko", "resiko", "ancaman", "masalah", "kesalahan", 
    "salah", "fatal", "hancur", "rugi", "gagal", "kacau", "berantakan",
  }
  caution_patterns = [
    r"\bhati-?hati\b",
    r"\bkalau\s+(tidak|nggak|gak)\b",
    r"\bjangan\s+sampai\b",
  ]
  has_warning = (w & warning_signals) or any(re.search(p, t) for p in caution_patterns)
  if has_warning:
    return "warning"
  
  # LESSON LEARNED - past experience, retrospective
  # Signals: I learned, when I, past tense, experience
  lesson_patterns = [
    r"\bsaya\s+(belajar|dapat|dapet)\b",
    r"\bpengalaman\s+saya\b",
    r"\bdulu\s+saya\b",
    r"\bwaktu\s+(itu\s+)?saya\b",
    r"\bpelajaran(nya)?\b",
  ]
  past_tense = {"dulu", "waktu", "pengalaman", "pelajaran", "ternyata"}
  has_lesson = any(re.search(p, t) for p in lesson_patterns) or (w & past_tense and len(w & {"saya", "aku", "gue", "kita"}) > 0)
  if has_lesson:
    return "lesson_learned"
  
  # INSIGHT - understanding, explanation, "what really matters"
  # Signals: key, the point is, actually, the truth is
  insight_signals = {
    "kunci", "kuncinya", "intinya", "sebenarnya", "faktanya", 
    "kenyataannya", "artinya", "poinnya", "alasannya",
  }
  insight_patterns = [
    r"\byang\s+penting\b",
    r"\byang\s+perlu\b",
    r"\bmasalah\s+utama\b",
  ]
  has_insight = (w & insight_signals) or any(re.search(p, t) for p in insight_patterns)
  if has_insight:
    return "insight"
  
  # MOTIVATIONAL - encouragement, belief, inspiration
  # Signals: believe, can do it, keep going, success
  motivational_signals = {
    "semangat", "percaya", "yakin", "pasti", "bisa", "sukses", 
    "berhasil", "terus", "lanjut", "jangan", "menyerah",
  }
  motivational_patterns = [
    r"\bkamu\s+bisa\b",
    r"\bpasti\s+bisa\b",
    r"\bjangan\s+menyerah\b",
  ]
  has_motivational = (w & motivational_signals and len(w & motivational_signals) >= 2) or any(re.search(p, t) for p in motivational_patterns)
  if has_motivational:
    return "motivational"
  
  # Default: insight (most general category for valuable content)
  return "insight"


def _hook_position_score(anchor_index: int, window_left: int, window_right: int, clip_duration: float, opening_strictness: int = 0) -> Tuple[int, Optional[str]]:
  """Score based on where the strongest statement (hook) appears in the clip.
  
  Editorial principle: The best clips have their "money line" early (first 25% of duration).
  Late hooks require viewers to wait too long for payoff.
  
  Returns: (score_modifier, reason)
  """
  if window_right <= window_left:
    return 0, None
  
  window_span = window_right - window_left + 1
  anchor_position = anchor_index - window_left
  position_ratio = anchor_position / window_span
  
  # Learning: if users often mark weak openings, tighten thresholds slightly.
  strict = 1 if opening_strictness >= 1 else 0
  ideal_thr = 0.20 if strict else 0.25
  good_thr = 0.35 if strict else 0.40
  late_thr = 0.55 if strict else 0.60

  # Ideal: hook early
  if position_ratio <= ideal_thr:
    return (4 if strict else 3), "Hook di awal (ideal)"
  # Good: hook in early-mid
  elif position_ratio <= good_thr:
    return 1, "Hook di early-mid"
  # Acceptable: hook in middle
  elif position_ratio <= late_thr:
    return 0, None
  # Poor: hook appears late
  else:
    return (-3 if strict else -2), "Hook terlambat"


def _calculate_core_editorial_pass(text: str) -> Tuple[bool, int]:
  """Determine if text passes core editorial requirements.
  
  Core requirements (must have at least ONE):
  - Strong declarative claim
  - Assertive/instructional tone
  - High meaning density with core keywords
  - Recognized punchline
  - Complete educational explanation (premise + reasoning + conclusion)
  
  Returns: (passes, core_score)
  """
  words = _tokenize(text)
  if not words:
    return False, 0
  
  core_score = 0
  
  # 0. EDUCATIONAL CONTENT CHECK (NEW!)
  # Educational content has different signals - it's valuable because of
  # complete informational structure, not editorial keywords
  if _is_educational_content(text):
    # Check if it's a complete educational explanation (not just isolated numbers)
    # Simple heuristic: needs reasoning markers + sufficient length
    reasoning_markers = {"artinya", "berarti", "jadi", "makanya", "hasilnya", "itulah", "begitulah"}
    has_reasoning = any(w in reasoning_markers for w in words)
    
    # Complete educational content gets immediate pass
    if has_reasoning and len(words) >= 10:
      core_score += 5  # Strong educational signal
    elif len(words) >= 8:
      core_score += 3  # Moderate educational signal
  
  # 1. Declarative claim check
  declarative_patterns = [
    r"\bkunci(nya)?\s+(itu\s+)?adalah\b",
    r"\bintinya\s+(itu\s+)?\b",
    r"\bartinya\s+\b",
    r"\bfakta(nya)?\s+(itu\s+)?\b",
    r"\bkenyataan(nya)?\s+(itu\s+)?\b",
    r"\bmasalah(nya)?\s+(itu\s+)?adalah\b",
    r"\bsolusi(nya)?\s+(itu\s+)?\b",
    r"\brahasia(nya)?\s+\w+\s+(itu|adalah)\b",  # "rahasia sukses itu/adalah"
    r"\bsebenarnya\s+\w+\s+(itu|adalah)\b",     # "sebenarnya X itu/adalah"
  ]
  is_declarative = any(re.search(pat, text.lower()) for pat in declarative_patterns)
  if is_declarative:
    core_score += 4
  
  # 2. Assertive/instructional check
  imperative_markers = {"harus", "wajib", "jangan", "pastikan", "ingat", "fokus", "hindari"}
  if any(w in imperative_markers for w in words):
    core_score += 3
  
  # 3. Core keywords check - opener words signal strong editorial anchor
  core_openers = {"masalahnya", "kuncinya", "intinya", "sebenarnya", "faktanya", "kenyataannya", "solusinya"}
  text_lower = text.lower()
  has_opener = any(text_lower.startswith(opener) for opener in core_openers)
  if has_opener:
    core_score += 3
  
  # 4. Core keywords check (general)
  core_keywords = {
    "penting", "kunci", "kuncinya", "masalah", "solusi", "kesalahan",
    "intinya", "sebenarnya", "faktanya", "kenyataannya", "rahasia",
  }
  if any(w in core_keywords for w in words):
    core_score += 2
  
  # 5. Punchline check
  if _is_punchline(text):
    core_score += 3
  
  # 6. Meaning density check
  filler, stop = _filler_and_stopwords()
  content_words = [w for w in words if (w not in filler and w not in stop)]
  content_ratio = len(content_words) / len(words) if words else 0
  if content_ratio >= 0.55 and len(content_words) >= 5:
    core_score += 2
  
  # Pass threshold: must have core_score >= 3
  # This ensures at least one strong editorial signal is present
  passes = core_score >= 3
  return passes, core_score


def _merge_chunks(transcript: List[Dict], min_duration: float = 30.0, max_duration: float = 90.0) -> List[Dict]:
  # Merge consecutive segments into chunks within [min_duration, max_duration]
  n = len(transcript)
  chunks: List[Dict] = []
  i = 0
  while i < n:
    j = i
    total = 0.0
    # Reach minimum duration
    while j < n and total < min_duration:
      total += float(transcript[j]["duration"])  # type: ignore
      j += 1
    # Extend up to max duration
    while j < n and total < max_duration:
      next_d = float(transcript[j]["duration"])  # type: ignore
      if total + next_d > max_duration:
        break
      total += next_d
      j += 1

    if j <= i:
      # Safety fallback
      j = i + 1

    start = float(transcript[i]["start"])  # type: ignore
    last = transcript[j - 1]
    end = float(last["start"]) + float(last["duration"])  # type: ignore
    text = " ".join(str(seg["text"]) for seg in transcript[i:j])

    chunks.append({
      "start": start,
      "end": end,
      "text": text,
      "_span": (i, j),  # internal for window stride
    })

    stride = max(1, (j - i) // 2)
    i += stride

  return chunks


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
  # Intersection-over-union (IoU) for time ranges
  inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
  union = max(1e-9, (a_end - a_start) + (b_end - b_start) - inter)
  return inter / union


def _build_window_around(
  transcript: List[Dict],
  center_index: int,
  min_duration: float,
  max_duration: float,
  pre_roll: float,
  post_roll: float,
  max_pre_context: float = 7.0,
  max_post_context: float = 10.0,
) -> Tuple[int, int, float, float]:
  """Build a time window centered around the strongest segment (anchor).

  Editorial principle:
  - The anchor (strongest statement) must be DOMINANT in the clip
  - Aggressively trim filler BEFORE the anchor
  - Ensure anchor appears in first 25% of clip duration when possible
  - Don't let weak pre-context dilute the hook

  Returns (left_index, right_index, start_seconds, end_seconds).
  """
  n = len(transcript)
  center_index = max(0, min(n - 1, center_index))
  l = r = center_index

  def seg_start(i: int) -> float:
    return float(transcript[i]["start"])  # type: ignore

  def seg_end(i: int) -> float:
    return float(transcript[i]["start"]) + float(transcript[i]["duration"])  # type: ignore

  anchor_start = seg_start(center_index)
  anchor_end = seg_end(center_index)
  anchor_duration = anchor_end - anchor_start

  # PASS 1: Expand RIGHT FIRST to meet min_duration
  # CRITICAL: Prioritize post-context (explanation/resolution) over pre-context
  # This ensures clips contain reasoning, not just claims
  while (seg_end(r) - seg_start(l)) < min_duration:
    cur_dur = seg_end(r) - seg_start(l)
    
    # Try RIGHT first (strongly preferred)
    can_go_right = r < n - 1 and (seg_end(r + 1) - anchor_end) <= max_post_context
    if can_go_right:
      r += 1
      continue
    
    # Only go LEFT if RIGHT is exhausted
    can_go_left = l > 0 and (anchor_start - seg_start(l - 1)) <= max_pre_context
    if can_go_left:
      l -= 1
      continue
    
    # Cannot expand further
    break

  # Second pass: opportunistically expand to max_duration if segments are strong
  # But don't add weak filler just to hit max_duration
  filler, _stop = _filler_and_stopwords()
  
  def is_valuable_segment(i: int) -> bool:
    """Check if segment adds editorial value (not just filler)."""
    seg_text = str(transcript[i].get("text", "")).strip()
    if not seg_text:
      return False
    # Quick check: does it pass core editorial requirements?
    passes, _ = _calculate_core_editorial_pass(seg_text)
    if passes:
      return True
    # Or at least not filler-heavy
    toks = _tokenize(seg_text)
    if not toks:
      return False
    filler_ratio = sum(1 for w in toks if w in filler) / len(toks)
    return filler_ratio < 0.30

  # Check if this is educational content - needs more aggressive expansion
  anchor_text = str(transcript[center_index].get("text", ""))
  is_educational_anchor = _is_educational_content(anchor_text) or _classify(anchor_text) == "educational"
  
  expanded = True
  expansion_count = 0
  max_expansions = 6 if is_educational_anchor else 4  # More iterations for educational
  
  while expanded and expansion_count < max_expansions:
    expanded = False
    expansion_count += 1
    cur = seg_end(r) - seg_start(l)
    if cur >= max_duration:
      break

    # Re-check if current window is educational (not just anchor)
    # This catches question->calculation patterns
    window_text = " ".join(str(transcript[i].get("text", "")) for i in range(l, r + 1))
    is_educational_window = is_educational_anchor or _is_educational_content(window_text)

    # Try expanding right (preferred to keep anchor early)
    if r < n - 1:
      if (seg_end(r + 1) - anchor_end) <= max_post_context:
        next_cur = seg_end(r + 1) - seg_start(l)
        # For educational: be MORE LENIENT - context segments are critical
        # Don't require every segment to be "valuable" individually
        should_expand_right = (is_educational_window or is_valuable_segment(r + 1))
        if next_cur <= max_duration and should_expand_right:
          r += 1
          expanded = True
          continue
    
    # Try expanding left (only if segment is valuable or educational)
    if l > 0:
      if (anchor_start - seg_start(l - 1)) <= max_pre_context:
        next_cur = seg_end(r) - seg_start(l - 1)
        should_expand_left = (is_educational_window or is_valuable_segment(l - 1))
        if next_cur <= max_duration and should_expand_left:
          l -= 1
          expanded = True

  # Insight Completion Layer (window-building assist):
  # If the anchor is a punchline but the window lacks reasoning/takeaway,
  # FORCE right-expansion until we find a reason OR implication marker,
  # or we hit max_duration. This is more important than keeping the anchor early.
  anchor_is_punchline = _is_punchline(anchor_text)
  if anchor_is_punchline:
    window_text = " ".join(str(transcript[i].get("text", "")) for i in range(l, r + 1))
    claim_present, reason_present, implication_present = _insight_components(window_text)

    # For punchlines, do not allow claim-only clips when we can fetch post-context.
    if claim_present and not (reason_present or implication_present):
      force_steps = 0
      max_force_steps = 12
      while r < n - 1 and force_steps < max_force_steps:
        new_dur = seg_end(r + 1) - seg_start(l)
        if new_dur > max_duration:
          break

        r += 1
        force_steps += 1

        window_text = " ".join(str(transcript[i].get("text", "")) for i in range(l, r + 1))
        _, reason_present, implication_present = _insight_components(window_text)
        if reason_present or implication_present:
          break

  # Third pass: AGGRESSIVELY trim filler from the start
  # Editorial: Don't let weak setup dilute the hook
  filler_starts = ("jadi", "nah", "oke", "ok", "ya", "eee", "eh", "em", "hmm", "terus", "lalu")
  
  def is_weak_lead(i: int) -> bool:
    """Identify segments that are weak lead-ins (should be trimmed)."""
    seg_text = str(transcript[i].get("text", "")).strip().lower()
    if not seg_text:
      return True
    
    # Don't trim if it's a strong statement
    passes, core = _calculate_core_editorial_pass(seg_text)
    if passes or core >= 3:
      return False
    
    # Trim if it's a hanging start (conjunction/connector)
    if _is_hanging_start(seg_text):
      return True
    
    # Trim connector/filler starts
    if seg_text.startswith(filler_starts):
      return True
    
    # Trim very filler-heavy segments
    toks = _tokenize(seg_text)
    if len(toks) <= 2:
      return True
    filler_ratio = sum(1 for w in toks if w in filler) / max(1, len(toks))
    return filler_ratio >= 0.35

  # Trim from left while maintaining min_duration
  while l < r and is_weak_lead(l):
    if (seg_end(r) - seg_start(l + 1)) >= min_duration:
      l += 1
    else:
      break

  # Fourth pass: trim weak endings
  filler_ends = {"ya", "yah", "kan", "dong", "deh", "sih", "gitu", "gini", "nih"}
  
  def is_weak_tail(i: int) -> bool:
    """Identify segments that are weak endings (should be trimmed)."""
    seg_text = str(transcript[i].get("text", "")).strip().lower()
    if not seg_text:
      return True
    
    # Don't trim if it's a strong statement
    passes, core = _calculate_core_editorial_pass(seg_text)
    if passes or core >= 3:
      return False
    
    toks = _tokenize(seg_text)
    if len(toks) <= 2:
      return True
    
    # Trim sentences ending with filler particles
    if toks and toks[-1] in filler_ends and len(toks) <= 6:
      return True
    
    filler_ratio = sum(1 for w in toks if w in filler) / max(1, len(toks))
    return filler_ratio >= 0.35

  # Trim from right while maintaining min_duration
  while r > l and is_weak_tail(r):
    if (seg_end(r - 1) - seg_start(l)) >= min_duration:
      r -= 1
    else:
      break

  # Fifth pass: Ensure complete thoughts (no hanging starts/incomplete ends)
  # CRITICAL: Check the entire window text, not just the first segment
  # This catches cases where individual segments might pass but the window starts with hanging connector
  
  # Check window-level hanging start
  window_full_text = " ".join(str(transcript[i].get("text", "")) for i in range(l, r + 1))
  
  # If window starts with hanging connector, aggressively trim left
  if _is_hanging_start(window_full_text):
    # Keep trimming until we don't start with a hanging word OR hit minimum
    max_trim_attempts = 5
    attempts = 0
    while l < r and attempts < max_trim_attempts:
      # Check if we can skip this segment and still have a valid clip
      next_window_text = " ".join(str(transcript[i].get("text", "")) for i in range(l + 1, r + 1))
      next_duration = seg_end(r) - seg_start(l + 1)
      next_segments = r - (l + 1) + 1
      
      # Can skip if:
      # 1. Next window doesn't start with hanging, OR
      # 2. We still meet minimum requirements (duration >= min OR segments >= 2)
      can_skip = (not _is_hanging_start(next_window_text)) or (next_duration >= min_duration and next_segments >= 2)
      
      if can_skip:
        l += 1
        window_full_text = next_window_text
        if not _is_hanging_start(window_full_text):
          break
      else:
        break
      attempts += 1
  
  # Check right boundary - FORCE RESOLUTION EXTENSION
  # NON-NEGOTIABLE: Never end on questions, transitional words, or incomplete phrases
  max_extension_attempts = 8  # Increased from 5 - prioritize completeness
  extension_attempts = 0
  
  while r < n - 1 and extension_attempts < max_extension_attempts:
    window_end_text = str(transcript[r].get("text", ""))
    end_lower = window_end_text.strip().lower()
    
    # HARD RULE: Must extend if ending is incomplete
    needs_extension = _is_incomplete_end(window_end_text)
    
    # HARD RULE: Must extend if ending on question without answer
    if "?" in window_end_text or "hah" in end_lower or "pertanyaannya" in end_lower:
      needs_extension = True
    
    # HARD RULE: Must extend if ending on transitional marker
    # These words signal "more is coming" - MUST include the resolution
    transitional_endings = ("jadi", "artinya", "berarti", "makanya", "jadinya", "sehingga", "maka")
    if any(end_lower.endswith(t) for t in transitional_endings):
      needs_extension = True
    
    # HARD RULE: Educational content ending with numbers needs conclusion
    if _is_educational_content(window_end_text) and any(char.isdigit() for char in window_end_text):
      # Check if we have a conclusion marker in next segment
      if r < n - 1:
        next_text = str(transcript[r + 1].get("text", "")).lower()
        if any(marker in next_text for marker in ["itulah", "begitulah", "jadi", "kesimpulannya"]):
          needs_extension = True
    
    if needs_extension:
      # FORCE extension - go beyond normal post_context if needed for resolution
      # Allow up to 1.5x max_post_context for critical resolution
      extended_post_context = max_post_context * 1.5
      if (seg_end(r + 1) - anchor_end) <= extended_post_context:
        new_dur = seg_end(r + 1) - seg_start(l)
        # Allow going slightly over max_duration for completeness (up to +5s)
        if new_dur <= max_duration + 5.0:
          r += 1
          extension_attempts += 1
        else:
          break
      else:
        break
    else:
      break

  # CRITICAL FIX: Re-check if window is educational or likely to become educational
  # Educational signals: numbers/units OR question-answer patterns OR calculation keywords
  # If educational (or likely), enforce 18s minimum duration
  window_text = " ".join(str(transcript[i].get("text", "")) for i in range(l, r + 1))
  window_lower = window_text.lower()
  
  # Check if window is educational OR has educational signals
  is_or_likely_educational = (
    _is_educational_content(window_text) or
    _classify(window_text) == "educational" or
    # Likely educational signals:
    ("berapa" in window_lower) or  # Question about quantity
    ("hitung" in window_lower) or  # Calculation keyword
    ("kalau" in window_lower and ("maka" in window_lower or "jadi" in window_lower)) or  # If-then reasoning
    ("1" in window_text and any(unit in window_lower for unit in ["juta", "ribu", "persen", "gram", "kilogram", "liter"]))  # Number + unit spotted
  )
  
  if is_or_likely_educational:
    EDUCATIONAL_MIN = 18.0
    current_duration = seg_end(r) - seg_start(l)
    
    # If under 18s, force expansion RIGHT (favor explanations)
    expansion_attempts = 0
    while current_duration < EDUCATIONAL_MIN and r < n - 1 and expansion_attempts < 8:
      # Check post-context limit (use educational profile: 20s)
      post_context_if_expanded = seg_end(r + 1) - anchor_end
      if post_context_if_expanded <= 20.0:  # Educational max_post_context
        new_duration = seg_end(r + 1) - seg_start(l)
        # Allow up to 35s (educational max_duration)
        if new_duration <= 35.0:
          r += 1
          current_duration = seg_end(r) - seg_start(l)
          expansion_attempts += 1
        else:
          break
      else:
        break

  # Calculate final timestamps with roll
  start = max(0.0, seg_start(l) - pre_roll)
  end = seg_end(r) + post_roll
  if end <= start:
    end = start + 1.0

  return l, r, start, end


def detect_highlights(
  transcript: list[dict],
  db_session=None,
  learning_profile: Optional[EditorialLearningProfile] = None,
  editorial_bias: Optional[EditorialBiasProfile] = None,
) -> list[dict]:
  """Detect highlight candidates from transcript using heuristics.

  Editorial Philosophy:
  - Fewer, higher-quality clips (aim for 3-5 clips max)
  - Each clip must center around a strong editorial anchor
  - Hook (strongest statement) must appear early in clip (first 25% ideal)
  - All clips must pass core editorial requirements
  - Clips must be complete thoughts (no half-sentences)
  
  Learning Layer:
  - Adapts scoring based on historical editorial feedback
  - Deterministic bias (no ML/AI)
  - Capped at ±4 points, never overrides core gates
  
  Args:
    transcript: List of segment dicts with start, duration, text
    db_session: Optional database session for loading editorial feedback
  
  Returns list of dicts: {start, end, score, category, reason, transcript}
  """
  if not transcript:
    return []

  # ============================================================
  # LOAD EDITORIAL LEARNING LAYER
  # ============================================================
  # Load historical feedback and compute bias profile
  # This is deterministic and safe (returns zero bias if no data)
  if editorial_bias is None:
    editorial_bias = load_editorial_bias(db_session)

  # ============================================================
  # PREPROCESSING: Clean transcript artifacts
  # ============================================================
  # Clean each segment's text while preserving timing data
  cleaned_transcript = []
  for seg in transcript:
    cleaned_seg = seg.copy()
    original_text = str(seg.get("text", ""))
    cleaned_text = _clean_transcript_text(original_text)
    cleaned_seg["text"] = cleaned_text
    cleaned_seg["_original_text"] = original_text  # Keep for debugging
    cleaned_transcript.append(cleaned_seg)
  
  # Use cleaned transcript for all analysis
  transcript = cleaned_transcript

  # Target: 3-5 high-quality clips (prefer quality over quantity)
  max_results = 5
  min_results = 2 if len(transcript) >= 20 else 1

  # ============================================================
  # STEP 1: Score each segment for anchor selection
  # ============================================================
  seg_scores: List[int] = []
  seg_reasons: List[List[str]] = []
  seg_punch: List[bool] = []
  seg_core_passes: List[bool] = []

  for seg in transcript:
    text = str(seg.get("text", ""))
    score, reasons = _score_text(text)
    is_punch = _is_punchline(text)
    core_pass, _ = _calculate_core_editorial_pass(text)
    
    seg_scores.append(score)
    seg_reasons.append(reasons)
    seg_punch.append(is_punch)
    seg_core_passes.append(core_pass)

  # ============================================================
  # STEP 2: Identify anchor candidates (local peaks + punchlines)
  # ============================================================
  n = len(transcript)
  peak_threshold = _adaptive_peak_threshold(seg_scores)
  peaks: List[int] = []

  # Local peak detection: segment must be locally maximal
  for i in range(n):
    s = seg_scores[i]
    
    # Must pass either: score threshold OR be a punchline OR pass core check
    if s < peak_threshold and not seg_punch[i] and not seg_core_passes[i]:
      continue
    
    left = seg_scores[i - 1] if i > 0 else -10**9
    right = seg_scores[i + 1] if i < n - 1 else -10**9
    
    # Local peak: better than or equal to both neighbors, and strictly better than at least one
    if s >= left and s >= right and (s > left or s > right):
      peaks.append(i)

  # Ensure all punchlines and core-passing segments are considered
  for i in range(n):
    if (seg_punch[i] or seg_core_passes[i]) and i not in peaks:
      peaks.append(i)
  
  # Add educational sequences as anchor candidates
  # These may have low individual scores but form complete explanations
  educational_anchors = _detect_educational_sequences(transcript, min_length=3)
  for i in educational_anchors:
    if i not in peaks:
      peaks.append(i)

  # Fallback: if no peaks found, use top scorers
  if not peaks:
    peaks = sorted(range(n), key=lambda i: seg_scores[i], reverse=True)[:max(6, min_results * 2)]

  # Rank peaks: punchlines first, then core-passing, then by score
  def peak_rank(i: int) -> Tuple[int, int, int]:
    punch = 1 if seg_punch[i] else 0
    core = 1 if seg_core_passes[i] else 0
    return (punch, core, seg_scores[i])
  
  ranked_indices = sorted(peaks, key=peak_rank, reverse=True)

  # ============================================================
  # STEP 3: Build clips around anchors
  # ============================================================
  candidates: List[Dict] = []

  for idx in ranked_indices:
    # Gate: only proceed if anchor has strong editorial value
    anchor_text = str(transcript[idx].get("text", ""))
    core_pass, core_score = _calculate_core_editorial_pass(anchor_text)
    
    # Check if this is an educational sequence anchor
    is_edu_anchor = idx in educational_anchors
    
    # Strict gate: anchor MUST pass core editorial check OR be part of educational sequence
    # Educational anchors may have low individual scores but form complete explanations
    if not core_pass and not seg_punch[idx] and not is_edu_anchor:
      continue


    # Get dynamic window profile based on category (+ optional feedback learning adjustments)
    prof, profile_reasons = _dynamic_window_profile(anchor_text, learning_profile=learning_profile)
    l, r, start, end = _build_window_around(
      transcript,
      idx,
      min_duration=float(prof["min_duration"]),
      max_duration=float(prof["max_duration"]),
      pre_roll=float(prof["pre_roll"]),
      post_roll=float(prof["post_roll"]),
      max_pre_context=float(prof["max_pre_context"]),
      max_post_context=float(prof["max_post_context"]),
    )
    
    # Enforce hard max clip length of 60s
    max_clip_len = 60.0
    if end - start > max_clip_len:
      end = start + max_clip_len

    # Deduplicate: skip if too similar to existing candidate
    if any(_overlap_ratio(start, end, c["start"], c["end"]) >= 0.60 for c in candidates):
      continue

    # ============================================================
    # STEP 4: Quality gates for clip viability
    # ============================================================
    # GATE 1: HARD MINIMUM DURATION ENFORCEMENT
    # Never produce clips shorter than category minimum - short clips confuse audiences
    num_segments = r - l + 1
    clip_duration = end - start
    
    # Determine category-specific hard minimum
    cat = _classify(anchor_text)
    is_punch = _is_punchline(anchor_text)
    
    if is_punch:
      hard_min = 8.0  # Only punchlines can be this short
    elif cat == "educational":
      hard_min = 18.0  # Educational MUST be 18s+ (never allow short explanations)
    elif cat in ["warning", "lesson_learned"]:
      hard_min = 15.0  # Need full story/context
    elif cat == "insight":
      hard_min = 12.0  # Need complete explanation
    else:
      hard_min = 12.0  # Default minimum for all other categories
    
    # HARD REJECT if below minimum
    if clip_duration < hard_min:
      continue
    
    # Additional check: never allow single short segment (too fragmentary)
    if num_segments < 2 and clip_duration < 10.0:
      continue
    
    # GATE 2: Score the complete clip window
    window_text = " ".join(str(transcript[i].get("text", "")) for i in range(l, r + 1))
    
    # Core editorial check for final window
    window_core_pass, window_core_score = _calculate_core_editorial_pass(window_text)
    if not window_core_pass:
      continue  # Strict: final clip must pass core check
    
    # GATE 3: Ensure clip contains claim + support/resolution
    # Check if window has multiple meaningful segments (not just anchor)
    meaningful_segments = 0
    for i in range(l, r + 1):
      seg_text = str(transcript[i].get("text", ""))
      if seg_text and len(_tokenize(seg_text)) >= 3:
        passes, _ = _calculate_core_editorial_pass(seg_text)
        if passes or len(_tokenize(seg_text)) >= 5:
          meaningful_segments += 1
    
    # Require at least 2 meaningful segments (claim + support/resolution)
    # EXCEPTION: Educational content gets a pass if it has >= 3 total segments
    # (educational value comes from complete explanation, not standalone segments)
    is_edu_window = _is_educational_content(window_text)
    if meaningful_segments < 2:
      if is_edu_window and num_segments >= 3:
        # Educational clips with 3+ segments can proceed even if individual segments are weak
        pass
      else:
        continue
    
    # GATE 4: INFORMATIONAL COMPLETENESS (MANDATORY)
    # CRITICAL: Completeness overrides score - never produce incomplete clips
    window_segment_texts = [str(transcript[i].get("text", "")) for i in range(l, r + 1)]
    is_complete, completeness_reason = _check_informational_completeness(window_segment_texts)
    
    # HARD RULE: Educational content MUST be complete
    is_edu_window = _is_educational_content(window_text)
    if is_edu_window and not is_complete:
      # Educational clips without complete information are INVALID
      continue
    
    # For non-educational content, still enforce completeness unless very strong
    if not is_complete and window_core_score < 8:
      # Incomplete clips confuse audiences - reject
      continue
    
    # ADDITIONAL CHECK: Educational clips must have minimum 3 segments
    # Single-segment educational clips are INVALID by definition
    if is_edu_window and num_segments < 3:
      continue
    
    # ADDITIONAL CHECK: Educational clips should be 20s+ (preferred range)
    # If under 20s, it's likely missing context
    if is_edu_window and clip_duration < 20.0 and not is_complete:
      continue

    # GATE 5: INSIGHT COMPLETION LAYER (NEW)
    # Non-educational clips MUST have 2-of-3 insight components:
    # claim + (reason and/or implication). Educational clips are allowed to proceed
    # if informational completeness already passes.
    is_punchline_clip = (seg_punch[idx] or _is_punchline(window_text))
    insight_claim, insight_reason, insight_implication = _insight_components(window_text)
    insight_pass, insight_reasons = _has_insight_structure(window_text)

    # HARD RULE: Punchlines WITHOUT reasoning or takeaway must NOT be output.
    if is_punchline_clip and insight_claim and not (insight_reason or insight_implication):
      continue

    # For non-educational content, reject if insight structure fails.
    # Learning can tighten this further (require insight even for educational).
    insight_required = (not is_edu_window) or (learning_profile is not None and learning_profile.require_insight_all)
    if insight_required and not insight_pass:
      continue
    
    window_score, window_reasons = _score_text(window_text)
    
    # Blend anchor strength + window coherence
    # For educational content, prioritize window score (completeness matters more than individual anchor)
    # For other content, anchor is more important - it's the "money shot"
    is_edu_window = _is_educational_content(window_text)
    if is_edu_window:
      # Educational: window coherence (80%) + anchor (20%)
      # The complete explanation is more valuable than any single segment
      base_score = int(round(0.20 * seg_scores[idx] + 0.80 * window_score))
    else:
      # Standard: anchor (70%) + window (30%)
      base_score = int(round(0.70 * seg_scores[idx] + 0.30 * window_score))

    # ============================================================
    # STEP 5: Apply bonuses and penalties
    # ============================================================
    
    # Punchline bonus - editorial gold
    if seg_punch[idx] or _is_punchline(window_text):
      base_score += 3

    # Hook position bonus/penalty - ensure strong line appears early
    clip_duration = end - start
    opening_strictness = learning_profile.opening_strictness if learning_profile is not None else 0
    hook_bonus, hook_reason = _hook_position_score(idx, l, r, clip_duration, opening_strictness=opening_strictness)
    base_score += hook_bonus

    # Duration bonuses - prefer target ranges by category
    # CRITICAL: Never penalize long clips for educational/explanation content
    duration_bonus = 0
    cat = _classify(anchor_text)
    is_edu = cat == "educational"
    
    if seg_punch[idx] or _is_punchline(window_text):
      # Punchlines: 8-18s ideal (only category optimized for brevity)
      if 8.0 <= clip_duration <= 18.0:
        duration_bonus += 2
      elif clip_duration < 8.0 or clip_duration > 25.0:
        duration_bonus -= 2
    elif is_edu:
      # Educational: 20-35s ideal (needs complete premise + reasoning + conclusion)
      # LONGER IS BETTER for understanding - never penalize length
      if 20.0 <= clip_duration <= 35.0:
        duration_bonus += 4  # Increased bonus - reward proper length
      elif 18.0 <= clip_duration < 20.0:
        duration_bonus += 2  # Still good, slightly short of ideal
      elif clip_duration < 18.0:
        # Should never reach here due to GATE 1, but penalize if it does
        duration_bonus -= 4  # Too short for educational content
      # NO PENALTY for clips over 35s if complete (longer explanations are fine)
      # Only penalize if exceeding absolute maximum (60s)
      if clip_duration > 60.0:
        duration_bonus -= 3
      # EXTRA BONUS for complete educational clips in ideal range
      if is_complete and 20.0 <= clip_duration <= 35.0:
        duration_bonus += 2  # Total +6 for ideal complete educational clip
    elif cat == "hard_advice":
      # Hard advice: 12-25s ideal
      if 12.0 <= clip_duration <= 25.0:
        duration_bonus += 2
      elif clip_duration < 10.0 or clip_duration > 30.0:
        duration_bonus -= 1
    elif cat == "lesson_learned":
      # Lessons: 18-35s ideal (needs story)
      if 18.0 <= clip_duration <= 35.0:
        duration_bonus += 2
      elif clip_duration < 15.0:
        duration_bonus -= 1
    else:
      # insight, warning, motivational: 15-30s ideal
      if 15.0 <= clip_duration <= 30.0:
        duration_bonus += 2
      elif clip_duration < 12.0 or clip_duration > 40.0:
        duration_bonus -= 1

    base_score += duration_bonus
    
    # Informational completeness bonus
    if is_complete:
      base_score += 2
      if is_edu:
        base_score += 1  # Extra bonus for complete educational clips

    # Claim completeness bonus
    claim_words = {"kunci", "penting", "masalah", "solusi", "harus", "jangan", "intinya"}
    support_words = {"karena", "sebab", "soalnya", "makanya", "jadi", "biar", "supaya"}
    window_words = _tokenize(window_text.lower())
    has_claim = any(w in claim_words for w in window_words)
    has_support = any(w in support_words for w in window_words)
    if has_claim and has_support:
      base_score += 1

    # Insight scoring bonus: reward full structure (claim + reason + implication)
    has_full_insight = insight_claim and insight_reason and insight_implication
    if has_full_insight:
      base_score += 2

    # ============================================================
    # APPLY EDITORIAL LEARNING BIAS
    # ============================================================
    # Apply historical feedback bias to score
    # This is additive and capped at ±4 points
    # IMPORTANT: Bias is applied BEFORE validation gates
    # Gates will still reject clips that fail core requirements
    
    # Compute hook position for bias (anchor position relative to clip)
    anchor_start_time = float(transcript[idx]["start"])
    anchor_offset_in_clip = anchor_start_time - start
    hook_position = anchor_offset_in_clip / clip_duration if clip_duration > 0 else 0.5
    
    # Prepare bias reasons list
    bias_reasons: List[str] = []
    
    # Apply bias (deterministic, pure function)
    final_score = apply_editorial_bias(
      base_score=base_score,
      category=cat,
      clip_duration=clip_duration,
      is_punchline=is_punchline_clip,
      hook_position=hook_position,
      bias_profile=editorial_bias,
      reasons=bias_reasons,
      learning_profile=learning_profile,
      insight_pass=insight_pass,
      full_insight=has_full_insight,
    )

    # ============================================================
    # STEP 6: FINAL VALIDATION CHECK (COMPREHENSIVE)
    # ============================================================
    # CRITICAL: These gates CANNOT be bypassed by bias
    # Bias only nudges scores; gates enforce absolute requirements
    # Before accepting clip, verify ALL requirements are met
    # This is the last line of defense against incomplete/confusing clips
    
    # Check 1: Minimum score threshold
    if is_punchline_clip:
      min_score = 4
    elif is_edu:
      min_score = 5
    else:
      min_score = 6
    
    if final_score < min_score:
      continue
    
    # Check 2: Category-specific hard minimum duration (CRITICAL)
    if is_edu and clip_duration < 18.0:
      continue  # Should never happen, but enforce
    elif not is_punch and clip_duration < 12.0:
      continue  # Non-punchline clips must be 12s+
    
    # Check 3: Informational completeness (RE-CHECK)
    if not is_complete and (is_edu or window_core_score < 8):
      continue  # Incomplete clips are invalid
    
    # Check 4: Never end on hanging transitions/questions (RE-CHECK)
    final_segment_text = str(transcript[r].get("text", "")).strip().lower()
    hanging_endings = ("jadi", "artinya", "berarti", "makanya", "pertanyaannya")
    if any(final_segment_text.endswith(h) for h in hanging_endings) or "?" in final_segment_text:
      continue  # Clip ends mid-thought - invalid
    
    # Check 5: Educational clips must have 3+ segments (multi-segment requirement)
    if is_edu and num_segments < 3:
      continue  # Single-segment educational clips are invalid
    
    # Check 6: All clips must have claim + support (minimum 2 meaningful segments)
    # Already checked in GATE 3, but re-verify
    if meaningful_segments < 2 and not (is_edu_window and num_segments >= 3):
      continue

    # ============================================================
    # STEP 7: Build reason string
    # ============================================================
    reason_bits: List[str] = []
    
    if is_punchline_clip:
      reason_bits.append("Punchline")

    if profile_reasons:
      reason_bits.extend(profile_reasons)

    # Insight explanation trace
    if has_full_insight:
      reason_bits.append("Insight lengkap")
    if insight_reasons:
      # Keep it short and high-signal
      reason_bits.extend(insight_reasons[:1])
    
    if hook_reason:
      reason_bits.append(hook_reason)
    
    if duration_bonus > 0:
      reason_bits.append(f"Durasi optimal ({clip_duration:.0f}s)")
    
    # Add editorial bias explanations (if any)
    if bias_reasons:
      reason_bits.extend(bias_reasons)
    
    # Add compressed window reasons
    reason_bits.extend(window_reasons[:2])  # Keep top 2 window reasons
    
    reason = "; ".join(_compress_reasons(reason_bits, limit=5)) if reason_bits else "Editorial heuristic"

    # Add to candidates
    candidates.append({
      "start": start,
      "end": end,
      "score": final_score,
      "category": cat,
      "reason": reason,
      "transcript": window_text,
    })

    # Stop if we have enough high-quality candidates
    if len(candidates) >= max_results:
      break

  # ============================================================
  # STEP 8: Backfill (SAFER - must pass core checks)
  # ============================================================
  if len(candidates) < min_results:
    # Consider next-best segments but with STRICT core requirement
    fallback_order = sorted(
      range(n), 
      key=lambda i: (1 if seg_core_passes[i] else 0, seg_scores[i]), 
      reverse=True
    )
    
    for idx in fallback_order:
      if len(candidates) >= min_results:
        break
      
      # STRICT: backfill must pass core check
      if not seg_core_passes[idx]:
        continue
      
      if seg_scores[idx] < 3:
        continue
      
      # Build window
      anchor_text = str(transcript[idx].get("text", ""))
      prof, profile_reasons = _dynamic_window_profile(anchor_text, learning_profile=learning_profile)
      l, r, start, end = _build_window_around(
        transcript,
        idx,
        min_duration=float(prof["min_duration"]),
        max_duration=float(prof["max_duration"]),
        pre_roll=float(prof["pre_roll"]),
        post_roll=float(prof["post_roll"]),
        max_pre_context=float(prof["max_pre_context"]),
        max_post_context=float(prof["max_post_context"]),
      )
      
      # Cap at 60s
      if end - start > 60.0:
        end = start + 60.0
      
      # Skip if overlaps existing
      if any(_overlap_ratio(start, end, c["start"], c["end"]) >= 0.60 for c in candidates):
        continue
      
      # Final window must ALSO pass core check
      window_text = " ".join(str(transcript[i].get("text", "")) for i in range(l, r + 1))
      window_core_pass, _ = _calculate_core_editorial_pass(window_text)
      if not window_core_pass:
        continue

      # Insight Completion Layer (same rules as primary path)
      is_edu_window = _is_educational_content(window_text)
      is_punchline_clip = _is_punchline(window_text)
      insight_claim, insight_reason, insight_implication = _insight_components(window_text)
      insight_pass, insight_reasons = _has_insight_structure(window_text)

      if is_punchline_clip and insight_claim and not (insight_reason or insight_implication):
        continue
      insight_required = (not is_edu_window) or (learning_profile is not None and learning_profile.require_insight_all)
      if insight_required and not insight_pass:
        continue
      
      window_score, _ = _score_text(window_text)
      score = int(round(0.70 * seg_scores[idx] + 0.30 * window_score))

      if insight_claim and insight_reason and insight_implication:
        score += 2
      
      candidates.append({
        "start": start,
        "end": end,
        "score": score,
        "category": _classify(window_text),
        "reason": "Backfill berkualitas (lulus core check)" + ("; " + "; ".join(profile_reasons) if profile_reasons else "") + ("; Insight lengkap" if (insight_claim and insight_reason and insight_implication) else ""),
        "transcript": window_text,
      })

  # ============================================================
  # STEP 9: Sort and filter final results
  # ============================================================
  # Sort by score (descending), then by duration (ascending - prefer shorter)
  candidates.sort(key=lambda x: (x["score"], -(x["end"] - x["start"])), reverse=True)

  # Dynamic quality cutoff: keep only top-tier clips
  if candidates:
    best_score = candidates[0]["score"]
    cutoff = max(6, best_score - 3)  # Allow some variance but maintain quality
    filtered = [c for c in candidates if c["score"] >= cutoff]
    
    # Don't over-filter into too few clips
    if len(filtered) >= min_results:
      candidates = filtered
    elif len(candidates) >= min_results:
      # Keep at least min_results even if below cutoff
      candidates = candidates[:min_results]

  return candidates[:max_results]

# Improve highlight detection by:
# - Merging consecutive transcript segments
# - Combine segments until duration reaches 30–90 seconds
# - Score merged chunk instead of single sentence
# - Preserve original timestamps

# Do NOT:
# - Call any external API
# - Modify transcript data in-place
# - Hardcode timestamps
# - Return only one highlight
