"""
Context:
- This module analyzes YouTube transcript segments
- Transcript input format:
  [
    { "start": float, "duration": float, "text": str }
  ]

Task:
- Implement a highlight detection function WITHOUT using any AI API
- Use heuristic-based scoring

Highlight definition:
- A highlight is a segment that:
  - Can stand alone
  - Contains strong statement, insight, advice, or emotion
  - Is suitable for short-form content

Scoring rules (example, feel free to adjust):
- +2 if sentence length between 8–25 words
- +2 if contains strong keywords (e.g. "penting", "kunci", "masalah", "harus")
- +2 if starts with emphasis words (e.g. "jadi", "intinya", "yang paling")
- +1 if contains first-person insight ("saya", "kita")
- -2 if sentence is a question
- -3 if too short (<5 words) or too long (>30 words)

Output:
- List of highlight candidates
- Each highlight must include:
  - start
  - end
  - score
  - category (e.g. insight, motivational, explanation)
  - reason (short explanation)

Rules:
- Do NOT modify original transcript
- Do NOT use external libraries
- Keep code readable and testable
"""
import re
from typing import List, Dict, Tuple, Optional


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
  """High-precision punchline detector.

  Editorial intent: pick standalone, clip-ready takes (hard advice/opinion/emotional spike)
  even if not fully explained.
  """
  t = (text or "").strip().lower()
  if not t:
    return False

  words = _tokenize(t)
  n = len(words)
  if n < 5 or n > 12:
    return False

  strong_openers = (
    "masalahnya", "kenyataannya", "faktanya", "intinya", "kuncinya",
  )
  absolute_language = {
    "selalu", "kebanyakan", "semua", "pasti", "cuma", "hanya", "satu-satunya",
    "never", "nggak", "tidak", "gak", "ga",
  }
  polar_phrases = (
    "tidak pernah", "nggak pernah", "gak pernah",
  )
  punch_verbs = {"harus", "wajib", "jangan", "berhenti", "fokus", "ingat"}

  has_opener = t.startswith(strong_openers)
  has_absolute = any(w in absolute_language for w in words) or any(p in t for p in polar_phrases)
  has_take = any(w in punch_verbs for w in words)
  has_emotion = _emotional_signal_score(t)[0] > 0

  # High precision: require a short statement AND at least one punchy signal.
  return has_opener or has_absolute or has_take or has_emotion


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


def _claim_completeness_bonus(text: str) -> Tuple[int, Optional[str]]:
  """Small bonus if a clip contains a claim + a brief support marker.

  Why: human-curated clips often include a point + a quick justification.
  """
  t = text.strip().lower()
  toks = _tokenize(t)
  claim = {"kunci", "kuncinya", "intinya", "penting", "masalah", "solusi", "harus", "wajib", "jangan"}
  support = {"karena", "sebab", "soalnya", "makanya", "jadi", "biar", "supaya"}
  has_claim = any(w in claim for w in toks) or any(re.search(p, t) for p in [r"\bkunci(nya)?\b", r"\bintinya\b", r"\bmasalah\b", r"\bsolusi\b"])
  has_support = any(w in support for w in toks)
  if has_claim and has_support:
    return 1, "Claim+support"
  return 0, None


def _window_emotion_bonus(transcript: List[Dict], l: int, r: int) -> Tuple[int, Optional[str]]:
  """Small bonus if emotion appears across multiple segments inside the window.

  Why: sustained emotion/urgency tends to produce better clips.
  """
  hits = 0
  for i in range(l, r + 1):
    em_score, _ = _emotional_signal_score(str(transcript[i].get("text", "")))
    if em_score > 0:
      hits += 1
      if hits >= 2:
        return 1, "Emotion sustained"
  return 0, None


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
    "Punchline",
    "Pernyataan deklaratif kuat",
    "Claim+support",
    "Meaning density tinggi",
    "Sinyal urgensi/risiko",
    "Sinyal gagal/frustrasi",
    "Sinyal sukses/lega",
    "Nada tegas",
    "Durasi",
    "Mengandung kata kunci kuat",
    "Panjang kalimat ideal (8–25 kata)",
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


def _clamp_clip_length(start: float, end: float, max_len: float = 60.0) -> Tuple[float, float]:
  """Ensure clip length never exceeds max_len seconds."""
  if end <= start:
    return start, start + 1.0
  if (end - start) > max_len:
    return start, start + max_len
  return start, end


def _dynamic_window_profile(anchor_text: str) -> Dict[str, float]:
  """Pick window sizing based on the anchor's 'topic' (category) + punchiness.

  Editorial intent:
  - punchline: short & tight
  - explanation: can breathe (but still <= 60s)
  - insight/motivational: medium
  """
  if _is_punchline(anchor_text):
    return {
      "min_duration": 10.0,
      "max_duration": 18.0,
      "pre_roll": 0.7,
      "post_roll": 1.3,
      "max_pre_context": 5.0,
      "max_post_context": 6.0,
    }

  cat = _classify(anchor_text)
  if cat == "explanation":
    return {
      "min_duration": 22.0,
      "max_duration": 55.0,
      "pre_roll": 1.0,
      "post_roll": 1.6,
      "max_pre_context": 10.0,
      "max_post_context": 15.0,
    }
  if cat == "insight":
    return {
      "min_duration": 16.0,
      "max_duration": 40.0,
      "pre_roll": 0.9,
      "post_roll": 1.5,
      "max_pre_context": 8.0,
      "max_post_context": 12.0,
    }
  # motivational / general
  return {
    "min_duration": 14.0,
    "max_duration": 35.0,
    "pre_roll": 0.9,
    "post_roll": 1.5,
    "max_pre_context": 7.0,
    "max_post_context": 10.0,
  }


def _meaning_density_score(text: str) -> Tuple[int, List[str]]:
  words = _tokenize(text)
  if not words:
    return -3, ["Tidak ada kata"]

  filler, stop = _filler_and_stopwords()
  total = len(words)
  filler_count = sum(1 for w in words if w in filler)
  stop_count = sum(1 for w in words if w in stop)
  content_words = [w for w in words if (w not in filler and w not in stop)]

  content = len(content_words)
  filler_ratio = filler_count / total
  content_ratio = content / total
  unique_content = len(set(content_words))
  unique_ratio = (unique_content / max(1, content))

  score = 0
  reasons: List[str] = []

  # Penalize filler-heavy text (low meaning density)
  if filler_ratio >= 0.35:
    score -= 3
    reasons.append("Banyak filler")
  elif filler_ratio >= 0.22:
    score -= 2
    reasons.append("Cukup banyak filler")

  # Reward content density and lexical variety (a proxy for information density)
  if content_ratio >= 0.55 and content >= 5:
    score += 2
    reasons.append("Meaning density tinggi")
  elif content_ratio >= 0.45 and content >= 4:
    score += 1
    reasons.append("Meaning density cukup")

  if unique_ratio >= 0.75 and content >= 6:
    score += 1
    reasons.append("Konten variatif")

  # Very stopword-dominant sentences tend to be low value.
  # Guard: don't over-penalize strong declarative claims.
  decl_score, _ = _statement_quality_score(text)
  if stop_count / total >= 0.70 and decl_score <= 0:
    score -= 2
    reasons.append("Terlalu banyak stopword")

  return score, reasons


def _statement_quality_score(text: str) -> Tuple[int, List[str]]:
  t = text.strip().lower()
  words = _tokenize(t)
  score = 0
  reasons: List[str] = []

  # Strong declarative patterns (editor-friendly)
  declarative_patterns = [
    r"\bkunci(nya)?\s+(itu\s+)?adalah\b",
    r"\bintinya\s+(itu\s+)?\b",
    r"\bartinya\s+\b",
    r"\byang\s+terjadi\s+(itu\s+)?adalah\b",
    r"\bpoin(nya)?\s+(itu\s+)?\b",
    r"\bpelajaran(nya)?\s+(itu\s+)?\b",
    r"\bmasalah\s+utamanya\s+(itu\s+)?\b",
    r"\bsolusi(nya)?\s+(itu\s+)?\b",
  ]
  is_declarative = any(re.search(pat, t) for pat in declarative_patterns)
  if is_declarative:
    score += 3
    reasons.append("Pernyataan deklaratif kuat")

  # Penalize storytelling fragments / connectors without a clear claim
  storytelling_markers = {
    "waktu", "dulu", "pas", "kemarin", "terus", "lalu", "kemudian",
    "jadi", "nah", "oke",
  }
  has_story = any(w in storytelling_markers for w in words)
  has_claim = any(w in {"kunci", "penting", "masalah", "solusi", "harus", "jangan", "intinya", "artinya"} for w in words)
  # Guard: if declarative, avoid storytelling penalty.
  if (not is_declarative) and has_story and not has_claim and len(words) >= 8:
    score -= 2
    reasons.append("Cenderung storytelling/filler")

  # Reward assertive modality (signals a claim, advice, or lesson)
  if any(w in {"harus", "wajib", "jangan", "pastikan", "ingat", "fokus"} for w in words):
    score += 1
    reasons.append("Nada tegas")

  return score, reasons


def _emotional_signal_score(text: str) -> Tuple[int, List[str]]:
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
  words = _tokenize(text)
  n = len(words)
  score = 0
  reasons: List[str] = []

  is_punch = _is_punchline(text)

  strong_keywords = {
    "penting", "kunci", "masalah", "solusi", "harus", "wajib",
    "jangan", "selalu", "fokus", "target", "tujuan", "nilai",
    "sukses", "gagal", "kesalahan", "intinya", "sebenarnya",
  }
  emphasis_starts = [
    "jadi", "intinya", "yang paling", "ingat", "sebenarnya", "singkatnya",
  ]
  first_person = {"saya", "aku", "kita", "kami", "gue"}
  question_starts = {"apa", "mengapa", "kenapa", "bagaimana", "kapan", "dimana", "berapa"}

  # Length-based scoring
  if 8 <= n <= 25:
    score += 2
    reasons.append("Panjang kalimat ideal (8–25 kata)")
  elif (n < 5 or n > 30) and (not is_punch):
    score -= 3
    reasons.append("Terlalu pendek/terlalu panjang")

  # Editorial: short punchlines should not be penalized.
  if is_punch and 5 <= n <= 12:
    score += 1
    reasons.append("Punchline")

  # Strong keywords
  if any(w in strong_keywords for w in words):
    score += 2
    reasons.append("Mengandung kata kunci kuat")

  # Emphasis start
  lowered = text.strip().lower()
  if any(lowered.startswith(prefix) for prefix in emphasis_starts):
    score += 2
    reasons.append("Awalan penekanan")

  # First-person insight
  if any(w in first_person for w in words):
    score += 1
    reasons.append("Insight orang pertama")

  # Question penalty
  if "?" in text or any(lowered.startswith(q) for q in question_starts):
    score -= 2
    reasons.append("Kalimat tanya")

  # Meaning density (concise, information-dense statements win)
  md_score, md_reasons = _meaning_density_score(text)
  score += md_score
  reasons.extend(md_reasons)

  # Sentence quality (strong declarative claims win)
  sq_score, sq_reasons = _statement_quality_score(text)
  score += sq_score
  reasons.extend(sq_reasons)

  # Emotional signal (highlights often have emotion/urgency)
  em_score, em_reasons = _emotional_signal_score(text)
  score += em_score
  reasons.extend(em_reasons)

  # Compress reasons for readability.
  reasons = _compress_reasons(reasons, limit=3)
  return score, reasons


def _classify(text: str) -> str:
  w = set(_tokenize(text))
  if w & {"harus", "jangan", "semangat", "percaya", "tujuan", "sukses"}:
    return "motivational"
  if w & {"penting", "kunci", "masalah", "solusi", "kesalahan", "intinya", "sebenarnya"}:
    return "insight"
  if w & {"contoh", "misalnya", "karena", "sebab", "proses", "langkah", "artinya"}:
    return "explanation"
  return "general"


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
  """Build a time window around a center transcript segment.

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

  # Expand to satisfy min_duration, but keep the clip close to the anchor.
  while l > 0 and (seg_end(r) - seg_start(l)) < min_duration:
    # Don't pull too far before the anchor.
    if (anchor_start - seg_start(l - 1)) > max_pre_context:
      break
    l -= 1

  while r < n - 1 and (seg_end(r) - seg_start(l)) < min_duration:
    # Don't push too far after the anchor.
    if (seg_end(r + 1) - anchor_end) > max_post_context:
      break
    r += 1

  # Optionally expand further but keep under max_duration.
  expanded = True
  while expanded:
    expanded = False
    cur = seg_end(r) - seg_start(l)
    if cur >= max_duration:
      break

    # Try add left then right if it fits, still respecting anchor proximity.
    if l > 0 and (anchor_start - seg_start(l - 1)) <= max_pre_context:
      next_cur = seg_end(r) - seg_start(l - 1)
      if next_cur <= max_duration:
        l -= 1
        expanded = True
        continue
    if r < n - 1 and (seg_end(r + 1) - anchor_end) <= max_post_context:
      next_cur = seg_end(r + 1) - seg_start(l)
      if next_cur <= max_duration:
        r += 1
        expanded = True

  start = max(0.0, seg_start(l) - pre_roll)
  end = seg_end(r) + post_roll
  if end <= start:
    end = start + 1.0

  # Boundary cleanup: avoid starting/ending on filler-heavy segments.
  filler, _stop = _filler_and_stopwords()
  filler_starts = ("jadi", "nah", "oke", "ok", "eee", "eh", "em", "hmm")
  filler_ends = {"ya", "yah", "kan", "dong", "deh", "sih", "gitu", "gini", "nih"}

  def is_filler_lead(i: int) -> bool:
    t = str(transcript[i].get("text", "")).strip().lower()
    if not t:
      return True
    # Guard: don't drop strong claims just because they start with a filler-like token.
    if _statement_quality_score(t)[0] >= 3:
      return False
    if t.startswith(filler_starts):
      return True
    toks = _tokenize(t)
    if len(toks) <= 2:
      return True
    filler_ratio = sum(1 for w in toks if w in filler) / max(1, len(toks))
    return filler_ratio >= 0.35

  def is_filler_tail(i: int) -> bool:
    t = str(transcript[i].get("text", "")).strip().lower()
    toks = _tokenize(t)
    if len(toks) <= 2:
      return True
    # Guard: keep strong declarative endings.
    if _statement_quality_score(t)[0] >= 3:
      return False
    if toks and toks[-1] in filler_ends and len(toks) <= 6:
      return True
    filler_ratio = sum(1 for w in toks if w in filler) / max(1, len(toks))
    return filler_ratio >= 0.35

  # Shift left boundary forward if it starts with filler, while preserving min_duration.
  while l < r and is_filler_lead(l) and (seg_end(r) - seg_start(l + 1)) >= min_duration:
    l += 1
  # Shift right boundary backward if it ends with filler, while preserving min_duration.
  while r > l and is_filler_tail(r) and (seg_end(r - 1) - seg_start(l)) >= min_duration:
    r -= 1

  start = max(0.0, seg_start(l) - pre_roll)
  end = seg_end(r) + post_roll
  if end <= start:
    end = start + 1.0

  return l, r, start, end


def detect_highlights(transcript: list[dict]) -> list[dict]:
  """Detect highlight candidates from transcript using heuristics.

  Returns list of dicts: {start, end, score, category, reason}
  """
  if not transcript:
    return []

  # Goal: return fewer, more "krusial" clips.
  # Hard limit to avoid too many highlights.
  max_results = 5
  # Don't output only 1 highlight when there is enough material.
  min_results = 3 if len(transcript) >= 40 else 2
  min_results = min(min_results, max_results)

  # 1) Score each transcript segment so we can anchor highlights more precisely.
  seg_scores: List[int] = []
  seg_reasons: List[List[str]] = []
  seg_punch: List[bool] = []

  for seg in transcript:
    text = str(seg.get("text", ""))
    s, rs = _score_text(text)
    seg_scores.append(s)
    seg_reasons.append(rs)
    seg_punch.append(_is_punchline(text))

  # 2) Create windows around the highest scoring segments.
  # Window sizing is dynamic per anchor "topic", with a hard cap of 60 seconds.
  max_clip_len = 60.0

  # 2a) Anchor selection via local peak detection.
  # A good highlight anchor is a *local max* (not just globally high).
  n = len(transcript)
  peak_threshold = _adaptive_peak_threshold(seg_scores)
  peaks: List[int] = []
  for i in range(n):
    s = seg_scores[i]
    # Punchline anchors may bypass the numeric threshold.
    if (s < peak_threshold) and (not seg_punch[i]):
      continue
    left = seg_scores[i - 1] if i - 1 >= 0 else -10**9
    right = seg_scores[i + 1] if i + 1 < n else -10**9
    # Local peak: not worse than neighbors, and strictly better than at least one side.
    if s >= left and s >= right and (s > left or s > right):
      peaks.append(i)

  # Ensure punchlines are considered even if not local maxima.
  for i in range(n):
    if seg_punch[i] and i not in peaks:
      peaks.append(i)

  # Fallback peaks if transcript is flat
  if not peaks:
    peaks = sorted(range(n), key=lambda i: seg_scores[i], reverse=True)[:max(5, min_results)]

  # Rank peaks: punchlines first, then score.
  ranked_indices = sorted(peaks, key=lambda i: (1 if seg_punch[i] else 0, seg_scores[i]), reverse=True)

  candidates: List[Dict] = []
  for idx in ranked_indices:
    # Gate: prioritize only high-quality anchors.
    if (seg_scores[idx] < peak_threshold) and (not seg_punch[idx]):
      continue

    anchor_text = str(transcript[idx].get("text", ""))
    prof = _dynamic_window_profile(anchor_text)
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
    start, end = _clamp_clip_length(start, end, max_len=max_clip_len)

    # Deduplicate near-identical/overlapping windows.
    if any(_overlap_ratio(start, end, c["start"], c["end"]) >= 0.60 for c in candidates):
      continue

    window_text = " ".join(str(transcript[i].get("text", "")) for i in range(l, r + 1))
    window_score, window_reasons = _score_text(window_text)
    # Blend anchor strength + window coherence.
    # Bias strong to anchor (krusial point), not surrounding filler.
    score = int(round(0.78 * seg_scores[idx] + 0.22 * window_score))

    # Editorial: punchlines should rank higher than explanatory clips.
    if seg_punch[idx] or _is_punchline(window_text):
      score += 2

    # Claim completeness: small boost for point + justification.
    cc_bonus, cc_reason = _claim_completeness_bonus(window_text)
    score += cc_bonus

    # Window-level emotional reinforcement.
    we_bonus, we_reason = _window_emotion_bonus(transcript, l, r)
    score += we_bonus

    # Duration preference:
    # - punchline: 10–16s
    # - regular: 15–25s
    # - explanation may be longer, but never > 60s
    duration = end - start
    duration_bonus = 0
    if seg_punch[idx] or _is_punchline(window_text):
      if 10.0 <= duration <= 16.0:
        duration_bonus += 2
      elif duration < 8.0:
        duration_bonus -= 2
      elif duration > 20.0:
        duration_bonus -= 2
    else:
      # Allow longer clips for explanatory anchors.
      if _classify(anchor_text) == "explanation":
        if 25.0 <= duration <= 45.0:
          duration_bonus += 1
        elif duration < 18.0:
          duration_bonus -= 1
        elif duration > 55.0:
          duration_bonus -= 2
      elif 15.0 <= duration <= 25.0:
        duration_bonus += 2
      elif duration < 12.0:
        duration_bonus -= 2
      elif duration > 35.0:
        duration_bonus -= 2
      elif duration > 28.0:
        duration_bonus -= 1
    score += duration_bonus

    # Require final window to still look strong.
    # Punchlines can pass with slightly lower numeric score.
    min_score = 2 if (seg_punch[idx] or _is_punchline(window_text)) else 3
    if score < min_score:
      continue

    reason_bits: List[str] = []
    if seg_punch[idx] or _is_punchline(window_text):
      reason_bits.append("Punchline")
    if cc_reason:
      reason_bits.append(cc_reason)
    if we_reason:
      reason_bits.append(we_reason)
    if duration_bonus:
      reason_bits.append(f"Durasi: {duration:.0f}s")
    # Merge with the already-compressed window reasons.
    reason_bits.extend(window_reasons or [])
    reason = "; ".join(_compress_reasons(reason_bits, limit=3)) if reason_bits else "Heuristik default"

    candidates.append({
      "start": start,
      "end": end,
      "score": score,
      "category": _classify(window_text),
      "reason": reason,
      "transcript": window_text,
    })

    if len(candidates) >= max_results:
      break

  # 3) Backfill: ensure we don't end up with only 1 highlight.
  if len(candidates) < min_results:
    # Consider next-best indices (including non-peaks) with relaxed threshold.
    fallback_order = sorted(range(len(transcript)), key=lambda i: seg_scores[i], reverse=True)
    for idx in fallback_order:
      if len(candidates) >= min_results:
        break
      if seg_scores[idx] < 2:
        break
      l, r, start, end = _build_window_around(
        transcript,
        idx,
        min_duration=min_duration,
        max_duration=max_duration,
        pre_roll=pre_roll,
        post_roll=post_roll,
      )
      if any(_overlap_ratio(start, end, c["start"], c["end"]) >= 0.60 for c in candidates):
        continue
      window_text = " ".join(str(transcript[i].get("text", "")) for i in range(l, r + 1))
      window_score, _ = _score_text(window_text)
      score = int(round(0.78 * seg_scores[idx] + 0.22 * window_score))
      candidates.append({
        "start": start,
        "end": end,
        "score": score,
        "category": _classify(window_text),
        "reason": "Backfill kandidat (threshold relax)",
        "transcript": window_text,
      })

  candidates.sort(key=lambda x: (x["score"], -(x["end"] - x["start"])), reverse=True)

  # Dynamic cutoff: keep only the truly top tier compared to the best highlight.
  if candidates:
    full = candidates[:]
    best = full[0]["score"]
    cutoff = max(3, best - 2)
    filtered = [c for c in full if c["score"] >= cutoff]
    # Don't over-filter into a single clip
    if len(filtered) >= min_results:
      candidates = filtered
    else:
      candidates = full

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
