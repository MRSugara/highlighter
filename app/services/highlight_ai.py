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
from typing import List, Dict, Tuple


def _tokenize(text: str) -> List[str]:
  # Simple, library-free tokenizer
  cleaned = []
  for ch in text.lower():
    if ch.isalnum() or ch in {" ", "-"}:
      cleaned.append(ch)
    else:
      cleaned.append(" ")
  return [w for w in "".join(cleaned).split() if w]


def _score_text(text: str) -> Tuple[int, List[str]]:
  words = _tokenize(text)
  n = len(words)
  score = 0
  reasons: List[str] = []

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
  elif n < 5 or n > 30:
    score -= 3
    reasons.append("Terlalu pendek/terlalu panjang")

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


def detect_highlights(transcript: list[dict]) -> list[dict]:
  """Detect highlight candidates from transcript using heuristics.

  Returns list of dicts: {start, end, score, category, reason}
  """
  if not transcript:
    return []

  chunks = _merge_chunks(transcript, 30.0, 90.0)

  candidates: List[Dict] = []
  for ch in chunks:
    score, reasons = _score_text(ch["text"])  # type: ignore
    category = _classify(ch["text"])  # type: ignore
    # Build reason string
    reason = "; ".join(reasons) if reasons else "Heuristik default"
    candidates.append({
      "start": ch["start"],
      "end": ch["end"],
      "score": score,
      "category": category,
      "reason": reason,
      "transcript": ch["text"],
    })

  # Sort by score descending; keep top 10 to avoid noise
  candidates.sort(key=lambda x: (x["score"], -(x["end"] - x["start"])), reverse=True)

  # Ensure not only one candidate returned; keep at least 3 if available
  top_n = 10 if len(candidates) >= 10 else max(3, len(candidates))
  return candidates[:top_n]

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
