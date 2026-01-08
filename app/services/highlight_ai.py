"""app.services.highlight_ai

Gemini-based highlight selection for short-form video.

This module intentionally does NOT implement local heuristic scoring.
It delegates highlight judgment to Google Gemini in a single request.

Input transcript format (chronological, already segmented):
[
  {"start": float, "duration": float, "text": str},
  ...
]

Output format:
[
  {"start": float, "end": float, "reason": str},
  ...
]

Environment variables:
- GEMINI_API_KEY (or GOOGLE_API_KEY): required
- GEMINI_MODEL: optional (default: gemini-1.5-flash)
- HIGHLIGHT_DEBUG=1: optional, logs Gemini/JSON issues to stderr
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests


_GEMINI_API_HOST = "https://generativelanguage.googleapis.com"
_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
_DEFAULT_GEMINI_API_VERSION = "v1"


def select_highlights_with_gemini(
  transcript: List[Dict[str, Any]],
  *,
  api_key: Optional[str] = None,
  model: Optional[str] = None,
  timeout_s: float = 60.0,
) -> List[Dict[str, Any]]:
  """Ask Gemini to select the best 10–30s highlight clips.

  Safeguards:
  - One Gemini request per transcript
  - No retries / no multi-step chaining
  - Returns [] on any error
  - Ignores malformed/invalid highlight entries
  """
  if not isinstance(transcript, list) or not transcript:
    return []

  resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
  if not resolved_key:
    return []

  resolved_model = _normalize_model_name(model or os.getenv("GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL)

  bounds = _transcript_time_bounds(transcript)
  if not bounds:
    return []
  transcript_start, transcript_end = bounds

  prompt = _build_gemini_prompt(transcript)

  try:
    raw_text = _gemini_generate_content(
      prompt=prompt,
      api_key=resolved_key,
      model=resolved_model,
      timeout_s=timeout_s,
    )
  except Exception as exc:
    if _debug_enabled():
      print(f"[highlight_ai] Gemini request failed: {exc}", file=sys.stderr)
    return []

  return _parse_highlights_from_gemini_text(
    raw_text,
    transcript_start=transcript_start,
    transcript_end=transcript_end,
  )


def detect_highlights(transcript: list[dict]) -> list[dict]:
  """Backward-compatible wrapper used by the routes."""
  return list(select_highlights_with_gemini(transcript))


def _normalize_model_name(model: str) -> str:
  """Normalize model names to something accepted by the REST endpoint.

  Users often paste model IDs like `models/gemini-1.5-flash-latest`. The v1beta
  generateContent endpoint may not accept the `-latest` aliases.
  This keeps a single request per transcript by normalizing before calling Gemini.
  """
  m = (model or "").strip()
  if not m:
    return _DEFAULT_GEMINI_MODEL
  if m.startswith("models/"):
    m = m[len("models/") :]

  aliases = {
    "gemini-1.5-flash-latest": "gemini-1.5-flash",
    "gemini-1.5-pro-latest": "gemini-1.5-pro",
    # Gemini 1.5 IDs are no longer listed for some API keys; map to a known-good 2.x model.
    "gemini-1.5-flash": "gemini-2.0-flash",
  }
  return aliases.get(m, m)


def _build_gemini_prompt(transcript: List[Dict[str, Any]]) -> str:
  # IMPORTANT: This prompt mirrors the user-provided template.
  # To support very large transcripts, we pack segments to reduce repeated JSON keys.
  # Format: [[start, duration, text], ...] (content unchanged)
  packed = _pack_transcript(transcript)
  transcript_json = json.dumps(packed, ensure_ascii=False, separators=(",", ":"))

  return (
    "You are a professional short-form video editor.\n\n"
    "Given the following YouTube transcript segments (in chronological order),\n"
    "select the BEST highlight clips for short-form video (TikTok/Reels/Shorts).\n\n"
    "Rules:\n"
    "- Choose 3–5 clips maximum\n"
    "- Each clip should be between 10–30 seconds\n"
    "- Each clip must be able to stand alone\n"
    "- Prefer:\n"
    "  • strong opinions\n"
    "  • clear advice\n"
    "  • emotional moments\n"
    "  • punchlines\n"
    "- Avoid:\n"
    "  • filler\n"
    "  • setup without payoff\n"
    "  • technical explanations\n"
    "  • repeated ideas\n\n"
    "Return ONLY valid JSON in this format:\n\n"
    "{\n"
    "  \"highlights\": [\n"
    "    {\n"
    "      \"start\": number,\n"
    "      \"end\": number,\n"
    "      \"reason\": string\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Transcript (JSON; each entry is [start, duration, text]):\n"
    f"<<<{transcript_json}>>>\n"
  )


def _gemini_generate_content(*, prompt: str, api_key: str, model: str, timeout_s: float) -> str:
  api_version = str(os.getenv("GEMINI_API_VERSION") or _DEFAULT_GEMINI_API_VERSION).strip() or _DEFAULT_GEMINI_API_VERSION
  url = f"{_GEMINI_API_HOST}/{api_version}/models/{model}:generateContent"
  params = {"key": api_key}
  generation_config: Dict[str, Any] = {
    "temperature": 0.4,
    "maxOutputTokens": 1024,
  }

  # Some endpoints reject this field (notably v1 at time of writing).
  if api_version.startswith("v1beta"):
    generation_config["responseMimeType"] = "application/json"

  payload: Dict[str, Any] = {
    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    "generationConfig": generation_config,
  }

  timeout: Tuple[float, float] = (10.0, max(10.0, float(timeout_s)))
  resp = requests.post(url, params=params, json=payload, timeout=timeout)
  if _debug_enabled():
    print(f"[highlight_ai] Gemini request url: {url}", file=sys.stderr)

  try:
    resp.raise_for_status()
  except requests.HTTPError as exc:
    body = (resp.text or "").strip()
    if len(body) > 1000:
      body = body[:1000] + "..."
    raise RuntimeError(f"Gemini HTTP {resp.status_code}: {body}") from exc

  data = resp.json()
  candidates = data.get("candidates")
  if not isinstance(candidates, list) or not candidates:
    raise RuntimeError("Gemini returned no candidates")

  cand0 = candidates[0] if isinstance(candidates[0], dict) else None
  content = cand0.get("content") if isinstance(cand0, dict) else None
  parts = content.get("parts") if isinstance(content, dict) else None
  if not isinstance(parts, list) or not parts:
    raise RuntimeError("Gemini candidate has no content parts")

  part0 = parts[0] if isinstance(parts[0], dict) else None
  text = part0.get("text") if isinstance(part0, dict) else None
  if not isinstance(text, str) or not text.strip():
    raise RuntimeError("Gemini returned empty text")
  return text


def _parse_highlights_from_gemini_text(
  text: str,
  *,
  transcript_start: float,
  transcript_end: float,
) -> List[Dict[str, Any]]:
  cleaned = _extract_json_object(text)
  if cleaned is None:
    if _debug_enabled():
      preview = (text or "")
      preview = preview[:500] + ("..." if len(preview) > 500 else "")
      print(f"[highlight_ai] Gemini output did not contain a JSON object. Preview: {preview}", file=sys.stderr)
    return []

  try:
    obj = json.loads(cleaned)
  except Exception as exc:
    if _debug_enabled():
      snippet = cleaned[:500] + ("..." if len(cleaned) > 500 else "")
      print(f"[highlight_ai] Failed to parse Gemini JSON: {exc}. Snippet: {snippet}", file=sys.stderr)
    return []

  raw = obj.get("highlights") if isinstance(obj, dict) else None
  if not isinstance(raw, list):
    if _debug_enabled():
      keys = list(obj.keys()) if isinstance(obj, dict) else str(type(obj))
      print(f"[highlight_ai] Gemini JSON missing 'highlights' list. Keys/type: {keys}", file=sys.stderr)
    return []

  out: List[Dict[str, Any]] = []
  for item in raw:
    if not isinstance(item, dict):
      continue
    start = _as_float(item.get("start"))
    end = _as_float(item.get("end"))
    reason = item.get("reason")

    if start is None or end is None:
      continue
    if not isinstance(reason, str) or not reason.strip():
      continue
    if end <= start:
      continue

    # Validate timestamps are within transcript bounds; ignore invalid entries.
    if start < transcript_start or end > transcript_end:
      continue

    # Enforce requested clip window (10–30 seconds).
    dur = end - start
    if dur < 10.0 or dur > 30.0:
      continue

    out.append({"start": float(start), "end": float(end), "reason": reason.strip()})

  # Keep Gemini ordering; cap at 5 per prompt.
  return out[:5]


def _pack_transcript(transcript: List[Dict[str, Any]]) -> List[List[Any]]:
  packed: List[List[Any]] = []
  for seg in transcript:
    if not isinstance(seg, dict):
      continue
    start = seg.get("start")
    duration = seg.get("duration")
    text = seg.get("text")
    if text is None:
      text = ""
    packed.append([start, duration, str(text)])
  return packed


def _extract_json_object(text: str) -> Optional[str]:
  if not isinstance(text, str):
    return None
  s = text.strip()

  # Strip ```json ... ``` fences if present
  m = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", s, flags=re.DOTALL | re.IGNORECASE)
  if m:
    s = m.group(1).strip()

  if s.startswith("{") and s.endswith("}"):
    return s

  start = s.find("{")
  end = s.rfind("}")
  if start == -1 or end == -1 or end <= start:
    return None

  candidate = s[start:end + 1].strip()
  if candidate.startswith("{") and candidate.endswith("}"):
    return candidate
  return None


def _transcript_time_bounds(transcript: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
  starts: List[float] = []
  ends: List[float] = []
  for seg in transcript:
    if not isinstance(seg, dict):
      continue
    s = _as_float(seg.get("start"))
    d = _as_float(seg.get("duration"))
    if s is None or d is None or d < 0:
      continue
    starts.append(float(s))
    ends.append(float(s + d))
  if not starts or not ends:
    return None
  return (min(starts), max(ends))


def _as_float(value: Any) -> Optional[float]:
  if isinstance(value, (int, float)):
    return float(value)
  if isinstance(value, str):
    try:
      return float(value.strip())
    except Exception:
      return None
  return None


def _debug_enabled() -> bool:
  return str(os.getenv("HIGHLIGHT_DEBUG") or os.getenv("GEMINI_DEBUG") or "").strip() in {"1", "true", "TRUE", "yes", "YES"}
