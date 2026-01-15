"""User feedback input layer (deterministic, no ML/AI, no external deps).

Scope:
- Validate structured user responses for clips
- Persist to DB when a session is provided
- No scoring changes
- No learning/bias application yet

Design notes:
- Weaknesses/strengths are stored as JSON arrays in Text columns for portability.
- This module is pure and testable: validation is deterministic and raises
  clear ValueErrors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json


WEAKNESS_TYPES = {
  "too_short",
  "too_long",
  "weak_opening",
  "weak_closing",
  "unclear_topic",
  "no_insight",
  "missing_reasoning",
  "no_takeaway",
  "too_much_context",
}

STRENGTH_TYPES = {
  "strong_opening",
  "clear_insight",
  "good_duration",
  "strong_reasoning",
  "clear_takeaway",
  "standalone_value",
  "good_topic",
}

# Backward-compatible aliases (older internal names or earlier drafts)
_STRENGTH_ALIASES = {
  "good_topic_choice": "good_topic",
}


@dataclass(frozen=True)
class ValidatedFeedback:
  clip_id: str
  rating: int
  weaknesses: List[str]
  strengths: List[str]
  notes: Optional[str]


def _normalize_enum_list(values: Optional[List[str]], allowed: set[str], aliases: Optional[Dict[str, str]] = None) -> List[str]:
  if not values:
    return []
  if not isinstance(values, list):
    raise ValueError("weaknesses/strengths must be a list")

  normalized: List[str] = []
  for v in values:
    if not isinstance(v, str):
      raise ValueError("enum values must be strings")
    key = v.strip()
    if not key:
      continue
    key = key.lower()
    if aliases and key in aliases:
      key = aliases[key]
    if key not in allowed:
      raise ValueError(f"invalid enum value: {key}")
    normalized.append(key)

  # Deterministic order: unique + sorted
  return sorted(set(normalized))


def validate_user_feedback(data: Dict[str, Any]) -> ValidatedFeedback:
  """Validate and normalize user feedback input.

  Rules:
  - clip_id required, non-empty
  - rating required integer 1..10
  - weaknesses/strengths optional lists of defined enums
  - notes optional string, length-limited
  """
  if not isinstance(data, dict):
    raise ValueError("data must be a dict")

  clip_id = data.get("clip_id")
  if not isinstance(clip_id, str) or not clip_id.strip():
    raise ValueError("clip_id is required")
  clip_id = clip_id.strip()
  if len(clip_id) > 128:
    raise ValueError("clip_id too long")

  rating = data.get("rating")
  if isinstance(rating, bool):
    raise ValueError("rating must be an integer 1-10")
  if not isinstance(rating, int):
    raise ValueError("rating must be an integer 1-10")
  if rating < 1 or rating > 10:
    raise ValueError("rating must be in range 1-10")

  weaknesses = _normalize_enum_list(data.get("weaknesses"), WEAKNESS_TYPES)
  strengths = _normalize_enum_list(data.get("strengths"), STRENGTH_TYPES, aliases=_STRENGTH_ALIASES)

  notes = data.get("notes")
  if notes is not None:
    if not isinstance(notes, str):
      raise ValueError("notes must be a string")
    notes = notes.strip()
    if len(notes) > 2000:
      raise ValueError("notes too long")
    if notes == "":
      notes = None

  return ValidatedFeedback(
    clip_id=clip_id,
    rating=rating,
    weaknesses=weaknesses,
    strengths=strengths,
    notes=notes,
  )


def save_user_feedback(feedback: ValidatedFeedback, db_session=None) -> None:
  """Persist validated feedback.

  - If db_session is None: no-op (safe default).
  - For AsyncSession: this function only calls `add()` (commit is caller-owned).
  - For sync Session: `add()` works as well.

  This function intentionally does not apply learning/scoring.
  """
  if db_session is None:
    return

  from app.database.models import ClipUserFeedback

  row = ClipUserFeedback(
    clip_id=feedback.clip_id,
    rating=feedback.rating,
    weaknesses=json.dumps(feedback.weaknesses, ensure_ascii=False),
    strengths=json.dumps(feedback.strengths, ensure_ascii=False),
    notes=feedback.notes,
  )

  # AsyncSession.add is sync; commit is async and caller-owned.
  db_session.add(row)


def submit_clip_feedback(
  clip_id: str,
  rating: int,
  weaknesses: Optional[List[str]] = None,
  strengths: Optional[List[str]] = None,
  notes: Optional[str] = None,
  db_session=None,
) -> None:
  """Public API: validate + normalize + persist feedback.

  Does NOT trigger learning yet.
  """
  validated = validate_user_feedback({
    "clip_id": clip_id,
    "rating": rating,
    "weaknesses": weaknesses,
    "strengths": strengths,
    "notes": notes,
  })
  save_user_feedback(validated, db_session=db_session)
