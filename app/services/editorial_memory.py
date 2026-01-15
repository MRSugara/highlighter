"""
Editorial Memory: Deterministic Learning Layer

This module implements a simple, deterministic learning system that adapts
highlight scoring based on historical editorial feedback (accept/reject decisions).

PHILOSOPHY:
- Learning is ADDITIVE (biasing), not replacing heuristics
- Purely deterministic (no ML, no randomness)
- Safe defaults (zero bias if no data)
- Gradual alignment with human editorial taste

IMPORTANT: This system NEVER overrides core editorial gates or completeness checks.
It only nudges scores toward historically successful patterns.
"""

from typing import Dict, Optional, List, Tuple, Iterable
from dataclasses import dataclass, field

import json


@dataclass
class EditorialBiasProfile:
  """
  Bias profile computed from historical editorial feedback.
  
  All biases are integers in range [-3, +3].
  Bias computation: clamp(round((accepted - rejected) / total * 3), -3, 3)
  
  Total applied bias is capped at ±4 points.
  """
  # Bias by category (e.g., "educational": +2, "punchline": -1)
  category_bias: Dict[str, int] = field(default_factory=dict)
  
  # Bias by duration bucket (e.g., "18-30s": +2, "<12s": -2)
  duration_bias: Dict[str, int] = field(default_factory=dict)
  
  # Bias for clips containing punchlines
  punchline_bias: int = 0
  
  # Bias for clips with early hook position (anchor in first 25%)
  early_hook_bias: int = 0
  
  # Total number of feedback records used
  feedback_count: int = 0
  
  def is_zero(self) -> bool:
    """Check if this is a zero-bias profile (no learning data)."""
    return (
      self.feedback_count == 0 and
      not self.category_bias and
      not self.duration_bias and
      self.punchline_bias == 0 and
      self.early_hook_bias == 0
    )


# ============================
# Structured User Feedback
# ============================

# Enumerations are represented as strings (stored as JSON arrays in DB)
WEAKNESS_TYPES = {
  "too_short",
  "too_long",
  "weak_opening",
  "weak_closing",
  "unclear_topic",
  "no_insight",
  "too_much_context",
  "missing_reasoning",
  "no_takeaway",
}

STRENGTH_TYPES = {
  "strong_opening",
  "clear_insight",
  "good_duration",
  "strong_reasoning",
  "clear_takeaway",
  "standalone_value",
  "good_topic_choice",
}


def _safe_json_list(value: Optional[str]) -> List[str]:
  if not value:
    return []
  try:
    data = json.loads(value)
    if isinstance(data, list):
      return [str(x) for x in data]
    return []
  except Exception:
    return []


def _hook_bucket(hook_position: Optional[float]) -> str:
  if hook_position is None:
    return "mid"
  try:
    hp = float(hook_position)
  except Exception:
    return "mid"
  if hp <= 0.25:
    return "early"
  if hp <= 0.55:
    return "mid"
  return "late"


def _rating_bias(avg_rating: float, strong_pos_rate: float, strong_neg_rate: float) -> int:
  """Convert average rating to an integer bias in [-3, +3].

  Rating is 1..10. Neutral ~5.5.
  Strong signals:
  - rating <= 3 counts as strong negative
  - rating >= 8 counts as strong positive
  """
  # Normalize around 5.5 into approximately [-3, +3]
  centered = (avg_rating - 5.5) / 4.5  # ~[-1, +1]
  bias = round(centered * 3)
  # Boost/penalize if strong signals dominate
  if strong_pos_rate >= 0.30:
    bias += 1
  if strong_neg_rate >= 0.30:
    bias -= 1
  return _clamp(int(bias), -3, 3)


@dataclass
class EditorialLearningProfile:
  """Aggregated, explainable learning from rich user feedback.

  Bias effects are intended to be ADDITIVE and later capped to ±4 total
  when combined with accept/reject bias.
  """
  feedback_count: int = 0

  # Preferred duration range per category, derived from feedback (min,max)
  preferred_duration_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)

  # Common patterns (sorted by frequency desc)
  common_failure_patterns: List[str] = field(default_factory=list)
  common_success_patterns: List[str] = field(default_factory=list)

  # Rating-derived bias maps
  rating_bias_by_category: Dict[str, int] = field(default_factory=dict)
  rating_bias_by_duration: Dict[str, int] = field(default_factory=dict)
  rating_bias_by_hook_bucket: Dict[str, int] = field(default_factory=dict)
  rating_bias_punchline: int = 0

  # Behavior adjustments (bounded deltas)
  min_duration_delta_by_category: Dict[str, float] = field(default_factory=dict)
  max_duration_delta_by_category: Dict[str, float] = field(default_factory=dict)
  max_pre_context_delta_by_category: Dict[str, float] = field(default_factory=dict)
  max_post_context_delta_by_category: Dict[str, float] = field(default_factory=dict)

  # Structural preference flags
  require_insight_all: bool = False
  opening_strictness: int = 0  # 0=default, 1=stricter early hooks

  def is_zero(self) -> bool:
    return self.feedback_count == 0


def _duration_bucket_for_feedback(clip_duration: float) -> str:
  return _duration_bucket(float(clip_duration))


def _top_k_counts(counter: Dict[str, int], k: int = 3) -> List[str]:
  return [
    key for key, _count in sorted(counter.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:k]
  ]


def build_editorial_learning_profile(feedback_rows: Iterable[Dict]) -> EditorialLearningProfile:
  """Pure aggregation: build learning profile from normalized feedback dicts.

  Expected keys per row:
    - category: str
    - clip_duration: float
    - is_punchline: bool
    - hook_position: Optional[float]
    - rating: int
    - weaknesses: List[str]
    - strengths: List[str]
  """
  rows = list(feedback_rows)
  profile = EditorialLearningProfile(feedback_count=len(rows))
  if not rows:
    return profile

  weakness_counts: Dict[str, int] = {}
  strength_counts: Dict[str, int] = {}

  # Rating buckets
  ratings_by_category: Dict[str, List[int]] = {}
  ratings_by_duration: Dict[str, List[int]] = {}
  ratings_by_hook: Dict[str, List[int]] = {"early": [], "mid": [], "late": []}
  ratings_punchline = {"punch": [], "non": []}

  # Strong-signal counts (for bias shaping)
  def is_strong_pos(r: int) -> bool:
    return r >= 8
  def is_strong_neg(r: int) -> bool:
    return r <= 3

  # Weakness rates for behavior tuning (per category)
  weakness_by_category: Dict[str, Dict[str, int]] = {}
  total_by_category: Dict[str, int] = {}

  # Duration preference hints per category
  duration_ratings_by_category_bucket: Dict[str, Dict[str, List[int]]] = {}

  for row in rows:
    category = str(row.get("category") or "") or "unknown"
    clip_duration = float(row.get("clip_duration") or 0.0)
    punch = bool(row.get("is_punchline") or False)
    hook_pos = row.get("hook_position")
    hook_b = _hook_bucket(hook_pos)
    rating = int(row.get("rating") or 0)
    rating = int(_clamp(rating, 1, 10))

    weaknesses = [w for w in (row.get("weaknesses") or []) if w in WEAKNESS_TYPES]
    strengths = [s for s in (row.get("strengths") or []) if s in STRENGTH_TYPES]

    # Counters
    for w in weaknesses:
      weakness_counts[w] = weakness_counts.get(w, 0) + 1
      weakness_by_category.setdefault(category, {})
      weakness_by_category[category][w] = weakness_by_category[category].get(w, 0) + 1
    for s in strengths:
      strength_counts[s] = strength_counts.get(s, 0) + 1

    total_by_category[category] = total_by_category.get(category, 0) + 1

    # Ratings by dimension
    ratings_by_category.setdefault(category, []).append(rating)
    dur_bucket = _duration_bucket_for_feedback(clip_duration)
    ratings_by_duration.setdefault(dur_bucket, []).append(rating)
    ratings_by_hook.setdefault(hook_b, []).append(rating)
    (ratings_punchline["punch"] if punch else ratings_punchline["non"]).append(rating)

    # Duration preference data
    duration_ratings_by_category_bucket.setdefault(category, {})
    duration_ratings_by_category_bucket[category].setdefault(dur_bucket, []).append(rating)

  # Patterns
  profile.common_failure_patterns = _top_k_counts(weakness_counts, k=4)
  profile.common_success_patterns = _top_k_counts(strength_counts, k=4)

  # Rating biases by category
  for category, rs in ratings_by_category.items():
    if len(rs) < 3:
      continue
    avg = sum(rs) / len(rs)
    sp = sum(1 for r in rs if is_strong_pos(r)) / len(rs)
    sn = sum(1 for r in rs if is_strong_neg(r)) / len(rs)
    profile.rating_bias_by_category[category] = _rating_bias(avg, sp, sn)

    # Preferred duration bucket for category
    buckets = duration_ratings_by_category_bucket.get(category, {})
    best_bucket = None
    best_avg = None
    for b, brs in buckets.items():
      if len(brs) < 2:
        continue
      bavg = sum(brs) / len(brs)
      if best_avg is None or bavg > best_avg:
        best_avg = bavg
        best_bucket = b
    if best_bucket:
      if best_bucket == "<12s":
        profile.preferred_duration_range[category] = (10.0, 12.0)
      elif best_bucket == "12-18s":
        profile.preferred_duration_range[category] = (12.0, 18.0)
      elif best_bucket == "18-30s":
        profile.preferred_duration_range[category] = (18.0, 30.0)
      else:
        profile.preferred_duration_range[category] = (30.0, 45.0)

  # Rating biases by duration bucket
  for bucket, rs in ratings_by_duration.items():
    if len(rs) < 3:
      continue
    avg = sum(rs) / len(rs)
    sp = sum(1 for r in rs if is_strong_pos(r)) / len(rs)
    sn = sum(1 for r in rs if is_strong_neg(r)) / len(rs)
    profile.rating_bias_by_duration[bucket] = _rating_bias(avg, sp, sn)

  # Rating biases by hook bucket
  for bucket, rs in ratings_by_hook.items():
    if len(rs) < 3:
      continue
    avg = sum(rs) / len(rs)
    sp = sum(1 for r in rs if is_strong_pos(r)) / len(rs)
    sn = sum(1 for r in rs if is_strong_neg(r)) / len(rs)
    profile.rating_bias_by_hook_bucket[bucket] = _rating_bias(avg, sp, sn)

  # Punchline rating bias
  p_rs = ratings_punchline["punch"]
  n_rs = ratings_punchline["non"]
  if len(p_rs) + len(n_rs) >= 6 and len(p_rs) >= 2 and len(n_rs) >= 2:
    p_avg = sum(p_rs) / len(p_rs)
    n_avg = sum(n_rs) / len(n_rs)
    # If punchlines consistently rated higher/lower, bias toward/away
    diff = p_avg - n_avg
    # Map ~[-3..+3]
    profile.rating_bias_punchline = _clamp(int(round(diff / 2.0)), -3, 3)

  # Behavior adjustments (gradual + bounded)
  for category, total in total_by_category.items():
    if total < 6:
      continue
    wcounts = weakness_by_category.get(category, {})
    too_short_rate = wcounts.get("too_short", 0) / total
    too_long_rate = wcounts.get("too_long", 0) / total
    weak_open_rate = wcounts.get("weak_opening", 0) / total
    no_insight_rate = (wcounts.get("no_insight", 0) + wcounts.get("missing_reasoning", 0) + wcounts.get("no_takeaway", 0)) / total

    # Duration deltas (max ±4s on min, max -6s)
    if too_short_rate >= 0.25:
      profile.min_duration_delta_by_category[category] = min(4.0, 2.0 + 4.0 * (too_short_rate - 0.25))
    if too_long_rate >= 0.25:
      profile.max_duration_delta_by_category[category] = -min(6.0, 2.0 + 6.0 * (too_long_rate - 0.25))

    # Opening tuning: keep anchor earlier by trimming pre-context and allowing more post-context
    if weak_open_rate >= 0.25:
      profile.opening_strictness = 1
      profile.max_pre_context_delta_by_category[category] = -min(3.0, 1.0 + 3.0 * (weak_open_rate - 0.25))
      profile.max_post_context_delta_by_category[category] = min(4.0, 2.0 + 4.0 * (weak_open_rate - 0.25))

    # Structural learning: require insight layer for all categories when frequent
    if no_insight_rate >= 0.25:
      profile.require_insight_all = True

  return profile


async def load_editorial_learning_profile_async(db_session=None) -> EditorialLearningProfile:
  """Load ClipFeedback rows via AsyncSession and build a learning profile.

  Safe fallback: returns zero profile if session is None or on any error.
  """
  if db_session is None:
    return EditorialLearningProfile()

  try:
    from app.database.models import ClipFeedback
    from sqlalchemy import select

    result = await db_session.execute(select(ClipFeedback))
    records = result.scalars().all()

    normalized: List[Dict] = []
    for r in records:
      clip_duration = float(r.clip_end - r.clip_start)
      weaknesses = [w for w in _safe_json_list(r.weaknesses) if w in WEAKNESS_TYPES]
      strengths = [s for s in _safe_json_list(r.strengths) if s in STRENGTH_TYPES]
      normalized.append({
        "category": r.category or "unknown",
        "clip_duration": clip_duration,
        "is_punchline": bool(int(r.is_punchline or 0)),
        "hook_position": r.hook_position,
        "rating": int(r.rating or 1),
        "weaknesses": weaknesses,
        "strengths": strengths,
      })

    return build_editorial_learning_profile(normalized)

  except Exception as e:
    print(f"Warning: Could not load editorial learning profile: {e}")
    return EditorialLearningProfile()


def _clamp(value: int, min_val: int, max_val: int) -> int:
  """Clamp value to range [min_val, max_val]."""
  return max(min_val, min(max_val, value))


def _compute_bias(accepted: int, rejected: int) -> int:
  """
  Compute bias from acceptance statistics.
  
  Formula: clamp(round((accepted - rejected) / total * 3), -3, 3)
  
  Examples:
    - 10 accepted, 0 rejected → +3 (100% acceptance)
    - 8 accepted, 2 rejected → +2 (80% acceptance)
    - 5 accepted, 5 rejected → 0 (50/50)
    - 2 accepted, 8 rejected → -2 (20% acceptance)
    - 0 accepted, 10 rejected → -3 (0% acceptance)
  """
  total = accepted + rejected
  if total == 0:
    return 0
  
  # Ratio in range [-1, +1]
  ratio = (accepted - rejected) / total
  
  # Scale to [-3, +3] and round
  bias = round(ratio * 3)
  
  return _clamp(bias, -3, 3)


def _duration_bucket(clip_duration: float) -> str:
  """Categorize clip duration into buckets for bias computation."""
  if clip_duration < 12.0:
    return "<12s"
  elif clip_duration < 18.0:
    return "12-18s"
  elif clip_duration < 30.0:
    return "18-30s"
  else:
    return ">30s"


def load_editorial_bias(db_session=None) -> EditorialBiasProfile:
  """
  Load historical editorial feedback and compute bias profile.
  
  This function aggregates accept/reject decisions by:
  - Category (educational, punchline, insight, etc.)
  - Duration bucket (<12s, 12-18s, 18-30s, >30s)
  - Punchline presence
  - Hook position (early vs late)
  
  Args:
    db_session: Optional database session. If None, returns zero bias.
  
  Returns:
    EditorialBiasProfile with computed biases, or zero bias if no data.
  
  IMPORTANT: This function is deterministic and side-effect free.
  """
  # Return zero bias if no database session provided
  if db_session is None:
    return EditorialBiasProfile()
  
  try:
    # Import here to avoid circular dependencies
    from app.database.models import Highlight
    from sqlalchemy import func
    
    # Query all highlights with editorial decisions
    # NOTE: For now, we'll use score as proxy:
    #   - score >= 8: accepted (high quality)
    #   - score < 5: rejected (low quality)
    #   - 5-7: neutral (not used for learning)
    # 
    # TODO: Add explicit "accepted" boolean column for real feedback
    
    highlights = db_session.query(Highlight).filter(
      Highlight.score.isnot(None)
    ).all()
    
    if not highlights:
      return EditorialBiasProfile()
    
    # Aggregate statistics
    category_stats: Dict[str, Tuple[int, int]] = {}  # {category: (accepted, rejected)}
    duration_stats: Dict[str, Tuple[int, int]] = {}  # {bucket: (accepted, rejected)}
    punchline_stats = [0, 0]  # [accepted, rejected]
    early_hook_stats = [0, 0]  # [accepted, rejected]
    
    for h in highlights:
      # Determine if accepted or rejected based on score
      # This is a proxy until we have explicit feedback
      if h.score >= 8:
        is_accepted = True
      elif h.score < 5:
        is_accepted = False
      else:
        continue  # Neutral zone - don't use for learning
      
      idx = 0 if is_accepted else 1
      
      # Category stats
      if h.category:
        if h.category not in category_stats:
          category_stats[h.category] = [0, 0]
        category_stats[h.category][idx] += 1
      
      # Duration stats
      clip_duration = h.end - h.start
      bucket = _duration_bucket(clip_duration)
      if bucket not in duration_stats:
        duration_stats[bucket] = [0, 0]
      duration_stats[bucket][idx] += 1
      
      # Punchline stats (check if category or reason mentions punchline)
      if h.category == "punchline" or (h.reason and "punchline" in h.reason.lower()):
        punchline_stats[idx] += 1
      
      # Early hook stats (check if reason mentions early hook)
      if h.reason and ("early hook" in h.reason.lower() or "hook early" in h.reason.lower()):
        early_hook_stats[idx] += 1
    
    # Compute biases
    profile = EditorialBiasProfile()
    profile.feedback_count = len(highlights)
    
    # Category biases
    for cat, (accepted, rejected) in category_stats.items():
      if accepted + rejected >= 3:  # Minimum 3 samples
        profile.category_bias[cat] = _compute_bias(accepted, rejected)
    
    # Duration biases
    for bucket, (accepted, rejected) in duration_stats.items():
      if accepted + rejected >= 3:  # Minimum 3 samples
        profile.duration_bias[bucket] = _compute_bias(accepted, rejected)
    
    # Punchline bias
    if sum(punchline_stats) >= 3:
      profile.punchline_bias = _compute_bias(punchline_stats[0], punchline_stats[1])
    
    # Early hook bias
    if sum(early_hook_stats) >= 3:
      profile.early_hook_bias = _compute_bias(early_hook_stats[0], early_hook_stats[1])
    
    return profile
  
  except Exception as e:
    # Safe fallback: return zero bias on any error
    # This ensures the system never fails due to learning layer issues
    print(f"Warning: Could not load editorial bias: {e}")
    return EditorialBiasProfile()


def apply_editorial_bias(
  base_score: int,
  category: str,
  clip_duration: float,
  is_punchline: bool,
  hook_position: float,
  bias_profile: EditorialBiasProfile,
  reasons: List[str],
  learning_profile: Optional[EditorialLearningProfile] = None,
  insight_pass: Optional[bool] = None,
  full_insight: Optional[bool] = None,
) -> int:
  """
  Apply editorial bias to base score.
  
  Args:
    base_score: Computed base score before bias
    category: Clip category (educational, punchline, insight, etc.)
    clip_duration: Clip duration in seconds
    is_punchline: Whether clip contains punchline
    hook_position: Relative position of hook (0.0 = start, 1.0 = end)
    bias_profile: Loaded bias profile
    reasons: List to append bias explanations to
  
  Returns:
    Final score after applying capped bias (base_score ± 4 max)
  
  IMPORTANT: This function is deterministic and pure (no side effects).
  """
  # If there's no learning and no historical bias, keep score unchanged.
  if bias_profile.is_zero() and (learning_profile is None or learning_profile.is_zero()):
    return base_score
  
  applied_biases: List[Tuple[str, int]] = []
  total_bias = 0
  
  # Category bias
  if category in bias_profile.category_bias:
    bias = bias_profile.category_bias[category]
    if bias != 0:
      applied_biases.append((f"{category} category preference", bias))
      total_bias += bias
  
  # Duration bias
  bucket = _duration_bucket(clip_duration)
  if bucket in bias_profile.duration_bias:
    bias = bias_profile.duration_bias[bucket]
    if bias != 0:
      applied_biases.append((f"{bucket} duration preference", bias))
      total_bias += bias
  
  # Punchline bias
  if is_punchline and bias_profile.punchline_bias != 0:
    applied_biases.append(("punchline preference", bias_profile.punchline_bias))
    total_bias += bias_profile.punchline_bias
  
  # Early hook bias (hook in first 25%)
  if hook_position <= 0.25 and bias_profile.early_hook_bias != 0:
    applied_biases.append(("early hook preference", bias_profile.early_hook_bias))
    total_bias += bias_profile.early_hook_bias

  # ============================
  # Rich user feedback learning
  # ============================
  if learning_profile is not None and not learning_profile.is_zero():
    # Rating-derived biases
    if category in learning_profile.rating_bias_by_category:
      b = learning_profile.rating_bias_by_category[category]
      if b:
        applied_biases.append(("rating preference (category)", b))
        total_bias += b

    bucket = _duration_bucket(clip_duration)
    if bucket in learning_profile.rating_bias_by_duration:
      b = learning_profile.rating_bias_by_duration[bucket]
      if b:
        applied_biases.append(("rating preference (duration)", b))
        total_bias += b

    hook_b = _hook_bucket(hook_position)
    if hook_b in learning_profile.rating_bias_by_hook_bucket:
      b = learning_profile.rating_bias_by_hook_bucket[hook_b]
      if b:
        applied_biases.append(("rating preference (hook)", b))
        total_bias += b

    if is_punchline and learning_profile.rating_bias_punchline:
      applied_biases.append(("rating preference (punchline)", learning_profile.rating_bias_punchline))
      total_bias += learning_profile.rating_bias_punchline

    # Weakness/strength pattern matching (bounded, explainable)
    # These do NOT bypass gates; they only nudge scores.
    preferred = learning_profile.preferred_duration_range.get(category)
    if preferred:
      pref_min, pref_max = preferred
      if clip_duration < pref_min and "too_short" in learning_profile.common_failure_patterns:
        applied_biases.append(("menghindari klip terlalu pendek", -2))
        total_bias -= 2
      if clip_duration > pref_max and "too_long" in learning_profile.common_failure_patterns:
        applied_biases.append(("menghindari klip terlalu panjang", -1))
        total_bias -= 1
      if pref_min <= clip_duration <= pref_max and "good_duration" in learning_profile.common_success_patterns:
        applied_biases.append(("sesuai durasi preferensi pengguna", +1))
        total_bias += 1

    # Opening preference
    if "weak_opening" in learning_profile.common_failure_patterns and hook_position > 0.40:
      applied_biases.append(("menghindari opening lemah", -1))
      total_bias -= 1
    if "strong_opening" in learning_profile.common_success_patterns and hook_position <= 0.25:
      applied_biases.append(("opening kuat", +1))
      total_bias += 1

    # Insight preference
    if insight_pass is not None:
      if ("no_insight" in learning_profile.common_failure_patterns or "missing_reasoning" in learning_profile.common_failure_patterns or "no_takeaway" in learning_profile.common_failure_patterns):
        if not insight_pass:
          applied_biases.append(("menghindari kurang insight", -3))
          total_bias -= 3
    if full_insight is not None:
      if full_insight and ("clear_insight" in learning_profile.common_success_patterns or "strong_reasoning" in learning_profile.common_success_patterns or "clear_takeaway" in learning_profile.common_success_patterns):
        applied_biases.append(("insight jelas (feedback)", +2))
        total_bias += 2
  
  # Cap total bias at ±4
  capped_bias = _clamp(total_bias, -4, 4)
  
  # Add explanation to reasons
  if capped_bias != 0:
    if learning_profile is not None and not learning_profile.is_zero():
      # Make trace explicit when user feedback is in play
      if capped_bias > 0:
        reasons.append(f"Disesuaikan dari feedback pengguna (+{capped_bias})")
      else:
        reasons.append(f"Menghindari pola yang sering dinilai buruk ({capped_bias})")
    else:
      if capped_bias > 0:
        reasons.append(f"Editorial preference match (+{capped_bias})")
      else:
        reasons.append(f"Historical pattern mismatch ({capped_bias})")
    
    # Add detailed breakdown if multiple biases
    if len(applied_biases) > 1:
      breakdown = ", ".join(f"{desc}: {bias:+d}" for desc, bias in applied_biases)
      reasons.append(f"Bias breakdown: {breakdown}")
  
  return base_score + capped_bias


# Unit test utilities
def _test_compute_bias():
  """Unit test for bias computation."""
  assert _compute_bias(10, 0) == 3, "100% acceptance should give +3"
  assert _compute_bias(8, 2) == 2, "80% acceptance should give +2"
  assert _compute_bias(5, 5) == 0, "50% acceptance should give 0"
  assert _compute_bias(2, 8) == -2, "20% acceptance should give -2"
  assert _compute_bias(0, 10) == -3, "0% acceptance should give -3"
  assert _compute_bias(0, 0) == 0, "No data should give 0"
  print("✓ All bias computation tests passed")


def _test_bias_profile():
  """Unit test for bias profile."""
  profile = EditorialBiasProfile()
  assert profile.is_zero(), "Empty profile should be zero"
  
  profile.category_bias["educational"] = 2
  assert not profile.is_zero(), "Non-empty profile should not be zero"
  
  profile.feedback_count = 10
  assert not profile.is_zero(), "Profile with data should not be zero"
  
  print("✓ All bias profile tests passed")


def _test_apply_bias():
  """Unit test for bias application."""
  profile = EditorialBiasProfile()
  profile.category_bias["educational"] = 2
  profile.duration_bias["18-30s"] = 1
  profile.punchline_bias = 1
  profile.early_hook_bias = 1
  profile.feedback_count = 50
  
  reasons = []
  
  # Test single bias
  score = apply_editorial_bias(
    base_score=10,
    category="educational",
    clip_duration=25.0,
    is_punchline=False,
    hook_position=0.5,
    bias_profile=profile,
    reasons=reasons,
  )
  assert score == 13, f"Expected 13, got {score}"  # +2 category, +1 duration
  
  # Test capping
  profile.category_bias["punchline"] = 3
  profile.punchline_bias = 3
  reasons = []
  score = apply_editorial_bias(
    base_score=10,
    category="punchline",
    clip_duration=15.0,
    is_punchline=True,
    hook_position=0.1,
    bias_profile=profile,
    reasons=reasons,
  )
  # Would be +3 (cat) +1 (dur) +3 (punch) +1 (hook) = +8, but capped at +4
  assert score == 14, f"Expected 14 (capped), got {score}"
  
  print("✓ All bias application tests passed")


if __name__ == "__main__":
  # Run unit tests
  _test_compute_bias()
  _test_bias_profile()
  _test_apply_bias()
  print("\n✅ All editorial memory tests passed!")
