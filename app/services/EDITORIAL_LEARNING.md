# Editorial Learning Layer

## Overview

The Editorial Learning Layer is a deterministic, heuristic-based system that adapts highlight scoring based on historical editorial feedback (accept/reject decisions). It **complements** the existing scoring heuristics without replacing them.

**Key Principle**: Learning is ADDITIVE (biasing), not replacing core logic.

## Philosophy

### What it does:
- Observes which clips humans ACCEPT or REJECT
- Aggregates patterns by category, duration, punchline presence, and hook position
- Computes simple integer biases (-3 to +3) based on acceptance ratios
- Applies capped bias (±4 max) to clip scores

### What it does NOT do:
- Does NOT use machine learning or AI
- Does NOT call external APIs
- Does NOT override core editorial gates
- Does NOT resurrect clips that fail completeness checks
- Does NOT introduce randomness or non-deterministic behavior

## Architecture

### Components

1. **`editorial_memory.py`** - Learning module
   - `EditorialBiasProfile` - Data structure for learned biases
   - `load_editorial_bias()` - Loads and computes biases from database
   - `apply_editorial_bias()` - Applies capped bias to scores

2. **Integration in `highlight_ai.py`**
   - Bias loaded once at start of `detect_highlights()`
   - Applied after base_score calculation, before validation gates
   - Explanations added to reason strings

### Bias Computation

```python
bias = clamp(round((accepted - rejected) / total * 3), -3, 3)
```

**Examples:**
- 10 accepted, 0 rejected → +3 bias (100% acceptance)
- 8 accepted, 2 rejected → +2 bias (80% acceptance)
- 5 accepted, 5 rejected → 0 bias (50/50)
- 2 accepted, 8 rejected → -2 bias (20% acceptance)
- 0 accepted, 10 rejected → -3 bias (0% acceptance)

### Aggregation Dimensions

1. **Category** (educational, punchline, insight, etc.)
2. **Duration buckets** (<12s, 12-18s, 18-30s, >30s)
3. **Punchline presence** (yes/no)
4. **Hook position** (early in first 25% vs late)

### Bias Application

```python
final_score = base_score + capped_bias

# Where capped_bias is sum of:
#   + category_bias
#   + duration_bias
#   + punchline_bias (if applicable)
#   + early_hook_bias (if applicable)
# Capped at ±4 total
```

## Usage

### Basic Usage (No Learning)

```python
from app.services.highlight_ai import detect_highlights

# Without database session - returns zero bias
results = detect_highlights(transcript)
```

### With Learning (Database Session)

```python
from app.services.highlight_ai import detect_highlights
from app.database.db import SessionLocal

# With database session - loads historical feedback
db = SessionLocal()
try:
    results = detect_highlights(transcript, db_session=db)
finally:
    db.close()
```

### Manual Bias Testing

```python
from app.services.editorial_memory import (
    EditorialBiasProfile,
    apply_editorial_bias
)

# Create mock bias profile
bias_profile = EditorialBiasProfile()
bias_profile.category_bias['educational'] = 2
bias_profile.duration_bias['18-30s'] = 2
bias_profile.early_hook_bias = 1

# Apply bias
reasons = []
final_score = apply_editorial_bias(
    base_score=10,
    category='educational',
    clip_duration=25.0,
    is_punchline=False,
    hook_position=0.15,
    bias_profile=bias_profile,
    reasons=reasons
)
# final_score = 14 (base 10 + bias 4)
# reasons = ["Editorial preference match (+4)", ...]
```

## Data Model

### Current Implementation (Proxy)

Since there's no explicit feedback column yet, we use **score as a proxy**:
- `score >= 8` → ACCEPTED (high quality)
- `score < 5` → REJECTED (low quality)
- `5-7` → NEUTRAL (not used for learning)

**TODO**: Add explicit `accepted` boolean column to `highlights` table for real feedback.

### Required Schema Changes (Future)

```sql
ALTER TABLE highlights ADD COLUMN accepted BOOLEAN DEFAULT NULL;
ALTER TABLE highlights ADD COLUMN rejected BOOLEAN DEFAULT NULL;
ALTER TABLE highlights ADD COLUMN feedback_timestamp TIMESTAMP;
```

## Guarantees

### Safety Guarantees

1. **No Breaking Changes**: Returns zero bias if database unavailable
2. **Deterministic**: Same input always produces same output
3. **Pure Functions**: No side effects, fully testable
4. **Error Resilient**: Catches exceptions and falls back to zero bias
5. **Gate Preservation**: Never bypasses core editorial requirements

### Quality Guarantees

1. **Minimum Sample Size**: Requires 3+ feedback records per pattern
2. **Bias Capping**: Total bias limited to ±4 points
3. **Transparent**: Adds explanation to reason strings
4. **Gradual Adaptation**: Biases evolve slowly with feedback

## Testing

### Unit Tests

```bash
cd /Users/a/Documents/www/highlighter
python3 -m app.services.editorial_memory
```

Expected output:
```
✓ All bias computation tests passed
✓ All bias profile tests passed
✓ All bias application tests passed

✅ All editorial memory tests passed!
```

### Integration Test

```bash
python3 -c "
from app.services.highlight_ai import detect_highlights

transcript = [
    {'start': 0.0, 'duration': 3.0, 'text': 'ini penting'},
    {'start': 3.0, 'duration': 4.0, 'text': '1 kilogram = 1000 gram'},
    ...
]

results = detect_highlights(transcript)
print(f'Detected {len(results)} clips')
"
```

## Examples

### Scenario: Educational Content Preferred

Historical data shows educational clips (18-30s) get accepted more often:

```python
bias_profile.category_bias['educational'] = 2
bias_profile.duration_bias['18-30s'] = 2
bias_profile.early_hook_bias = 1
```

**Result**: Educational clips (18-30s, early hook) get +4 bias boost.

**Impact**:
- Before: Educational clip scores 10 → barely passes threshold
- After: Educational clip scores 14 → confidently passes

### Scenario: Short Clips Rejected

Historical data shows short clips (<12s) get rejected often:

```python
bias_profile.duration_bias['<12s'] = -2
```

**Result**: Short clips get -2 bias penalty.

**Impact**:
- Before: Short clip scores 6 → passes minimum threshold
- After: Short clip scores 4 → fails minimum threshold

### Scenario: Punchlines Always Win

Historical data shows punchlines get universally accepted:

```python
bias_profile.category_bias['punchline'] = 3
bias_profile.punchline_bias = 3
```

**Result**: Punchlines get maximum bias (+4 after capping).

**Impact**: Punchlines consistently rank higher in final results.

## Monitoring & Metrics

### Key Metrics to Track

1. **Bias Coverage**: % of clips with non-zero bias
2. **Bias Distribution**: Histogram of applied bias values
3. **Acceptance Rate**: Before vs after bias implementation
4. **Feedback Count**: Total historical records used
5. **Pattern Strength**: Average |bias| per dimension

### Logging Recommendations

```python
# Log when bias is applied
if final_score != base_score:
    logger.info(f"Bias applied: {final_score - base_score:+d} | "
                f"Category: {category} | Duration: {clip_duration:.1f}s")
```

## Limitations

1. **Cold Start**: Requires historical data to function (fallback: zero bias)
2. **Proxy Feedback**: Currently uses score as proxy for accept/reject
3. **Static Patterns**: Doesn't adapt to changing content trends
4. **Category Dependent**: New categories start with zero bias
5. **Minimum Samples**: Needs 3+ records per pattern for reliability

## Future Enhancements

### Short Term
- [ ] Add explicit `accepted` column to database
- [ ] Implement feedback collection UI
- [ ] Add bias strength confidence scoring
- [ ] Track bias effectiveness metrics

### Medium Term
- [ ] Time-weighted feedback (recent feedback counts more)
- [ ] Cross-pattern interactions (e.g., educational + early hook)
- [ ] Adaptive bias decay (reduce old patterns over time)
- [ ] Multi-dimensional pattern learning

### Long Term
- [ ] A/B testing framework for bias validation
- [ ] Automated bias tuning based on acceptance rates
- [ ] Pattern discovery (find new successful combinations)
- [ ] Editor-specific bias profiles

## Troubleshooting

### Problem: Bias not being applied

**Check:**
1. Is `db_session` passed to `detect_highlights()`?
2. Does database have feedback records?
3. Are there 3+ records per pattern?
4. Check logs for "Could not load editorial bias" warning

### Problem: Bias too aggressive

**Solutions:**
1. Reduce bias cap from ±4 to ±3 or ±2
2. Increase minimum sample requirement from 3 to 5
3. Add bias decay over time
4. Filter outliers in feedback data

### Problem: Unexpected clip rankings

**Debug:**
1. Check reason strings for bias explanations
2. Log base_score vs final_score delta
3. Verify bias_profile values
4. Test with zero-bias mode (db_session=None)

## Credits

**Design**: Deterministic learning layer for editorial preference alignment  
**Implementation**: Pure Python, no external dependencies  
**Integration**: Additive bias system preserving core heuristics  
**Philosophy**: Gradual adaptation without opacity  

---

**Last Updated**: January 11, 2026  
**Version**: 1.0.0  
**Status**: Production Ready
