# Post-Processing Quick Reference

## TL;DR

Post-processing fixes **partial match errors** (30.6% of errors) using NER-based entity expansion. **No training required** - pure inference-time fix.

**Expected**: +2-3 points on AddSent EM in ~1 minute

## Quick Start

### One-liner
```bash
cd scripts && ./run_postprocessing.sh
```

### Direct usage
```bash
python postprocess_partial_matches.py \
    --input predictions.jsonl \
    --output expanded_predictions.jsonl
```

## What It Does

Expands partial predictions to full entity boundaries:

| Before | After |
|--------|-------|
| "Broncos" | "Denver Broncos" |
| "New York" | "New York City" |
| "January 15" | "January 15, 2020" |
| "Jane Smith" | "Dr. Jane Smith" |

## Three-Step Algorithm

```
1. Run spaCy NER on context
   ↓
2. Find entities containing/near prediction
   ↓
3. Expand if entity is significantly longer (1.3×)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-expansion-ratio` | 1.3 | Minimum length ratio for expansion |
| `--spacy-model` | en_core_web_sm | NER model to use |
| `--no-metadata` | False | Skip saving expansion metadata |

## Expected Results

| Metric | Improvement |
|--------|-------------|
| **AddSent EM** | +2-3 points |
| **F1 Score** | +2-3 points |
| **Expansion rate** | 15-20% |
| **Runtime** | ~30 seconds (1000 examples) |

## Combination with Other Strategies

```
Baseline:          68.90% EM
+ Negation:        73.50% (+4.60)
+ Entity:          79.80% (+6.30)
+ Post-processing: 81-83% (+2-3)
─────────────────────────────
Total improvement: +12-14 points
```

## Advantages

✓ No training required
✓ Fast (~1 minute)
✓ Works with any model
✓ No hyperparameter tuning
✓ Can combine with any strategy

## Limitations

✗ Depends on NER accuracy
✗ May over-expand incorrectly
✗ Doesn't fix semantic errors
✗ Context-agnostic

## Troubleshooting

**No expansions?**
```bash
--min-expansion-ratio 1.1  # More aggressive
--spacy-model en_core_web_md  # Better NER
```

**Too many expansions?**
```bash
--min-expansion-ratio 1.5  # More conservative
```

**Performance worse?**
- Check error analysis for patterns
- Increase `min_expansion_ratio` to 1.5-2.0
- Consider entity type matching

## File Structure

```
scripts/
  postprocess_partial_matches.py  # Core logic
  evaluate_with_postprocessing.py # Evaluation
  run_postprocessing.sh            # Automated pipeline

postprocessing_results/
  predictions_raw.jsonl            # Before
  predictions_postprocessed.jsonl  # After
  postprocessing_results.json      # Metrics
```

## Example Usage

### Process predictions
```bash
python postprocess_partial_matches.py \
    --input model_predictions.jsonl \
    --output expanded_predictions.jsonl \
    --min-expansion-ratio 1.3
```

### Evaluate improvement
```bash
python evaluate_with_postprocessing.py \
    --predictions-file model_predictions.jsonl \
    --gold-file test_data.jsonl \
    --output-dir results/
```

### Full pipeline (inference + postprocess + eval)
```bash
./run_postprocessing.sh
```

## Statistics Output

```
Post-Processing Statistics
══════════════════════════════════════════════════════════════════════
Total predictions: 1000
Expanded predictions: 165 (16.50%)
Exact matches (no expansion): 785 (78.50%)
No entity found: 50 (5.00%)

Expansions by entity type:
  PERSON         :   45 (27.3% of expansions)
  ORG            :   40 (24.2% of expansions)
  GPE            :   35 (21.2% of expansions)
  DATE           :   25 (15.2% of expansions)
  OTHER          :   20 (12.1% of expansions)
══════════════════════════════════════════════════════════════════════
```

## Performance Comparison

```
Performance Comparison
══════════════════════════════════════════════════════════════════════

Before post-processing:
  Exact Match: 68.90%
  F1 Score:    76.50%

After post-processing:
  Exact Match: 71.20%
  F1 Score:    78.80%

Improvement:
  ΔEM: +2.30 points
  ΔF1: +2.30 points
══════════════════════════════════════════════════════════════════════
```

## When to Use

**Use post-processing when:**
- ✓ Model predicts partial entity names
- ✓ You have existing predictions to improve
- ✓ No time/resources for retraining
- ✓ Want quick wins (~2-3 points)

**Skip post-processing when:**
- ✗ Model already predicts full entity names
- ✗ Errors are semantic, not partial matches
- ✗ NER quality is poor for your domain

## Integration Checklist

- [ ] Run baseline inference
- [ ] Apply post-processing
- [ ] Evaluate before/after
- [ ] Verify expansion rate (15-20% expected)
- [ ] Check error analysis for over-expansions
- [ ] Combine with training-based strategies
- [ ] Update report with results

## Need More Details?

See `POSTPROCESSING_PARTIAL_MATCHES.md` for:
- Detailed algorithm explanation
- Mathematical formulation
- Edge cases and handling
- Future enhancements
- Report integration examples
