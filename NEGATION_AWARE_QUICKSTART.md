# Negation-Aware Training - Quick Reference

## TL;DR

```bash
# Run complete pipeline (2.5 hours)
cd scripts
./run_negation_aware_training.sh
```

## What It Does

1. **Generates** negation-aware training data (30% augmentation)
2. **Trains** ELECTRA-base with 3× weight on negation examples
3. **Evaluates** on SQuAD (clean) and AddSent (adversarial)

## Expected Impact

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| SQuAD EM | 85.46% | ~85-86% | Maintained |
| AddSent EM | 68.90% | ~73-77% | **+4-8%** |
| Negation Errors | 40.4% | ~25-30% | **-10-15%** |

## Key Files

### Input
- `data/mixed_training_80_20.jsonl` - Base training data
- `data/squad.jsonl` - Clean evaluation data
- `data/addsent_eval.jsonl` - Adversarial evaluation data

### Output
- `data/mixed_training_80_20_negation_aware.jsonl` - Augmented training data
- `trained_model_negation_aware/` - Trained model
- `trained_model_negation_aware/negation_aware_results.json` - Results

### Scripts
- `generate_negation_contrastive_pairs.py` - Step 1: Data augmentation
- `negation_aware_trainer.py` - Step 2: Custom trainer
- `train_negation_aware.py` - Step 3: Training pipeline
- `run_negation_aware_training.sh` - Complete automation

## Manual Steps

### 1. Generate Data (5 minutes)

```bash
python generate_negation_contrastive_pairs.py \
    --input ../data/mixed_training_80_20.jsonl \
    --output ../data/mixed_training_80_20_negation_aware.jsonl \
    --negation-weight 3.0 \
    --augmentation-ratio 0.3
```

**Output**: 13,500 examples (128% of original), 38.5% weighted 3×

### 2. Train Model (2 hours on V100)

```bash
python train_negation_aware.py \
    --train-data ../data/mixed_training_80_20_negation_aware.jsonl \
    --model google/electra-base-discriminator \
    --output-dir ../trained_model_negation_aware \
    --batch-size 16 \
    --num-epochs 3
```

**Output**: Trained model + metrics

### 3. Check Results (10 minutes)

```bash
# View metrics
cat ../trained_model_negation_aware/negation_aware_results.json

# Run error analysis
python advanced_error_analysis.py \
    --predictions ../trained_model_negation_aware/eval_addsent/eval_predictions.jsonl \
    --output ../trained_model_negation_aware/error_analysis.json
```

## Configuration Options

### Data Generation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--negation-weight` | 3.0 | Loss multiplier for negation examples |
| `--augmentation-ratio` | 0.3 | Fraction of examples to augment |
| `--seed` | 42 | Random seed |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 16 | Training batch size |
| `--learning-rate` | 3e-5 | Learning rate |
| `--num-epochs` | 3 | Training epochs |
| `--warmup-ratio` | 0.1 | Warmup ratio |

## Troubleshooting

### Out of Memory
```bash
python train_negation_aware.py --batch-size 8  # Reduce from 16
```

### Too Slow
```bash
# Use smaller augmentation
python generate_negation_contrastive_pairs.py --augmentation-ratio 0.2
```

### Not Improving
```bash
# Increase negation emphasis
python generate_negation_contrastive_pairs.py --negation-weight 5.0

# Or train longer
python train_negation_aware.py --num-epochs 5
```

## How It Works

### Negation Detection (17 patterns)
```python
not, no, never, neither, nor, none, nobody, nothing, nowhere,
n't, don't, doesn't, didn't, won't, wouldn't, can't, couldn't,
without, unless, except, hardly, scarcely, barely, rarely, seldom
```

### Augmentation Types

**Type 1: Additive (Distractor)**
- Original: "Broncos won." → Broncos
- Augmented: "Broncos won. Some say they didn't." → Broncos
- Goal: Ignore negated distractors

**Type 2: Transformative (Modification)**
- Original: "Broncos won." → Broncos
- Augmented: "Broncos didn't win." → [No Answer]
- Goal: Recognize answer changes

### Weighted Loss

```
Regular example:  loss_weight = 1.0
Negation example: loss_weight = 3.0

Effective training: Negation examples seen 3× more
```

## Integration with Report

Add to **Section 7 (Discussion)**:

```latex
\input{negation_aware_section.tex}
```

Or copy content from `Project/negation_aware_section.tex`

## Verification Checklist

After training, verify:

- [ ] Model saved to `trained_model_negation_aware/`
- [ ] Results file exists with SQuAD and AddSent metrics
- [ ] SQuAD EM maintained (~85%)
- [ ] AddSent EM improved (+4-8%)
- [ ] Negation error rate reduced (check error analysis)
- [ ] Training logs show ~38% weighted examples

## Next Steps

1. **Run pipeline**: `./run_negation_aware_training.sh`
2. **Analyze results**: Check metrics and error patterns
3. **Update report**: Add mitigation section
4. **Compare**: Baseline vs negation-aware
5. **Extend**: Try entity-aware or numeric-aware training

## Resources

- **Full Documentation**: `NEGATION_AWARE_TRAINING.md`
- **Error Analysis**: `analysis/structured_error_analysis_final.json`
- **Baseline Results**: `evaluation/electra_base_80_20_results.json`
- **Report Section**: `Project/negation_aware_section.tex`

---

**Time Budget**: 2.5 hours total
**GPU Required**: V100 or better (16GB+ VRAM)
**Status**: Ready to run
