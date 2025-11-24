# Negation-Aware Contrastive Training

## Overview

This implementation addresses the **#1 error pattern** identified in our error analysis: **Negation Confusion (40.4% of errors)**.

The strategy uses weighted contrastive learning to teach the model to distinguish between affirmative and negated statements, forcing it to pay 3× more attention to negation cues.

## Problem Statement

**Error Pattern**: Models fail to recognize negation words like "not", "never", "didn't"  
**Frequency**: 40.4% of adversarial errors  
**Impact**: Model answers based on surface similarity, ignoring critical negation  

**Example**:
```
Context: "The Broncos won the Super Bowl."
Added: "However, some sources claim the Broncos did not win."
Question: "Who won the Super Bowl?"
Ground Truth: Broncos
Model Prediction: (wrong entity or "did not win")  ← Confused by negation
```

## Solution: 3-Step Training Strategy

### Step 1: Generate Contrastive Pairs

**Script**: `generate_negation_contrastive_pairs.py`

Creates two types of negation-aware training examples:

1. **Additive Negation** (like AddSent attacks)
   - Original: "The Broncos won." → Answer: Broncos
   - Augmented: "The Broncos won. However, some claim they didn't win." → Answer: Broncos
   - **Goal**: Teach model to ignore negated distractors

2. **Transformative Negation** (modified context)
   - Original: "The Broncos won." → Answer: Broncos
   - Augmented: "The Broncos didn't win." → Answer: [No Answer]
   - **Goal**: Teach model to recognize when negation changes the answer

**Key Features**:
- Identifies 17+ negation patterns (not, never, n't, etc.)
- Rule-based templates for robust augmentation
- Marks examples with `loss_weight: 3.0` for training
- Augments 30% of positive examples by default

**Output Statistics**:
```
Original examples: 10,570
  - Already containing negation: ~800 (7.6%)
  - Positive examples: ~9,770 (92.4%)

Augmented examples: ~2,930
  - Additive negation: ~1,465
  - Transformative negation: ~1,465

Total output: ~13,500 (128% of original)
Negation examples (weighted 3x): ~5,200 (38.5%)
```

### Step 2: Weighted Loss Training

**Script**: `negation_aware_trainer.py`

Custom `NegationAwareQATrainer` that:

1. **Reads loss weights** from training examples
2. **Computes per-example loss** (start + end positions)
3. **Applies 3× weight** to negation examples
4. **Maintains standard evaluation**

**Loss Formula**:
```
For regular examples:     loss_weight = 1.0
For negation examples:    loss_weight = 3.0

Total Loss = (Σ loss_i × weight_i) / batch_size
```

This forces the model to prioritize learning from negation examples, effectively seeing them 3× during training.

**Trainer Features**:
- Compatible with HuggingFace `Trainer` API
- Logs weight statistics every 100 steps
- Supports both QA and classification tasks
- Automatic gradient scaling

### Step 3: Train and Evaluate

**Script**: `train_negation_aware.py`

Complete training pipeline:

1. **Load** negation-augmented data with weights
2. **Train** ELECTRA-base with weighted loss
3. **Evaluate** on both clean (SQuAD) and adversarial (AddSent)
4. **Log** performance metrics and weight statistics

**Training Configuration**:
```python
Model: google/electra-base-discriminator (110M params)
Batch size: 16
Learning rate: 3e-5
Epochs: 3
Mixed precision: FP16 (faster training)
```

## Quick Start

### 1. Generate Negation-Aware Data

```bash
cd scripts

python generate_negation_contrastive_pairs.py \
    --input ../data/mixed_training_80_20.jsonl \
    --output ../data/mixed_training_80_20_negation_aware.jsonl \
    --negation-weight 3.0 \
    --augmentation-ratio 0.3
```

**Arguments**:
- `--input`: Original training data (SQuAD + AddSent 80-20 mix)
- `--output`: Output path for augmented data
- `--negation-weight`: Loss multiplier for negation examples (default: 3.0)
- `--augmentation-ratio`: Fraction of positive examples to augment (default: 0.3)

### 2. Train Model

```bash
python train_negation_aware.py \
    --train-data ../data/mixed_training_80_20_negation_aware.jsonl \
    --model google/electra-base-discriminator \
    --output-dir ../trained_model_negation_aware \
    --batch-size 16 \
    --num-epochs 3
```

**Arguments**:
- `--train-data`: Negation-augmented training data
- `--model`: Base model (ELECTRA-base recommended)
- `--output-dir`: Where to save trained model
- `--batch-size`: Batch size per GPU
- `--num-epochs`: Training epochs

### 3. Run Complete Pipeline

```bash
cd scripts
./run_negation_aware_training.sh
```

This script:
1. Generates negation pairs (Step 1)
2. Trains with weighted loss (Step 2)
3. Evaluates on SQuAD and AddSent (Step 3)
4. Saves all results and metrics

**Expected Runtime**:
- Data generation: ~5 minutes
- Training (3 epochs on V100): ~2 hours
- Evaluation: ~10 minutes
- **Total**: ~2.5 hours

## Expected Results

### Baseline (ELECTRA-base 80-20)

From our experiments:
```
SQuAD EM:   85.46%
AddSent EM: 68.90%
Drop:       -16.56%
```

### Target (Negation-Aware)

**Conservative Estimate** (10% improvement on 40.4% of errors):
```
SQuAD EM:   ~85-86% (maintain clean performance)
AddSent EM: ~73-75% (+4-6% absolute)
Drop:       ~11-13% (reduction in vulnerability)
```

**Optimistic Estimate** (15% improvement on negation errors):
```
SQuAD EM:   ~85-87%
AddSent EM: ~75-77% (+6-8% absolute)
Drop:       ~8-12%
```

### Verification

Check negation-specific improvement:
```bash
# Run error analysis on trained model
python ../scripts/advanced_error_analysis.py \
    --predictions ../trained_model_negation_aware/eval_addsent/eval_predictions.jsonl \
    --output ../trained_model_negation_aware/error_analysis.json
```

Look for:
- **Negation Confusion**: Should drop from 40.4% → 25-30%
- **Overall Error Rate**: Should improve by 4-8%
- **Clean Performance**: Should maintain ~85%

## Implementation Details

### Data Format

**Input** (standard SQuAD format):
```json
{
  "id": "56be4db0acb8001400a502ee",
  "context": "The Broncos won Super Bowl 50...",
  "question": "Who won Super Bowl 50?",
  "answers": {
    "text": ["Broncos"],
    "answer_start": [12]
  }
}
```

**Output** (with negation awareness):
```json
{
  "id": "56be4db0acb8001400a502ee",
  "context": "The Broncos won Super Bowl 50...",
  "question": "Who won Super Bowl 50?",
  "answers": {"text": ["Broncos"], "answer_start": [12]},
  "loss_weight": 1.0,
  "is_negation_example": false
}
```

**Augmented (additive negation)**:
```json
{
  "id": "56be4db0acb8001400a502ee_neg_additive",
  "context": "The Broncos won Super Bowl 50. However, some claim they didn't win.",
  "question": "Who won Super Bowl 50?",
  "answers": {"text": ["Broncos"], "answer_start": [12]},
  "loss_weight": 3.0,
  "is_negation_example": true,
  "negation_type": "additive"
}
```

**Augmented (transformative negation)**:
```json
{
  "id": "56be4db0acb8001400a502ee_neg_transform",
  "context": "The Broncos didn't win Super Bowl 50...",
  "question": "Who won Super Bowl 50?",
  "answers": {"text": [], "answer_start": []},
  "is_impossible": true,
  "loss_weight": 3.0,
  "is_negation_example": true,
  "negation_type": "transformative"
}
```

### Negation Detection

**17 Negation Patterns**:
```python
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nor', 'none',
    'nobody', 'nothing', 'nowhere',
    "n't", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "can't", "couldn't", "shouldn't",
    'without', 'unless', 'except',
    'hardly', 'scarcely', 'barely', 'rarely', 'seldom',
}
```

**Template Transformations**:
```python
"is" → "is not" / "isn't"
"was" → "was not" / "wasn't"
"did" → "did not" / "didn't"
"can" → "cannot" / "can't"
... (6 templates total)
```

### Loss Computation

**Standard QA Loss**:
```python
start_loss = CrossEntropy(start_logits, start_positions)
end_loss = CrossEntropy(end_logits, end_positions)
loss = (start_loss + end_loss) / 2
```

**Negation-Aware Loss**:
```python
# Per-example losses
start_loss = CrossEntropy(start_logits, start_positions, reduction='none')
end_loss = CrossEntropy(end_logits, end_positions, reduction='none')
total_loss = (start_loss + end_loss) / 2

# Apply weights
weighted_loss = total_loss * loss_weights  # [batch_size]

# Final loss
loss = weighted_loss.mean()
```

**Effect**:
- Regular example (weight=1.0): Normal gradient
- Negation example (weight=3.0): 3× larger gradient
- Model "sees" negation examples 3× more often

## Integration with Report

### Section to Add

Add a new subsection in **Section 7 (Discussion)**:

```latex
\subsection{Mitigation Strategy: Negation-Aware Contrastive Training}

To address the dominant error pattern (Negation Confusion, 40.4\% of errors),
we implement a targeted mitigation strategy based on weighted contrastive learning.

\subsubsection{Method}

The approach consists of three steps:

\textbf{Step 1: Contrastive Pair Generation.} We augment the training data
with two types of negation examples:
\begin{itemize}
    \item \textbf{Additive negation}: Add negated distractors similar to
          AddSent attacks to teach robustness to negated claims.
    \item \textbf{Transformative negation}: Modify original statements with
          negation to teach recognition of answer changes.
\end{itemize}

\textbf{Step 2: Weighted Loss Training.} We modify the training objective
to assign 3× higher loss weight to negation examples:
\begin{equation}
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot \ell_i
\end{equation}
where $w_i = 3.0$ for negation examples and $w_i = 1.0$ otherwise.

\textbf{Step 3: Evaluation.} We evaluate on both clean (SQuAD) and
adversarial (AddSent) datasets to measure effectiveness.

\subsubsection{Results}

[Include table with baseline vs negation-aware results]
[Include error analysis showing reduction in negation confusion]

The negation-aware training achieves X\% improvement on AddSent while
maintaining Y\% on SQuAD, with negation confusion errors reduced from
40.4\% to Z\%.
```

### Table Template

```latex
\begin{table}[h]
\centering
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{SQuAD} & \textbf{AddSent} & \textbf{Drop} & \textbf{Neg. Err.} \\
\midrule
ELECTRA-base 80-20 & 85.46 & 68.90 & -16.56 & 40.4\% \\
+ Negation-Aware & XX.XX & YY.YY & -ZZ.ZZ & WW.W\% \\
\midrule
Improvement & +A.AA & +B.BB & +C.CC & -D.D\% \\
\bottomrule
\end{tabular}
\caption{Impact of Negation-Aware Contrastive Training on model performance
         and error patterns. Neg. Err. shows percentage of errors due to
         negation confusion.}
\label{tab:negation_aware_results}
\end{table}
```

## Files Created

### Core Implementation
- `scripts/generate_negation_contrastive_pairs.py` - Data augmentation (Step 1)
- `scripts/negation_aware_trainer.py` - Custom trainer with weighted loss (Step 2)
- `scripts/train_negation_aware.py` - Training pipeline (Step 3)
- `scripts/run_negation_aware_training.sh` - Complete automation script

### Documentation
- `NEGATION_AWARE_TRAINING.md` - This file (comprehensive guide)

### Output Files (after running)
- `data/mixed_training_80_20_negation_aware.jsonl` - Augmented training data
- `data/mixed_training_80_20_negation_aware_stats.json` - Augmentation statistics
- `trained_model_negation_aware/` - Trained model checkpoint
- `trained_model_negation_aware/negation_aware_results.json` - Full results

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size
```bash
python train_negation_aware.py --batch-size 8  # Instead of 16
```

### Issue: Data augmentation too slow

**Solution**: Reduce augmentation ratio
```bash
python generate_negation_contrastive_pairs.py --augmentation-ratio 0.2  # Instead of 0.3
```

### Issue: Training too slow

**Solution**: Use gradient accumulation
```bash
# Modify training_args in train_negation_aware.py:
gradient_accumulation_steps=2  # Effective batch = 16 * 2 = 32
```

### Issue: Model not improving on negation

**Solutions**:
1. Increase negation weight: `--negation-weight 5.0`
2. Increase augmentation: `--augmentation-ratio 0.5`
3. Train longer: `--num-epochs 5`

## Next Steps

1. **Run the pipeline**:
   ```bash
   cd scripts
   ./run_negation_aware_training.sh
   ```

2. **Analyze results**:
   ```bash
   # Check metrics
   cat ../trained_model_negation_aware/negation_aware_results.json
   
   # Run error analysis
   python advanced_error_analysis.py \
       --predictions ../trained_model_negation_aware/eval_addsent/eval_predictions.jsonl
   ```

3. **Update report**:
   - Add subsection in Discussion
   - Include results table
   - Add qualitative examples
   - Update conclusion with mitigation findings

4. **Further improvements**:
   - Try other mitigation strategies (entity-aware, numeric-aware)
   - Combine multiple strategies
   - Experiment with different weight values
   - Use LLM for more sophisticated negation augmentation

## References

- **Error Analysis**: `analysis/structured_error_analysis_final.json`
- **Baseline Results**: `evaluation/electra_base_80_20_results.json`
- **Original Paper**: Our project report, Section 5.2 (Error Taxonomy)

## Contact & Support

For questions or issues:
1. Check this documentation
2. Review error logs in `trained_model_negation_aware/logs/`
3. Verify data format in `data/mixed_training_80_20_negation_aware.jsonl`

---

**Last Updated**: November 23, 2025  
**Status**: Ready to run  
**Estimated Time**: ~2.5 hours full pipeline
