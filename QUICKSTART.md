# Quick Start Guide: Fine-tuning ELECTRA on SQuAD

## Step 1: Run Baseline Training

You have two options to start training:

### Option A: Use the training script (Recommended)
```bash
cd /Users/vrajasingh/Documents/MyDocs/Masters/Courses/NLP/Homeworks/nlp-dataset-artifacts
./train_baseline.sh
```

### Option B: Run directly with Python
```bash
cd /Users/vrajasingh/Documents/MyDocs/Masters/Courses/NLP/Homeworks/nlp-dataset-artifacts

python3 run.py \
  --do_train \
  --do_eval \
  --task qa \
  --dataset squad \
  --model google/electra-small-discriminator \
  --output_dir ./experiments/baseline_squad \
  --per_device_train_batch_size 16 \
  --num_train_epochs 3 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --seed 42
```

## What to Expect

**Training time:** ~2-3 hours on GPU, longer on CPU
**Expected performance on SQuAD dev set:**
- Exact Match: ~78%
- F1 Score: ~86%

## Output Files

After training completes, check these files:

```
experiments/baseline_squad/
├── config.json                    # Model configuration
├── pytorch_model.bin              # Trained model weights
├── trainer_state.json             # Training state/history
├── eval_metrics.json              # ⭐ Performance metrics (EM, F1)
├── eval_predictions.jsonl         # Individual predictions on dev set
└── checkpoint-*/                  # Intermediate checkpoints
```

## Verify Results

View your performance metrics:
```bash
cat experiments/baseline_squad/eval_metrics.json
```

You should see output like:
```json
{
  "eval_exact_match": 78.5,
  "eval_f1": 86.2,
  "eval_runtime": 120.45,
  ...
}
```

## Troubleshooting

### Out of Memory Error
Reduce batch size:
```bash
--per_device_train_batch_size 8
```

### Running on CPU
Add the `--no_cuda` flag:
```bash
python3 run.py --no_cuda --do_train ...
```

### Monitor Training Progress
Training logs show up in terminal. Look for:
- Loss decreasing over time
- Evaluation scores after each epoch

## Next Steps After Baseline

1. ✅ **Record baseline metrics** in `EXPERIMENT_LOG.md`
2. 📊 **Analyze predictions** in `eval_predictions.jsonl`
3. 🎯 **Create adversarial examples** (Phase 2)
4. 🔍 **Evaluate on adversarial dataset** (Phase 3)
5. 🛡️ **Implement mitigation strategies** (Phase 4)

---

## Quick Commands Reference

```bash
# Check training progress (if running in background)
tail -f experiments/baseline_squad/trainer_state.json

# View results
cat experiments/baseline_squad/eval_metrics.json | python3 -m json.tool

# Count predictions
wc -l experiments/baseline_squad/eval_predictions.jsonl

# Load model for inference later
# In Python:
# from transformers import AutoModelForQuestionAnswering
# model = AutoModelForQuestionAnswering.from_pretrained("./experiments/baseline_squad")
```
