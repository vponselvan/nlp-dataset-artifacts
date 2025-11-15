# Lightning.ai Workflow Guide

## Your Current Situation

You trained a model on Lightning.ai using:
```bash
python3 run.py --do_train --task qa --dataset squad --output_dir ./trained_model/
```

✅ **Training completed** (3 epochs, as shown in your `trainer_state.json`)  
❌ **No evaluation ran** (because you didn't include `--do_eval`)

## Quick Fix: Evaluate Your Trained Model

### On Lightning.ai, run:

```bash
python3 run.py \
  --do_eval \
  --task qa \
  --dataset squad \
  --model ./trained_model/ \
  --output_dir ./eval_results/ \
  --per_device_eval_batch_size 32
```

Or use the script I created:
```bash
./evaluate_model.sh ./trained_model/ ./eval_results/
```

This will:
1. Load your trained model from `./trained_model/`
2. Evaluate it on SQuAD dev set
3. Create `eval_results/eval_metrics.json` with your EM and F1 scores
4. Create `eval_results/eval_predictions.jsonl` with all predictions

### View Your Results:

```bash
cat eval_results/eval_metrics.json
```

You should see something like:
```json
{
  "eval_exact_match": 78.5,
  "eval_f1": 86.2,
  "eval_runtime": 120.45,
  ...
}
```

## For Future Training Runs

### Option 1: Train + Evaluate in One Command
```bash
python3 run.py \
  --do_train \
  --do_eval \
  --task qa \
  --dataset squad \
  --model google/electra-small-discriminator \
  --output_dir ./trained_model/ \
  --per_device_train_batch_size 16 \
  --num_train_epochs 3 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 500 \
  --seed 42
```

### Option 2: Train First, Then Evaluate Separately
```bash
# Step 1: Train
python3 run.py \
  --do_train \
  --task qa \
  --dataset squad \
  --model google/electra-small-discriminator \
  --output_dir ./trained_model/ \
  --per_device_train_batch_size 16 \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --seed 42

# Step 2: Evaluate
python3 run.py \
  --do_eval \
  --task qa \
  --dataset squad \
  --model ./trained_model/ \
  --output_dir ./eval_results/ \
  --per_device_eval_batch_size 32
```

## Understanding Your Training Results

From your `trainer_state.json`:
- ✅ Training completed: **3.0 epochs** (32,895 steps)
- ✅ Final training loss: **~0.73-0.79** (last few steps)
- ✅ Model saved successfully
- ⏱️ Batch size used: **8** (you might have hit memory limits)

**Next step:** Run the evaluation command above to get your baseline EM and F1 scores!

## Files in Your Trained Model Directory

After training, `./trained_model/` should contain:
```
trained_model/
├── config.json              # Model configuration
├── pytorch_model.bin        # Trained weights ⭐
├── tokenizer_config.json    # Tokenizer settings
├── vocab.txt               # Vocabulary
├── trainer_state.json      # Training history
└── training_args.bin       # Training arguments
```

The key file is `pytorch_model.bin` - this is your trained model!

## Common Lightning.ai Tips

### Check GPU Usage:
```bash
nvidia-smi
```

### Monitor Training in Real-time:
```bash
tail -f trained_model/trainer_state.json
```

### If You Hit Memory Issues:
Reduce batch size:
```bash
--per_device_train_batch_size 8  # or even 4
```

### Download Results Locally:
After evaluation, download these files:
- `eval_results/eval_metrics.json` (your scores!)
- `eval_results/eval_predictions.jsonl` (predictions for analysis)
- `trained_model/` (entire model directory for later use)
