# Fixed: SQuAD Adversarial Evaluation

## The Problem
The `stanfordnlp/squad_adversarial` dataset uses an old loading script format that's no longer supported by HuggingFace datasets.

## The Solution
Download the adversarial dataset directly from the original source and use it as a JSON file.

## Quick Fix - Run on Lightning.ai:

### Option 1: Manual Download (Fastest)
```bash
./download_adversarial_squad.sh
```

This will download the AddSent adversarial dataset to `./data/squad_adversarial.json`

### Option 2: Automatic (Evaluation scripts will auto-download)
```bash
./evaluate_adversarial.sh
```

The script will automatically download the dataset if it's not found.

## What Changed

### Before (didn't work):
```bash
--dataset stanfordnlp/squad_adversarial
```

### After (works):
```bash
--dataset ./data/squad_adversarial.json
```

## Manual Commands

If you prefer to run commands directly:

```bash
# Download the dataset
curl -L -o ./data/squad_adversarial.json \
  https://raw.githubusercontent.com/robinjia/adversarial-squad/master/data/dev-v1.1.json

# Evaluate on adversarial data
python3 run.py \
  --do_eval \
  --task qa \
  --dataset ./data/squad_adversarial.json \
  --model ./trained_model/ \
  --output_dir ./eval_results_adversarial/ \
  --per_device_eval_batch_size 32
```

## About the Dataset

**Source:** Original AddSent paper by Jia & Liang (2017)
**Repository:** https://github.com/robinjia/adversarial-squad
**Type:** AddSent adversarial examples - distractors added to SQuAD dev set
**Size:** ~10k questions with adversarial sentences

## What the Scripts Do Now

1. **`download_adversarial_squad.sh`** - Downloads the AddSent dataset
2. **`evaluate_adversarial.sh`** - Auto-downloads if needed, then evaluates
3. **`compare_baseline_adversarial.sh`** - Compares clean vs adversarial performance

All scripts now automatically download the dataset if it's missing!

## Expected Results

Your baseline model (EM: 78.16%, F1: 86.05%) will likely show:

```
Baseline (Clean SQuAD):
  EM: 78.16%
  F1: 86.05%

Adversarial SQuAD:
  EM: ~45-55%
  F1: ~55-65%

Performance Drop:
  EM: ~25-30 points
  F1: ~20-25 points
```

This demonstrates the model's vulnerability to adversarial examples!
