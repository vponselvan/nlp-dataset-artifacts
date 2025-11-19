# Quick Start - Adversarial Fine-Tuning Experiments

## 📊 Original Experiments (Non-Augmented)

```bash
# Train all 5 models (4-5 hours on GPU)
bash scripts/train_all_adversarial_models.sh

# Evaluate all models (1 hour)
bash scripts/evaluate_all_adversarial_models.sh

# Compare results
python3 scripts/compare_adversarial_models.py

# Create visualizations
python3 scripts/visualize_results.py
```

## 🎨 Augmented Experiments (With Data Augmentation)

```bash
# Generate augmented datasets (5 min)
bash scripts/generate_all_augmented_datasets.sh

# Train all 5 augmented models (4-5 hours on GPU)
bash scripts/train_all_augmented_models.sh

# Evaluate all augmented models (1 hour)
bash scripts/evaluate_all_augmented_models.sh

# Compare augmented results
python3 scripts/compare_augmented_models.py

# Create augmented visualizations
python3 scripts/visualize_augmented_comparison.py
```

## 📁 Project Structure

```
.
├── scripts/
│   ├── train_all_adversarial_models.sh         # Train all 5 original models
│   ├── evaluate_all_adversarial_models.sh      # Evaluate all 5 models
│   ├── generate_all_augmented_datasets.sh      # Generate augmented data
│   ├── train_all_augmented_models.sh           # Train all 5 augmented models
│   ├── evaluate_all_augmented_models.sh        # Evaluate augmented models
│   ├── compare_adversarial_models.py           # Compare original results
│   ├── compare_augmented_models.py             # Compare augmented results
│   ├── visualize_results.py                    # Create plots
│   └── visualize_augmented_comparison.py       # Create augmented plots
├── evaluation/
│   ├── adversarial_squad/                      # Baseline evaluation
│   ├── squad/                                  # Baseline SQuAD
│   ├── adversarial_90_10/                      # Original results (5 ratios)
│   ├── adversarial_80_20/
│   ├── adversarial_70_30/
│   ├── adversarial_60_40/
│   ├── adversarial_50_50/
│   ├── adversarial_90_10_augmented/            # Augmented results (5 ratios)
│   ├── adversarial_80_20_augmented/
│   ├── adversarial_70_30_augmented/
│   ├── adversarial_60_40_augmented/
│   ├── adversarial_50_50_augmented/
│   ├── comparison_results.json                 # Original comparison
│   ├── augmented_comparison_results.json       # Augmented comparison
│   └── plots/                                  # Visualizations (7 plots)
├── data/
│   ├── squad.jsonl                             # Clean SQuAD data
│   ├── addsent_adversarial.jsonl               # Full AddSent adversarial
│   ├── addsent_train.jsonl                     # AddSent training split
│   ├── addsent_eval.jsonl                      # AddSent evaluation split
│   ├── addsent_train_augmented.jsonl           # Augmented AddSent (2,495 examples)
│   ├── mixed_training_90_10.jsonl              # Original mixed datasets (5 ratios)
│   ├── mixed_training_80_20.jsonl
│   ├── mixed_training_70_30.jsonl
│   ├── mixed_training_60_40.jsonl
│   ├── mixed_training_50_50.jsonl
│   ├── mixed_training_90_10_augmented.jsonl    # Augmented mixed datasets (5 ratios)
│   ├── mixed_training_80_20_augmented.jsonl
│   ├── mixed_training_70_30_augmented.jsonl
│   ├── mixed_training_60_40_augmented.jsonl
│   └── mixed_training_50_50_augmented.jsonl
└── trained_model_adversarial_*/                # Trained models
```

## ⏱️ Time Estimates (with GPU)

**Original Experiments:**
- Training: ~4-5 hours (all 5 models)
- Evaluation: ~1 hour
- Total: ~5-6 hours

**Augmented Experiments:**
- Data generation: ~5 minutes
- Training: ~4-5 hours (all 5 models)
- Evaluation: ~1 hour
- Total: ~5-6 hours

## 📖 Documentation
- `START_HERE.md` - Quick start for automation
- `README.md` - Main project documentation
- `RUN_ALL_5_EXPERIMENTS.md` - Detailed experiment guide
- `DISCUSSION_OF_FINDINGS.md` - Results analysis
- `DATA_AUGMENTATION_SUMMARY.md` - Augmentation details
- `IMPROVEMENT_STRATEGIES.md` - Future improvements
