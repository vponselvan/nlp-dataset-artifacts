# Quick Start - Adversarial Fine-Tuning Experiments

## 📊 Run All 5 Experiments

```bash
# Check status
bash check_status.sh

# Run experiments (on Colab with A100)
bash scripts/train_adversarial_90_10.sh && bash scripts/evaluate_adversarial_90_10.sh
bash scripts/train_adversarial_80_20.sh && bash scripts/evaluate_adversarial_80_20.sh
bash scripts/train_adversarial_70_30.sh && bash scripts/evaluate_adversarial_70_30.sh
bash scripts/train_adversarial_60_40.sh && bash scripts/evaluate_adversarial_60_40.sh
bash scripts/train_adversarial_50_50.sh && bash scripts/evaluate_adversarial_50_50.sh

# Compare all results
python3 scripts/compare_all_models.py

# Create visualizations (optional)
python3 scripts/visualize_results.py
```

## 📁 Project Structure

```
.
├── scripts/
│   ├── train_adversarial_90_10.sh          # Training scripts
│   ├── train_adversarial_80_20.sh
│   ├── train_adversarial_70_30.sh
│   ├── train_adversarial_60_40.sh
│   ├── train_adversarial_50_50.sh
│   ├── evaluate_adversarial_90_10.sh       # Evaluation scripts
│   ├── evaluate_adversarial_80_20.sh
│   ├── evaluate_adversarial_70_30.sh
│   ├── evaluate_adversarial_60_40.sh
│   ├── evaluate_adversarial_50_50.sh
│   ├── prepare_adversarial_training.py     # Data preparation
│   └── compare_all_models.py               # Results comparison
├── evaluation/                              # All evaluation results
│   ├── adversarial_squad/                  # Baseline
│   ├── squad/
│   ├── adversarial_90_10/                  # Experiment results
│   ├── adversarial_80_20/
│   ├── adversarial_70_30/
│   ├── adversarial_60_40/
│   └── adversarial_50_50/
├── data/                                    # Training data
├── trained_model_adversarial_*/            # Trained models
├── check_status.sh                         # Check progress
└── README_EXPERIMENTS.md                   # Full documentation
```

## ⏱️ Time (with A100)
- Each experiment: ~40-60 minutes
- Total: ~3-4 hours

## 📖 Documentation
- `README_EXPERIMENTS.md` - Quick start guide
- `RUN_ALL_5_EXPERIMENTS.md` - Detailed guide for publication
- `EXPERIMENT_SUMMARY.md` - Quick reference
