# 🚀 Start Here - Augmented Experiments

## ✅ What's Ready

You have **complete automation** for running augmented adversarial experiments with 4 scripts:

1. ✅ `scripts/generate_all_augmented_datasets.sh` - Generate datasets
2. ✅ `scripts/train_all_augmented_models.sh` - Train models
3. ✅ `scripts/evaluate_all_augmented_models.sh` - Evaluate models
4. ✅ `scripts/compare_augmented_models.py` - Compare results

---

## 🎯 Run the Complete Pipeline

Execute these 4 commands in order:

```bash
# Step 1: Generate all augmented datasets (5 minutes)
bash scripts/generate_all_augmented_datasets.sh

# Step 2: Train all 5 models (4-5 hours on GPU)
bash scripts/train_all_augmented_models.sh

# Step 3: Evaluate all models (1 hour)
bash scripts/evaluate_all_augmented_models.sh

# Step 4: Compare results (<1 minute)
python3 scripts/compare_augmented_models.py
```

**Total time:** ~5-6 hours (mostly unattended)

---

## 📊 What You'll Get

After running all 4 steps:

- **6 augmented datasets** (90-10 through 50-50)
- **5 trained models** 
- **10 evaluation results** (AddSent + SQuAD for each model)
- **Comparison analysis** showing original vs augmented

**Expected improvements:**
- 80-20: +5-8% EM (72-75%)
- 70-30: +14-19% EM (may fix catastrophic failure!)
- 60-40: +13-18% EM (may fix catastrophic failure!)
- 50-50: +9-15% EM (may fix catastrophic failure!)

---

## 📖 Documentation

- **`QUICK_AUTOMATION_REFERENCE.md`** - Quick commands and tips
- **`AUTOMATION_GUIDE.md`** - Complete guide with examples
- **`AUTOMATION_SUMMARY.md`** - Detailed overview

---

## 💡 Tips

1. **Run overnight:** Training takes 4-5 hours, perfect for overnight run
2. **Check GPU:** Make sure GPU is available (`nvidia-smi`)
3. **Monitor progress:** Scripts show progress and time estimates
4. **Pause/resume:** Scripts check for existing files before overwriting

---

## 🎉 Ready to Start!

```bash
bash scripts/generate_all_augmented_datasets.sh
```

Good luck! 🚀
