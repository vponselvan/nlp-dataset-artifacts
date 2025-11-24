# Checkpoint Functionality Summary

## ✅ All Mitigation Scripts Now Have Checkpoints!

All three mitigation strategy scripts have been updated with checkpoint functionality to allow resuming if interrupted.

---

## 📍 Checkpoint Structure

Each script creates checkpoints in its output directory:

### 1. Negation-Aware Training
```
./trained_model_negation_aware/checkpoints/
├── step1_data_generation.done
├── step2_training.done
└── step3_evaluation.done
```

### 2. Entity-Aware Training
```
./trained_model_entity_aware/checkpoints/
├── step1_data_generation.done
├── step2_training.done
└── step3_evaluation.done
```

### 3. Post-Processing
```
./postprocessing_results/checkpoints/
├── step1_inference.done
├── step2_postprocessing.done
└── step3_evaluation.done
```

---

## 🚀 Usage

### Run Any Script:
```bash
# Negation-Aware
bash scripts/run_negation_aware_training.sh

# Entity-Aware
bash scripts/run_entity_aware_training.sh

# Post-Processing
bash scripts/run_postprocessing.sh
```

### If Interrupted:
Just re-run the same command - it will skip completed steps and continue from where it left off!

---

## 🔄 Re-running Steps

### Re-run Everything:
```bash
# Negation-Aware
rm -rf ./trained_model_negation_aware/checkpoints
bash scripts/run_negation_aware_training.sh

# Entity-Aware
rm -rf ./trained_model_entity_aware/checkpoints
bash scripts/run_entity_aware_training.sh

# Post-Processing
rm -rf ./postprocessing_results/checkpoints
bash scripts/run_postprocessing.sh
```

### Re-run from Step 2:
```bash
# Negation-Aware
rm ./trained_model_negation_aware/checkpoints/step2_training.done
rm ./trained_model_negation_aware/checkpoints/step3_evaluation.done
bash scripts/run_negation_aware_training.sh

# Entity-Aware
rm ./trained_model_entity_aware/checkpoints/step2_training.done
rm ./trained_model_entity_aware/checkpoints/step3_evaluation.done
bash scripts/run_entity_aware_training.sh

# Post-Processing
rm ./postprocessing_results/checkpoints/step2_postprocessing.done
rm ./postprocessing_results/checkpoints/step3_evaluation.done
bash scripts/run_postprocessing.sh
```

### Re-run Only Evaluation (Step 3):
```bash
# Negation-Aware
rm ./trained_model_negation_aware/checkpoints/step3_evaluation.done
bash scripts/run_negation_aware_training.sh

# Entity-Aware
rm ./trained_model_entity_aware/checkpoints/step3_evaluation.done
bash scripts/run_entity_aware_training.sh

# Post-Processing
rm ./postprocessing_results/checkpoints/step3_evaluation.done
bash scripts/run_postprocessing.sh
```

---

## 📋 What Each Step Does

### Negation-Aware Training:
- **Step 1**: Generate negation contrastive pairs (~5 min)
- **Step 2**: Train ELECTRA-base with 3x weighted loss (~2-3 hours)
- **Step 3**: Evaluate on SQuAD and AddSent (~30 min)

### Entity-Aware Training:
- **Step 1**: Generate entity contrastive pairs with NER (~10 min)
- **Step 2**: Train ELECTRA-base with contrastive loss (~2-3 hours)
- **Step 3**: Evaluate on SQuAD and AddSent (~30 min)

### Post-Processing:
- **Step 1**: Run inference on test set (~15 min)
- **Step 2**: Apply NER-based entity expansion (~5 min)
- **Step 3**: Evaluate and compare results (~5 min)

---

## 🎯 Benefits

1. **No Lost Progress**: If training fails, just re-run - completed steps are skipped
2. **Flexible**: Can re-run specific steps by deleting their checkpoints
3. **Time-Saving**: Don't waste hours re-running data generation or completed training
4. **Safe**: Each checkpoint is timestamped for tracking

---

## 📝 Checkpoint File Format

Each checkpoint file contains:
- Timestamp of completion
- Created by `touch` command
- Empty file acts as a flag

Example:
```bash
$ cat ./trained_model_negation_aware/checkpoints/step1_data_generation.done
Mon Nov 25 00:15:32 PST 2024
```

---

## 🔍 Checking Status

To see which steps are completed:
```bash
# Negation-Aware
ls -la ./trained_model_negation_aware/checkpoints/

# Entity-Aware
ls -la ./trained_model_entity_aware/checkpoints/

# Post-Processing
ls -la ./postprocessing_results/checkpoints/
```

---

## ✅ All Scripts Updated

- ✅ `scripts/run_negation_aware_training.sh` - Checkpoints added
- ✅ `scripts/run_entity_aware_training.sh` - Checkpoints added
- ✅ `scripts/run_postprocessing.sh` - Checkpoints added

All scripts also updated to:
- Work from root directory (not scripts/)
- Use `python3` instead of `python`
- Have correct paths for data and models
- Show checkpoint status in summary

---

## 🚀 Ready to Use!

All three mitigation strategies are now checkpoint-enabled and ready to run in Colab. Pull the latest changes and start training!

```bash
cd /content/drive/MyDrive/nlp-dataset-artifacts
git pull origin mitigation
bash scripts/run_negation_aware_training.sh
```
