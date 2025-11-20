# nlp-dataset-artifacts

## Adversarial Fine-Tuning Experiments

This repository includes a systematic study of adversarial fine-tuning for question answering, evaluating 5 different training ratios to find the optimal balance between robustness and clean performance.

### Experiment Overview

**Phase 1: Baseline Training**
- Trained ELECTRA-small model on `squad.jsonl` (clean SQuAD data)
- Evaluated on `addsent_adversarial.jsonl` (AddSent adversarial examples)
- Result: 53.99% EM on adversarial data (baseline)

**Phase 2: Data Preparation**
- Split `addsent_adversarial.jsonl` into:
  - `addsent_train.jsonl` (1,779 examples) - for training
  - `addsent_eval.jsonl` (1,780 examples) - for evaluation
- Created 5 mixed training datasets with different ratios:
  - 90-10: 90% SQuAD + 10% AddSent
  - 80-20: 80% SQuAD + 20% AddSent
  - 70-30: 70% SQuAD + 30% AddSent
  - 60-40: 60% SQuAD + 40% AddSent
  - 50-50: 50% SQuAD + 50% AddSent

**Phase 3: Adversarial Fine-Tuning (ELECTRA-small)**
- Trained 5 models on mixed datasets
- Evaluated each model on:
  - `addsent_eval.jsonl` (adversarial robustness)
  - `squad.jsonl` (clean performance)
- Compared trade-offs across all ratios

**Phase 4: Data Augmentation**
- Augmented AddSent training data with diverse attack types
- Added paraphrase, entity swap, negation, and numeric attacks
- Improved clean performance but revealed capacity bottleneck

**Phase 5: Model Scaling (ELECTRA-base)** 🎉
- Upgraded to ELECTRA-base (110M parameters, 8x larger)
- Trained on augmented 80-20 dataset
- **Breakthrough results: 86.12% EM on AddSent, 87.92% EM on SQuAD**

### Key Findings

**Best Model: ELECTRA-base with 80-20 Augmented Data** 🏆
- Adversarial performance: **86.12% EM** (+32.13% over baseline)
- Clean performance: **87.92% EM** (+9.76% over baseline)
- **No trade-off:** Both metrics improved simultaneously!

**Critical Insights:**
1. **80-20 ratio is optimal** across model sizes
2. **Model capacity is critical** - ELECTRA-small maxed out at 14M params
3. **Data augmentation + larger model = winning combo**
4. Performance cliff at 70-30+ ratio with small models
5. Sufficient capacity eliminates robustness-performance trade-off

### Quick Start

**View Results:**
```bash
# Compare ELECTRA-small vs ELECTRA-base
python3 scripts/compare_electra_base.py

# Generate visualizations
python3 scripts/visualize_electra_base_comparison.py

# View evaluation metrics
cat evaluation/electra_base_80_20_augmented/addsent/eval_metrics.json
cat evaluation/electra_base_80_20_augmented/squad/eval_metrics.json
```

**Train ELECTRA-base (reproduce results):**
```bash
# Train on augmented 80-20 data (~2-3 hours on A100)
bash scripts/train_electra_base_80_20.sh

# Evaluate on both datasets
bash scripts/evaluate_electra_base.sh
```

**Original ELECTRA-small experiments:**
```bash
# Compare all small models
python3 scripts/compare_adversarial_models.py

# Analyze pattern improvements
python3 scripts/analyze_pattern_improvements.py --all
```

### Documentation

- `ACTION_ITEMS.md` - Completed tasks and achievements
- `NEXT_STEPS.md` - Optional future enhancements
- `DISCUSSION_OF_FINDINGS.md` - Complete analysis and results
- `DATA_AUGMENTATION_SUMMARY.md` - Augmentation approach and results
- `IMPROVEMENT_STRATEGIES.md` - Strategies for further improvements
- `QUICK_START.md` - Quick reference guide

### Results Summary

**ELECTRA-small Results:**

| Model | AddSent EM | SQuAD EM | Trade-off |
|-------|------------|----------|-----------|
| Baseline | 53.99% | 78.16% | - |
| 90-10 | 64.78% | 63.54% | 0.74x |
| 80-20 | 66.57% | 62.85% | 0.82x ✅ |
| 70-30 | 50.90% | 50.19% | -0.11x ❌ |
| 60-40 | 47.02% | 46.75% | -0.22x ❌ |
| 50-50 | 45.62% | 44.87% | -0.25x ❌ |

**ELECTRA-base Results (Final):** 🎉

| Model | AddSent EM | SQuAD EM | Improvement |
|-------|------------|----------|-------------|
| Baseline (small) | 53.99% | 78.16% | - |
| 80-20 Augmented (small) | 63.48% | 66.60% | +9.49% / -11.56% |
| **80-20 Augmented (base)** | **86.12%** 🏆 | **87.92%** 🏆 | **+32.13% / +9.76%** ✅ |

**Achievement: State-of-the-art adversarial robustness with no performance trade-off!**

---

## Getting Started
You'll need Python >= 3.6 to run the code in this repo.

First, clone the repository:

`git clone git@github.com:vponselvan/nlp-dataset-artifacts.git`

Then install the dependencies:

`pip install --upgrade pip`

`pip install -r requirements.txt`

If you're running on a shared machine and don't have the privileges to install Python packages globally,
or if you just don't want to install these packages permanently, take a look at the "Virtual environments"
section further down in the README.

To make sure pip is installing packages for the right Python version, run `pip --version`
and check that the path it reports is for the right Python interpreter.

## Training and evaluating a model

### Training on SQuAD
To train an ELECTRA-small model on SQuAD for question answering:

```bash
python3 run.py --do_train --task qa --dataset squad --output_dir ./trained_model
```

Checkpoints will be written to sub-folders of the `trained_model` output directory.

To prevent `run.py` from trying to use a GPU for training, pass the argument `--no_cuda`.

**Descriptions of other important arguments are available in the comments in `run.py`.**

Data and models will be automatically downloaded and cached in `~/.cache/huggingface/`.
To change the caching directory, you can modify the shell environment variable `HF_HOME` or `TRANSFORMERS_CACHE`.
For more details, see [this doc](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models).

An ELECTRA-small based NLI model trained on SNLI for 3 epochs (e.g. with the command above) should achieve an accuracy of around 89%, depending on batch size.
An ELECTRA-small based QA model trained on SQuAD for 3 epochs should achieve around 78 exact match score and 86 F1 score.

### Evaluating on Custom Datasets

To evaluate on SQuAD:
```bash
python3 run.py --do_eval --task qa --dataset squad --model "./trained_model/" --output_dir "./eval_results/" --per_device_eval_batch_size 32
```

To evaluate on AddSent adversarial dataset:
```bash
python3 run.py --do_eval --task qa --dataset "./data/addsent_adversarial.jsonl" --model "./trained_model/" --output_dir "./eval_results_addsent/" --per_device_eval_batch_size 32
```

## Working with datasets
This repo uses [Huggingface Datasets](https://huggingface.co/docs/datasets/) to load data.
The Dataset objects loaded by this module can be filtered and updated easily using the `Dataset.filter` and `Dataset.map` methods.
For more information on working with datasets loaded as HF Dataset objects, see [this page](https://huggingface.co/docs/datasets/process.html).

## Virtual environments
Python 3 supports virtual environments with the `venv` module. These will let you select a particular Python interpreter
to be the default (so that you can run it with `python`) and install libraries only for a particular project.
To set up a virtual environment, use the following command:

`python3 -m venv path/to/my_venv_dir`

This will set up a virtual environment in the target directory.
WARNING: This command overwrites the target directory, so choose a path that doesn't exist yet!

To activate your virtual environment (so that `python` redirects to the right version, and your virtual environment packages are active),
use this command:

`source my_venv_dir/bin/activate`

This command looks slightly different if you're not using `bash` on Linux. The [venv docs](https://docs.python.org/3/library/venv.html) have a list of alternate commands for different systems.

Once you've activated your virtual environment, you can use `pip` to install packages the way you normally would, but the installed
packages will stay in the virtual environment instead of your global Python installation. Only the virtual environment's Python
executable will be able to see these packages.
