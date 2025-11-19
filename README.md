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

**Phase 3: Adversarial Fine-Tuning**
- Trained 5 models on mixed datasets
- Evaluated each model on:
  - `addsent_eval.jsonl` (adversarial robustness)
  - `squad.jsonl` (clean performance)
- Compared trade-offs across all ratios

### Key Findings

**Best Model: 80-20 Ratio**
- Adversarial performance: 66.57% EM (+12.58% over baseline)
- Clean performance: 62.85% EM (-15.31% from baseline)
- Trade-off ratio: 0.82x (best efficiency)

**Performance Cliff Discovery**
- Only 90-10 and 80-20 improve over baseline
- 70-30, 60-40, 50-50 show catastrophic degradation
- More adversarial data is NOT always better

### Quick Start

```bash
# Check experiment status
bash check_status.sh

# Compare all models
python3 scripts/compare_adversarial_models.py

# Generate visualizations
python3 scripts/visualize_results.py

# Analyze pattern improvements
python3 scripts/analyze_pattern_improvements.py --all
```

### Documentation

- `DISCUSSION_OF_FINDINGS.md` - Complete analysis and results
- `RUN_ALL_5_EXPERIMENTS.md` - Detailed experimental procedure
- `QUICK_START.md` - Quick reference guide

### Results Summary

| Model | AddSent EM | SQuAD EM | Trade-off |
|-------|------------|----------|-----------|
| Baseline | 53.99% | 78.16% | - |
| 90-10 | 64.78% | 63.54% | 0.74x |
| **80-20** | **66.57%** | **62.85%** | **0.82x** ✅ |
| 70-30 | 50.90% | 50.19% | -0.11x ❌ |
| 60-40 | 47.02% | 46.75% | -0.22x ❌ |
| 50-50 | 45.62% | 44.87% | -0.25x ❌ |

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
