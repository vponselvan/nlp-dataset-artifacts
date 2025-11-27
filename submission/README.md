# nlp-dataset-artifacts

## Getting Started
You'll need Python >= 3.6 to run the code

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
