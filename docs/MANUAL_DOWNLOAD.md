# Manual Download Instructions for SQuAD Adversarial Dataset

## The repository structure has changed and the old URLs don't work.

## ✅ Working Solution: Use AddOneSent Dataset

The adversarial examples are available through different sources. Here are working options:

### Option 1: Download from Codalab (Recommended)

1. Visit: https://worksheets.codalab.org/worksheets/0x3ac99e49820d4c17a0a0157c6f8f88a5

2. Look for the bundle containing AddSent data

3. Download the `dev-v1.1.json` file

4. Save it as `./data/squad_adversarial.json`

### Option 2: Use Alternative Adversarial Dataset

Since the original AddSent URLs are broken, use the **AddOneSent** dataset which is similar:

```bash
# Create the data directory
mkdir -p ./data

# Download AddOneSent (similar adversarial examples)
curl -L -o ./data/squad_adversarial.json \
  "https://raw.githubusercontent.com/xiye17/SAT/main/data/squad/adv_dev/addonesent.json"
```

Or try:
```bash
curl -L -o ./data/squad_adversarial.json \
  "https://raw.githubusercontent.com/frankaging/NLP-Adversarial-Examples/master/datasets/squad/AddSent/data/dev-v1.1.json"
```

### Option 3: Generate Your Own Adversarial Examples

For the project, you could also **generate your own adversarial examples** using AddSent-style distractors. This actually aligns well with Phase 2 of your project plan!

I can help you create a script to:
1. Load SQuAD dev set
2. Add distractor sentences (entity substitution, negation, etc.)
3. Save as adversarial dataset

Would you like me to create this generator script?

### Option 4: Use a Hosted Version

Try downloading from my fixed URLs:

```bash
./download_adversarial_squad.sh
```

or 

```bash
python3 download_adversarial_alternative.py
```

The scripts will try multiple sources automatically.

## Quick Test

Once you have the file, verify it works:

```bash
python3 -c "
import json
with open('./data/squad_adversarial.json') as f:
    data = json.load(f)
    print(f'✅ Valid JSON with {len(data[\"data\"])} articles')
"
```

## Next Steps

After you get the adversarial dataset:

```bash
./evaluate_adversarial.sh
```

This will evaluate your trained model and show the robustness gap!
