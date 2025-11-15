# Experiment Tracking Log

## Project: Adversarial QA Robustness (SQuAD)

### Experiment 1: Baseline Fine-tuning
**Date:** 
**Goal:** Establish baseline performance of ELECTRA-small on clean SQuAD v1.1 data

**Configuration:**
- Model: `google/electra-small-discriminator`
- Dataset: SQuAD v1.1 (train + dev)
- Epochs: 3
- Batch size: 16 (train), 32 (eval)
- Learning rate: 3e-5
- Optimizer: AdamW with warmup (0.1 ratio)
- Seed: 42

**Expected Results:**
- Exact Match (EM): ~78%
- F1 Score: ~86%

**Actual Results:**
- EM: 
- F1: 
- Training time: 

**Notes:**


---

## Next Steps After Baseline

### Phase 1: Baseline Evaluation
- [ ] Fine-tune ELECTRA-small on SQuAD v1.1
- [ ] Record EM and F1 scores on clean dev set
- [ ] Save model checkpoint for later use

### Phase 2: Adversarial Dataset Creation
- [ ] Implement AddSent-style distractor generation
- [ ] Create adversarial version of SQuAD dev set
- [ ] Categories to implement:
  - Entity substitution
  - Negation
  - Semantic confusion
  - Irrelevant facts

### Phase 3: Adversarial Evaluation
- [ ] Evaluate baseline model on adversarial dev set
- [ ] Analyze performance drop
- [ ] Categorize failure modes

### Phase 4: Mitigation Strategies

#### Strategy 1: Adversarial Fine-tuning
- [ ] Generate adversarial training examples
- [ ] Mix clean + adversarial examples (various ratios)
- [ ] Fine-tune model
- [ ] Evaluate on both clean and adversarial dev sets

#### Strategy 2: Dataset Cartography Reweighting
- [ ] Compute training dynamics (confidence, variability, correctness)
- [ ] Identify "hard" examples using cartography
- [ ] Implement example reweighting in training loop
- [ ] Fine-tune with reweighted loss
- [ ] Evaluate robustness

### Phase 5: Analysis & Visualization
- [ ] Compare EM/F1 across all conditions
- [ ] Visualize performance by distractor type
- [ ] Create cartography plots
- [ ] Document findings

---

## File Organization

```
experiments/
├── baseline_squad/          # Baseline model checkpoint + eval results
├── adversarial_dataset/     # Generated adversarial examples
├── adv_finetuned/          # Adversarially fine-tuned model
├── cartography_reweighted/ # Dataset cartography model
└── results/                # Comparison tables, plots, analysis
```
