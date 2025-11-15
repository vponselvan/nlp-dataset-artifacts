# AddSent Error Analysis Summary

## 📊 Overall Performance
- **Total Examples**: 3,560
- **Correct**: 1,922 (53.99%)
- **Incorrect**: 1,638 (46.01%)
- **Performance Drop from Baseline**: -24.17% EM

---

## 📋 Performance by Question Type

| Question Type | Correct | Total | Accuracy | Error Rate |
|--------------|---------|-------|----------|------------|
| **WHY/HOW**  | 81      | 195   | **41.54%** | **58.46%** ⚠️ Most Vulnerable |
| **OTHER**    | 5       | 13    | 38.46%   | 61.54% |
| **WHERE**    | 90      | 159   | 56.60%   | 43.40% |
| **WHAT**     | 1,176   | 2,167 | 54.27%   | 45.73% |
| **WHO**      | 208     | 385   | 54.03%   | 45.97% |
| **NUMBER**   | 156     | 275   | 56.73%   | 43.27% |
| **WHEN**     | 206     | 366   | **56.28%** | 43.72% ✅ Most Robust |

---

## 📍 Performance by Answer Position

| Position | Correct | Total | Accuracy | Notes |
|----------|---------|-------|----------|-------|
| **First Half** | 1,236 | 2,260 | 54.69% | Slightly better |
| **Second Half** | 686 | 1,300 | 52.77% | Slightly worse |

**Difference**: 1.92% - Minor recency bias detected

---

## 🔍 Key Findings

### 1. **WHY/HOW Questions Most Vulnerable** (41.54% accuracy)
   - **58.46% error rate** - worst performing category
   - These questions require reasoning and inference
   - Distractors with causal language likely confuse the model
   - Example errors show model picks plausible but incorrect causal explanations

### 2. **WHAT Questions Dominate Dataset** (2,167 examples - 61% of dataset)
   - Moderate performance: 54.27% accuracy
   - Largest impact on overall metrics due to volume
   - Critical to improve for overall score

### 3. **Factual Questions More Robust**
   - WHEN: 56.28% (temporal facts)
   - NUMBER: 56.73% (numeric facts)
   - WHERE: 56.60% (location facts)
   - These are ~14% better than WHY/HOW questions

### 4. **Minor Recency Bias** (1.92% difference)
   - First half: 54.69%
   - Second half: 52.77%
   - Model slightly prefers information earlier in context
   - **Less recency bias than expected** - distractors are effective regardless of position

### 5. **Sample Error Patterns**
   - Model predicts **completely wrong entities** (e.g., "Chicago" instead of "Santa Clara")
   - Model picks **plausible but incorrect alternatives** (e.g., "Easter" for a date)
   - Suggests model is being fooled by adversarial distractors

---

## 💡 Implications for Mitigation

### High Priority Targets:
1. **WHY/HOW Questions** - Need adversarial training with causal reasoning examples
2. **WHAT Questions** - Largest volume, moderate performance, high impact potential

### Strategy Recommendations:

#### 1. **Adversarial Fine-tuning**
   - Mix clean SQuAD + AddSent data (80:20 ratio)
   - Upweight WHY/HOW examples in loss function
   - Expected gain: 10-15% on adversarial, maintain ~75% on clean

#### 2. **Dataset Cartography**
   - Identify "hard" WHY/HOW examples during training
   - Reweight these examples for focused learning
   - Target: Bring WHY/HOW from 41% → 55%+

#### 3. **Error Type Specific Training**
   - Focus on examples where model picks wrong entity types
   - Add contrastive examples with correct vs distractor answers
   - Teach model to ignore misleading causal statements

---

## 🎯 Expected Improvements

| Approach | Expected Adversarial | Expected Clean | Trade-off |
|----------|---------------------|----------------|-----------|
| **Baseline** | 53.99% | 78.16% | - |
| **Adversarial Training** | 65-70% | 75-77% | +11-16% adv, -1-3% clean |
| **Dataset Cartography** | 60-65% | 76-78% | +6-11% adv, -0-2% clean |
| **Combined Approach** | 68-72% | 74-76% | Best robustness |

---

## 📈 Next Steps

1. ✅ **Completed**: Baseline evaluation and error analysis
2. 🔄 **Next**: Implement adversarial fine-tuning
   - Create mixed training dataset
   - Train for 2-3 epochs
   - Evaluate on both clean and adversarial test sets
3. 📊 **Then**: Compare mitigation strategies
4. 📝 **Finally**: Write up findings for project report

---

## 🔬 Research Insights

This analysis demonstrates:
- **Reasoning questions** (WHY/HOW) are significantly more vulnerable than factual questions
- **Adversarial examples work** - 24% performance drop
- **Minor recency bias** - model doesn't just pick last sentence
- **Entity confusion** - model struggles with similar entities introduced as distractors

These findings align with prior research on adversarial QA robustness and provide a solid foundation for testing mitigation strategies.
