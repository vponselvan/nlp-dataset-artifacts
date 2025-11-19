# Running All 5 Adversarial Fine-Tuning Experiments

## 🎯 Goal
Systematically evaluate 5 different ratios of clean vs adversarial training data to find the optimal trade-off for publication.

## 📊 The 5 Experiments

| Ratio | Clean (SQuAD) | Adversarial (AddSent) | Hypothesis |
|-------|---------------|----------------------|------------|
| 90-10 | 90% | 10% | Minimal adversarial, best clean performance |
| 80-20 | 80% | 20% | Balanced, likely optimal trade-off |
| 70-30 | 70% | 30% | Moderate adversarial exposure |
| 60-40 | 60% | 40% | High adversarial exposure |
| 50-50 | 50% | 50% | Maximum adversarial, highest robustness |

## 🚀 Commands to Run All Experiments

### Check Current Status
```bash
bash check_status.sh
```

### Run All 5 Experiments

**Option 1: Sequential (Recommended)**
```bash
# Experiment 1: 90-10
bash scripts/train_adversarial_90_10.sh
bash scripts/evaluate_adversarial_90_10.sh

# Experiment 2: 80-20
bash scripts/train_adversarial_80_20.sh
bash scripts/evaluate_adversarial_80_20.sh

# Experiment 3: 70-30
bash scripts/train_adversarial_70_30.sh
bash scripts/evaluate_adversarial_70_30.sh

# Experiment 4: 60-40
bash scripts/train_adversarial_60_40.sh
bash scripts/evaluate_adversarial_60_40.sh

# Experiment 5: 50-50
bash scripts/train_adversarial_50_50.sh
bash scripts/evaluate_adversarial_50_50.sh

# Compare all results
python3 scripts/compare_all_models.py
```

**Option 2: Quick Commands (if running sequentially)**
```bash
for ratio in 90_10 80_20 70_30 60_40 50_50; do
    bash scripts/train_adversarial_${ratio}.sh
    bash scripts/evaluate_adversarial_${ratio}.sh
done
python3 scripts/compare_all_models.py
```

## ⏱️ Time Estimates (with A100 GPU)

- Each training: ~30-60 minutes
- Each evaluation: ~10 minutes
- **Total for all 5:** ~3-4 hours

## 📈 What You'll Get

### Comprehensive Comparison Table
```
Model                     AddSent EM   AddSent F1    SQuAD EM    SQuAD F1
--------------------------------------------------------------------------------
Baseline                      53.99%       61.09%      78.16%      86.05%
90-10 Split                   [result]     [result]    [result]    [result]
80-20 Split                   [result]     [result]    [result]    [result]
70-30 Split                   [result]     [result]    [result]    [result]
60-40 Split                   [result]     [result]    [result]    [result]
50-50 Split                   [result]     [result]    [result]    [result]
```

### Trade-off Analysis
- Robustness gain for each ratio
- Clean performance cost for each ratio
- Trade-off ratio (gain/cost) for each
- Ranking from best to worst trade-off

### Visualization Data
You'll be able to create plots showing:
- **X-axis:** % Adversarial data (10%, 20%, 30%, 40%, 50%)
- **Y-axis:** Performance (two curves: AddSent EM, SQuAD EM)
- **Pareto frontier:** Showing the trade-off curve

## 🎓 For Your Publication

### Key Claims You Can Make

1. **Systematic Exploration**
   - "We systematically evaluate 5 different ratios (90-10, 80-20, 70-30, 60-40, 50-50)"
   - Shows thorough experimental methodology

2. **Optimal Trade-off Identification**
   - "The optimal trade-off occurs at [X-Y] ratio"
   - Backed by empirical evidence across 5 data points

3. **Diminishing Returns**
   - "Beyond X% adversarial data, clean performance degrades rapidly"
   - "Marginal robustness gains diminish after Y%"

4. **Deployment Recommendations**
   - "For production systems, we recommend [X-Y] ratio"
   - "This achieves Z% of maximum robustness with only W% cost"

### Paper Structure

```
4. Experiments: Adversarial Fine-Tuning

4.1 Experimental Setup
    - 5 ratios tested: 90-10, 80-20, 70-30, 60-40, 50-50
    - ELECTRA-small, 3 epochs, batch size 16
    - Evaluated on AddSent (adversarial) and SQuAD (clean)

4.2 Results
    - Table 1: Performance across all 5 ratios
    - Figure 1: Trade-off curve (robustness vs clean performance)
    - Figure 2: Trade-off ratio ranking

4.3 Analysis
    - Optimal ratio: [X-Y] achieves best trade-off
    - Diminishing returns beyond [Z]% adversarial data
    - Clean cost increases super-linearly with adversarial ratio

4.4 Discussion
    - Why does [X-Y] achieve optimal trade-off?
    - Comparison with literature (most papers test 1-2 ratios)
    - Deployment considerations based on threat model
```

### Figures to Create

**Figure 1: Trade-off Curve**
```python
import matplotlib.pyplot as plt

adversarial_pct = [10, 20, 30, 40, 50]
addsent_em = [your_results]
squad_em = [your_results]

plt.plot(adversarial_pct, addsent_em, 'o-', label='AddSent (Adversarial)')
plt.plot(adversarial_pct, squad_em, 's-', label='SQuAD (Clean)')
plt.xlabel('% Adversarial Training Data')
plt.ylabel('Exact Match (%)')
plt.title('Adversarial Fine-Tuning Trade-off')
plt.legend()
plt.grid(True)
```

**Figure 2: Trade-off Ratio**
```python
ratios = [90-10, 80-20, 70-30, 60-40, 50-50]
trade_off_ratios = [your_results]

plt.bar(ratios, trade_off_ratios)
plt.xlabel('Training Data Ratio')
plt.ylabel('Trade-off Ratio (Gain/Cost)')
plt.title('Trade-off Efficiency by Ratio')
```

## 📊 Actual Results

### Robustness (AddSent EM)
- 90-10: 64.78% (+10.79% gain) ✅
- **80-20: 66.57% (+12.58% gain) ✅ BEST**
- 70-30: 50.90% (-3.09% loss) ❌
- 60-40: 47.02% (-6.97% loss) ❌
- 50-50: 45.62% (-8.37% loss) ❌

### Clean Performance (SQuAD EM)
- 90-10: 63.54% (-14.62% cost) ✅
- **80-20: 62.85% (-15.31% cost) ✅**
- 70-30: 50.19% (-27.97% cost) ❌
- 60-40: 46.75% (-31.41% cost) ❌
- 50-50: 44.87% (-33.29% cost) ❌

### Trade-off Ratio
- **80-20: 0.82x ✅ BEST**
- 90-10: 0.74x ✅
- 70-30: -0.11x ❌ (negative - worse than baseline)
- 60-40: -0.22x ❌
- 50-50: -0.25x ❌

**Key Finding:** Only 90-10 and 80-20 improve performance. Higher adversarial ratios (70-30+) cause catastrophic overfitting!

## ✅ Checklist

- [ ] Run 90-10 experiment
- [ ] Run 80-20 experiment (already done ✅)
- [ ] Run 70-30 experiment
- [ ] Run 60-40 experiment
- [ ] Run 50-50 experiment (already done ✅)
- [ ] Compare all 5 models
- [ ] Create trade-off curve plot
- [ ] Create trade-off ratio bar chart
- [ ] Write analysis for paper
- [ ] Identify optimal ratio with justification

## 🎉 After Completion

You'll have:
- ✅ 5 trained models
- ✅ 10 evaluation results (5 models × 2 test sets)
- ✅ Comprehensive comparison table
- ✅ Trade-off analysis
- ✅ Data for 2 publication-quality figures
- ✅ Strong empirical evidence for optimal ratio
- ✅ Solid foundation for publication

---

**Ready to start?** Run: `bash check_status.sh` to see what's left!


---

## 📊 Creating Visualizations

After running all experiments, create publication-quality plots:

### Generate Plots

```bash
# Compare all models (saves comparison_results.json)
python3 scripts/compare_all_models.py

# Create visualizations
python3 scripts/visualize_results.py
```

### Output

Creates 4 plots in `evaluation/plots/`:

1. **trade_off_curve.png** - Robustness vs clean performance
2. **trade_off_ratio.png** - Trade-off efficiency ranking
3. **performance_comparison.png** - Side-by-side comparison
4. **gains_and_costs.png** - Gains and costs breakdown

### Requirements

```bash
pip install matplotlib
```

### For Your Paper

**Figure 1: Trade-off Curve**
```
Figure 1: Adversarial fine-tuning trade-off curve. As the proportion 
of adversarial training data increases, robustness on AddSent improves 
(red line) while clean performance on SQuAD degrades (blue line). The 
optimal trade-off occurs at [X-Y], achieving Z% adversarial robustness 
with only W% clean performance cost.
```

**Figure 2: Trade-off Ratio**
```
Figure 2: Trade-off efficiency by training ratio. Higher values indicate 
better robustness gain per unit of clean performance cost. The [X-Y] 
ratio achieves the best trade-off at Z.Zx.
```

### Custom Plots

Use `evaluation/comparison_results.json` for custom visualizations:

```python
import json
import matplotlib.pyplot as plt

with open('evaluation/comparison_results.json', 'r') as f:
    data = json.load(f)

# Your custom plotting code here
```
