#!/bin/bash
# Check status of experiments

echo "=========================================="
echo "Experiment Status Check"
echo "=========================================="
echo ""

# Check data files
echo "Data Files:"
[ -f ./data/squad.jsonl ] && echo "  ✅ SQuAD" || echo "  ❌ SQuAD"
[ -f ./data/addsent_eval.jsonl ] && echo "  ✅ AddSent Eval" || echo "  ❌ AddSent"
[ -f ./data/addsent_train.jsonl ] && echo "  ✅ AddSent Train" || echo "  ❌ AddSent"
[ -f ./data/mixed_training_90_10.jsonl ] && echo "  ✅ Mixed 90-10" || echo "  ❌ Mixed 90-10"
[ -f ./data/mixed_training_80_20.jsonl ] && echo "  ✅ Mixed 80-20" || echo "  ❌ Mixed 80-20"
[ -f ./data/mixed_training_70_30.jsonl ] && echo "  ✅ Mixed 70-30" || echo "  ❌ Mixed 70-30"
[ -f ./data/mixed_training_60_40.jsonl ] && echo "  ✅ Mixed 60-40" || echo "  ❌ Mixed 60-40"
[ -f ./data/mixed_training_50_50.jsonl ] && echo "  ✅ Mixed 50-50" || echo "  ❌ Mixed 50-50"
echo ""

# Check models
echo "Trained Models:"
[ -d ./trained_model ] && echo "  ✅ Baseline" || echo "  ❌ Baseline"
[ -d ./trained_model_adversarial_90_10 ] && echo "  ✅ 90-10" || echo "  ❌ 90-10"
[ -d ./trained_model_adversarial_80_20 ] && echo "  ✅ 80-20" || echo "  ❌ 80-20"
[ -d ./trained_model_adversarial_70_30 ] && echo "  ✅ 70-30" || echo "  ❌ 70-30"
[ -d ./trained_model_adversarial_60_40 ] && echo "  ✅ 60-40" || echo "  ❌ 60-40"
[ -d ./trained_model_adversarial_50_50 ] && echo "  ✅ 50-50" || echo "  ❌ 50-50"
echo ""

# Check evaluations
echo "Evaluations:"
[ -f ./evaluation/adversarial_squad/eval_metrics.json ] && echo "  ✅ Baseline" || echo "  ❌ Baseline"
[ -f ./evaluation/adversarial_90_10/eval_metrics.json ] && [ -f ./evaluation/adversarial_90_10/squad/eval_metrics.json ] && echo "  ✅ 90-10" || echo "  ❌ 90-10"
[ -f ./evaluation/adversarial_80_20/eval_metrics.json ] && [ -f ./evaluation/adversarial_80_20/squad/eval_metrics.json ] && echo "  ✅ 80-20" || echo "  ❌ 80-20"
[ -f ./evaluation/adversarial_70_30/eval_metrics.json ] && [ -f ./evaluation/adversarial_70_30/squad/eval_metrics.json ] && echo "  ✅ 70-30" || echo "  ❌ 70-30"
[ -f ./evaluation/adversarial_60_40/eval_metrics.json ] && [ -f ./evaluation/adversarial_60_40/squad/eval_metrics.json ] && echo "  ✅ 60-40" || echo "  ❌ 60-40"
[ -f ./evaluation/adversarial_50_50/eval_metrics.json ] && [ -f ./evaluation/adversarial_50_50/squad/eval_metrics.json ] && echo "  ✅ 50-50" || echo "  ❌ 50-50"
echo ""

# Count completed experiments
completed=0
[ -f ./evaluation/adversarial_90_10/eval_metrics.json ] && ((completed++))
[ -f ./evaluation/adversarial_80_20/eval_metrics.json ] && ((completed++))
[ -f ./evaluation/adversarial_70_30/eval_metrics.json ] && ((completed++))
[ -f ./evaluation/adversarial_60_40/eval_metrics.json ] && ((completed++))
[ -f ./evaluation/adversarial_50_50/eval_metrics.json ] && ((completed++))

echo "=========================================="
echo "Progress: $completed/5 experiments complete"
echo "=========================================="
echo ""

# Show what to run next
if [ $completed -eq 5 ]; then
    echo "✅ All experiments complete!"
    echo ""
    echo "Next: python3 scripts/compare_all_models.py"
else
    echo "Remaining:"
    [ ! -f ./evaluation/adversarial_90_10/eval_metrics.json ] && echo "  • 90-10"
    [ ! -f ./evaluation/adversarial_80_20/eval_metrics.json ] && echo "  • 80-20"
    [ ! -f ./evaluation/adversarial_70_30/eval_metrics.json ] && echo "  • 70-30"
    [ ! -f ./evaluation/adversarial_60_40/eval_metrics.json ] && echo "  • 60-40"
    [ ! -f ./evaluation/adversarial_50_50/eval_metrics.json ] && echo "  • 50-50"
fi

echo ""
