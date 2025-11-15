#!/bin/bash

# Compare model performance on clean vs adversarial SQuAD
# This script evaluates the same model on both datasets

MODEL_PATH=${1:-"./trained_model/"}

echo "=========================================="
echo "Baseline vs Adversarial Comparison"
echo "=========================================="
echo ""
echo "Model: $MODEL_PATH"
echo ""

# Step 1: Evaluate on clean SQuAD
echo "Step 1/2: Evaluating on clean SQuAD dev set..."
echo ""

python3 run.py \
  --do_eval \
  --task qa \
  --dataset squad \
  --model "$MODEL_PATH" \
  --output_dir ./eval_results_baseline \
  --per_device_eval_batch_size 32

if [ $? -ne 0 ]; then
    echo "❌ Baseline evaluation failed"
    exit 1
fi

# Step 2: Evaluate on adversarial SQuAD
echo ""
echo "Step 2/2: Evaluating on adversarial SQuAD..."
echo ""

python3 run.py \
  --do_eval \
  --task qa \
  --dataset stanfordnlp/squad_adversarial \
  --model "$MODEL_PATH" \
  --output_dir ./eval_results_adversarial \
  --per_device_eval_batch_size 32

if [ $? -ne 0 ]; then
    echo "❌ Adversarial evaluation failed"
    exit 1
fi

# Display comparison
echo ""
echo "=========================================="
echo "📊 RESULTS COMPARISON"
echo "=========================================="
echo ""

echo "Baseline (Clean SQuAD):"
cat ./eval_results_baseline/eval_metrics.json | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"  EM: {data['eval_exact_match']:.2f}%\"); print(f\"  F1: {data['eval_f1']:.2f}%\")"

echo ""
echo "Adversarial SQuAD:"
cat ./eval_results_adversarial/eval_metrics.json | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"  EM: {data['eval_exact_match']:.2f}%\"); print(f\"  F1: {data['eval_f1']:.2f}%\")"

echo ""
echo "Performance Drop:"
python3 -c "
import json
with open('./eval_results_baseline/eval_metrics.json') as f:
    baseline = json.load(f)
with open('./eval_results_adversarial/eval_metrics.json') as f:
    adversarial = json.load(f)

em_drop = baseline['eval_exact_match'] - adversarial['eval_exact_match']
f1_drop = baseline['eval_f1'] - adversarial['eval_f1']

print(f\"  EM: -{em_drop:.2f}% (from {baseline['eval_exact_match']:.2f}% to {adversarial['eval_exact_match']:.2f}%)\")
print(f\"  F1: -{f1_drop:.2f}% (from {baseline['eval_f1']:.2f}% to {adversarial['eval_f1']:.2f}%)\")
"

echo ""
echo "=========================================="
echo "✅ Comparison complete!"
echo "=========================================="
echo ""
echo "Files saved:"
echo "  ./eval_results_baseline/eval_metrics.json"
echo "  ./eval_results_adversarial/eval_metrics.json"
echo ""
