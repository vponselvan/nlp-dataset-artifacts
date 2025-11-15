#!/bin/bash

# Evaluate trained model on SQuAD Adversarial dataset
# This tests the model's robustness to adversarial examples

MODEL_PATH=${1:-"./trained_model/"}
OUTPUT_DIR=${2:-"./eval_results_adversarial/"}
ADV_DATASET=${3:-"./data/addsent_adversarial.jsonl"}

echo "=========================================="
echo "Evaluating on SQuAD Adversarial Dataset"
echo "=========================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

python3 run.py \
  --do_eval \
  --task qa \
  --dataset "$ADV_DATASET" \
  --model "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_eval_batch_size 32

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Evaluation complete!"
    echo "=========================================="
    echo ""
    echo "View adversarial results:"
    echo "  cat $OUTPUT_DIR/eval_metrics.json"
    echo ""
    echo "Compare with baseline (clean SQuAD):"
    echo "  Baseline:     EM=78.16%, F1=86.05%"
    echo "  Adversarial:  (check above file)"
    echo ""
    echo "Expected performance drop:"
    echo "  EM: ~40-50% (significant drop!)"
    echo "  F1: ~50-60%"
    echo ""
else
    echo ""
    echo "❌ Evaluation failed"
    exit 1
fi
