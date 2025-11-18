#!/bin/bash
# Evaluate 70-30 model

set -e

echo "=========================================="
echo "Evaluating Model (70-30 Split)"
echo "=========================================="
echo ""

MODEL_DIR="./trained_model_adversarial_70_30"
ADDSENT_PATH="./data/addsent_adversarial.jsonl"

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model not found at $MODEL_DIR"
    echo "Please run train_adversarial_70_30.sh first"
    exit 1
fi

echo "Model: $MODEL_DIR"
echo ""

EVAL_DIR="./evaluation/adversarial_70_30"

# Evaluate on AddSent (adversarial)
echo "Evaluating on AddSent (adversarial)..."
python3 run.py \
    --do_eval \
    --task qa \
    --dataset "$ADDSENT_PATH" \
    --model "$MODEL_DIR" \
    --output_dir "$EVAL_DIR" \
    --per_device_eval_batch_size 32

echo ""

# Evaluate on clean SQuAD
echo "Evaluating on SQuAD (clean)..."
python3 run.py \
    --do_eval \
    --task qa \
    --dataset squad \
    --model "$MODEL_DIR" \
    --output_dir "$EVAL_DIR/squad" \
    --per_device_eval_batch_size 32

echo ""
echo "✅ Evaluation complete!"
echo ""
echo "Results saved to:"
echo "  $EVAL_DIR/eval_metrics.json (AddSent)"
echo "  $EVAL_DIR/squad/eval_metrics.json (SQuAD)"
