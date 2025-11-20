#!/bin/bash
# Evaluate ELECTRA-base model on both AddSent and SQuAD

set -e

MODEL_DIR="./trained_model_electra_base_80_20_augmented"
ADDSENT_PATH="./data/addsent_eval.jsonl"
SQUAD_PATH="./data/squad.jsonl"
OUTPUT_BASE="./evaluation/electra_base_80_20_augmented"

echo "=========================================="
echo "Evaluating ELECTRA-base (80-20 Augmented)"
echo "=========================================="

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model not found at $MODEL_DIR"
    echo "Please run train_electra_base_80_20.sh first"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

# Evaluate on AddSent
echo ""
echo "Evaluating on AddSent (adversarial)..."
python3 run.py \
  --do_eval \
  --task qa \
  --dataset "$ADDSENT_PATH" \
  --model "$MODEL_DIR" \
  --output_dir "${OUTPUT_BASE}/addsent" \
  --per_device_eval_batch_size 32 \
  --max_length 384

# Evaluate on SQuAD
echo ""
echo "Evaluating on SQuAD (clean)..."
python3 run.py \
  --do_eval \
  --task qa \
  --dataset squad \
  --model "$MODEL_DIR" \
  --output_dir "${OUTPUT_BASE}/squad" \
  --per_device_eval_batch_size 32 \
  --max_length 384

echo ""
echo "✅ Evaluation completed!"
echo ""
echo "Results saved to:"
echo "  AddSent: ${OUTPUT_BASE}/addsent/"
echo "  SQuAD: ${OUTPUT_BASE}/squad/"
echo ""
echo "View results:"
echo "  cat ${OUTPUT_BASE}/addsent/eval_metrics.json"
echo "  cat ${OUTPUT_BASE}/squad/eval_metrics.json"
