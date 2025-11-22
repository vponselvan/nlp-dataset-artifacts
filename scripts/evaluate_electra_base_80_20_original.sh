#!/bin/bash
# Evaluate ELECTRA-base 80-20 Original (non-augmented) model

echo "=========================================="
echo "Evaluating ELECTRA-base 80-20 Original"
echo "=========================================="
echo ""

# Evaluate on AddSent (adversarial)
echo "1. Evaluating on AddSent (adversarial)..."
python3 run.py \
  --do_eval \
  --task qa \
  --dataset ./data/addsent_eval.jsonl \
  --model ./trained_model_electra_base_80_20/ \
  --output_dir ./evaluation/electra_base_80_20/addsent/ \
  --per_device_eval_batch_size 32

echo ""
echo "2. Evaluating on SQuAD (clean)..."
python3 run.py \
  --do_eval \
  --task qa \
  --dataset squad \
  --model ./trained_model_electra_base_80_20/ \
  --output_dir ./evaluation/electra_base_80_20/squad/ \
  --per_device_eval_batch_size 32

echo ""
echo "=========================================="
echo "✅ Evaluation complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - evaluation/electra_base_80_20/addsent/eval_metrics.json"
echo "  - evaluation/electra_base_80_20/squad/eval_metrics.json"
echo ""
