#!/bin/bash
# Train ELECTRA-base with 80-20 augmented data
# Expected: +8-12% EM over ELECTRA-small

set -e

echo "=========================================="
echo "Training ELECTRA-base with 80-20 Augmented Data"
echo "=========================================="

# Configuration
MODEL_NAME="google/electra-base-discriminator"  # Upgraded from small
MIXED_DATA="./data/mixed_training_80_20_augmented.jsonl"
OUTPUT_DIR="./trained_model_electra_base_80_20_augmented"
BATCH_SIZE=8  # Reduced due to larger model
GRADIENT_ACCUMULATION=2  # Effective batch size = 16
LEARNING_RATE=2e-5  # Slightly lower for larger model
NUM_EPOCHS=3
MAX_SEQ_LENGTH=384
DOC_STRIDE=128

echo ""
echo "Model: ELECTRA-base (110M parameters)"
echo "Training data: 80% SQuAD + 20% AddSent (augmented)"
echo "Expected improvement: +8-12% EM"
echo ""

# Check if data exists
if [ ! -f "$MIXED_DATA" ]; then
    echo "❌ Error: Mixed training data not found at $MIXED_DATA"
    echo "Please run generate_all_augmented_datasets.sh first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Step 1: Fine-tuning ELECTRA-base..."
echo "Note: This will take longer than ELECTRA-small (~6-8 hours on GPU)"
echo ""

python3 run.py \
  --do_train \
  --task qa \
  --model "$MODEL_NAME" \
  --dataset "$MIXED_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --max_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --save_strategy epoch \
  --save_total_limit 1 \
  --logging_steps 100 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --fp16

echo ""
echo "✅ Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Evaluate the model"
echo "  bash scripts/evaluate_electra_base.sh"
