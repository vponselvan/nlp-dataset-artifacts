#!/bin/bash
# Train model with 50% SQuAD + 50% AddSent

set -e

echo "=========================================="
echo "Adversarial Fine-Tuning (50-50 Split)"
echo "=========================================="
echo ""

# Configuration
SQUAD_PATH="./data/squad.jsonl"
ADDSENT_PATH="./data/addsent_train.jsonl"
MIXED_PATH="./data/mixed_training_50_50.jsonl"
OUTPUT_DIR="./trained_model_adversarial_50_50"

EPOCHS=3
BATCH_SIZE=16
LEARNING_RATE=3e-5

echo "Configuration:"
echo "  Training data: 50% SQuAD + 50% AddSent"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Step 1: Prepare mixed dataset
echo "Step 1: Preparing mixed training dataset (50-50)..."
python3 scripts/prepare_adversarial_training.py \
    --squad_path "$SQUAD_PATH" \
    --addsent_path "$ADDSENT_PATH" \
    --output_path "$MIXED_PATH" \
    --squad_ratio 0.50 \
    --addsent_ratio 0.50

echo ""

# Step 2: Train model
echo "Step 2: Training model..."
python3 run.py \
    --do_train \
    --task qa \
    --dataset "$MIXED_PATH" \
    --model google/electra-small-discriminator \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --save_strategy epoch \
    --logging_steps 100 \
    --save_total_limit 2

echo ""
echo "✅ Training complete!"
echo "Model saved to: $OUTPUT_DIR"
