#!/bin/bash
# Curriculum Learning: Gradually increase adversarial ratio during training
# This addresses the overfitting problem observed at 70-30+ ratios

set -e

echo "=========================================="
echo "Adversarial Curriculum Learning Training"
echo "=========================================="

# Configuration
MODEL_NAME="google/electra-base-discriminator"  # Upgrade to base model
SQUAD_PATH="./data/squad_train.jsonl"
ADDSENT_PATH="./data/addsent_train.jsonl"
OUTPUT_BASE="./models/curriculum_learning"
BATCH_SIZE=16
LEARNING_RATE=3e-5
MAX_SEQ_LENGTH=384
DOC_STRIDE=128

# Curriculum schedule (ratio changes per epoch)
declare -a CURRICULUM=(
    "95:5"   # Epoch 0: Warm-up with mostly clean data
    "90:10"  # Epoch 1: Gradual introduction
    "85:15"  # Epoch 2: Increasing adversarial
    "82:18"  # Epoch 3: Approaching target
    "80:20"  # Epoch 4-7: Final target ratio
    "80:20"
    "80:20"
    "80:20"
)

echo ""
echo "Curriculum Schedule:"
echo "  Epoch 0: 95% SQuAD + 5% AddSent (warm-up)"
echo "  Epoch 1: 90% SQuAD + 10% AddSent"
echo "  Epoch 2: 85% SQuAD + 15% AddSent"
echo "  Epoch 3: 82% SQuAD + 18% AddSent"
echo "  Epoch 4-7: 80% SQuAD + 20% AddSent (target)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"
mkdir -p ./data/curriculum

# Training loop - one epoch at a time with different ratios
for epoch in {0..7}; do
    ratio="${CURRICULUM[$epoch]}"
    squad_ratio=$(echo $ratio | cut -d: -f1)
    addsent_ratio=$(echo $ratio | cut -d: -f2)
    
    echo "=========================================="
    echo "Epoch $epoch: Training with $squad_ratio% SQuAD + $addsent_ratio% AddSent"
    echo "=========================================="
    
    # Prepare mixed dataset for this epoch
    echo ""
    echo "Step 1: Preparing mixed dataset..."
    python scripts/prepare_adversarial_training.py \
        --squad_path "$SQUAD_PATH" \
        --addsent_path "$ADDSENT_PATH" \
        --output_path "./data/curriculum/mixed_${squad_ratio}_${addsent_ratio}.jsonl" \
        --squad_ratio $(echo "scale=2; $squad_ratio / 100" | bc) \
        --addsent_ratio $(echo "scale=2; $addsent_ratio / 100" | bc) \
        --seed $((42 + epoch))  # Different seed per epoch
    
    echo ""
    echo "Step 2: Fine-tuning model (Epoch $epoch)..."
    
    # Determine if this is first epoch or continuation
    if [ $epoch -eq 0 ]; then
        # First epoch: start from pretrained
        INPUT_MODEL="$MODEL_NAME"
        EPOCHS_THIS_STEP=1
    else
        # Subsequent epochs: continue from previous checkpoint
        INPUT_MODEL="$OUTPUT_BASE/epoch_$((epoch-1))"
        EPOCHS_THIS_STEP=1
    fi
    
    python train_qa.py \
        --model_name_or_path "$INPUT_MODEL" \
        --train_file "./data/curriculum/mixed_${squad_ratio}_${addsent_ratio}.jsonl" \
        --output_dir "$OUTPUT_BASE/epoch_$epoch" \
        --max_seq_length $MAX_SEQ_LENGTH \
        --doc_stride $DOC_STRIDE \
        --per_device_train_batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $EPOCHS_THIS_STEP \
        --save_strategy "no" \
        --logging_steps 100 \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --fp16
    
    echo ""
    echo "✅ Epoch $epoch completed!"
    echo "   Model saved to: $OUTPUT_BASE/epoch_$epoch"
    echo ""
    
    # Optional: Evaluate at each epoch to track progress
    if [ $((epoch % 2)) -eq 1 ]; then  # Evaluate every 2 epochs
        echo "Evaluating progress at epoch $epoch..."
        python evaluate_qa.py \
            --model_path "$OUTPUT_BASE/epoch_$epoch" \
            --test_file "./data/addsent_eval.jsonl" \
            --output_file "$OUTPUT_BASE/epoch_${epoch}_addsent_predictions.json" \
            --max_seq_length $MAX_SEQ_LENGTH \
            --doc_stride $DOC_STRIDE \
            --per_device_eval_batch_size 32
        
        python scripts/evaluate_predictions.py \
            --predictions "$OUTPUT_BASE/epoch_${epoch}_addsent_predictions.json" \
            --dataset "./data/addsent_eval.jsonl"
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "✅ Curriculum Learning Training Complete!"
echo "=========================================="
echo ""
echo "Final model location: $OUTPUT_BASE/epoch_7"
echo ""
echo "Next steps:"
echo "1. Evaluate final model:"
echo "   ./evaluate_adversarial_curriculum.sh"
echo ""
echo "2. Compare with standard 80-20 training:"
echo "   python compare_models.py \\"
echo "     --model1 models/adversarial_80_20 \\"
echo "     --model2 models/curriculum_learning/epoch_7"
