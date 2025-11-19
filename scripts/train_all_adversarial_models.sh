#!/bin/bash
# Train all 5 original (non-augmented) models sequentially
# This will take approximately 4-5 hours on GPU

set -e

echo "=========================================="
echo "Training All 5 Original Models"
echo "=========================================="
echo ""

# Configuration
MODEL_NAME="google/electra-small-discriminator"
EPOCHS=3
BATCH_SIZE=16
LEARNING_RATE=3e-5

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Check if datasets exist
echo "Checking for datasets..."
declare -a RATIOS=("90_10" "80_20" "70_30" "60_40" "50_50")
all_exist=true

for ratio in "${RATIOS[@]}"; do
    dataset_path="./data/mixed_training_${ratio}.jsonl"
    if [ ! -f "$dataset_path" ]; then
        echo "❌ Missing: $dataset_path"
        all_exist=false
    else
        echo "✅ Found: $dataset_path"
    fi
done

echo ""

if [ "$all_exist" = false ]; then
    echo "❌ Error: Some datasets are missing!"
    echo ""
    echo "Please create them first using prepare_adversarial_training.py"
    echo ""
    exit 1
fi

echo "All datasets found! ✅"
echo ""

# Training function
train_model() {
    local ratio=$1
    local dataset_path="./data/mixed_training_${ratio}.jsonl"
    local output_dir="./trained_model_adversarial_${ratio}"
    
    echo "=========================================="
    echo "Training ${ratio} Model"
    echo "=========================================="
    echo ""
    echo "Dataset: $dataset_path"
    echo "Output: $output_dir"
    echo ""
    
    # Check if model already exists
    if [ -d "$output_dir" ]; then
        echo "⚠️  Model directory already exists: $output_dir"
        read -p "Do you want to retrain? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping ${ratio}..."
            echo ""
            return
        fi
        echo "Retraining ${ratio}..."
    fi
    
    # Train
    python3 run.py \
        --do_train \
        --task qa \
        --dataset "$dataset_path" \
        --model "$MODEL_NAME" \
        --output_dir "$output_dir" \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --save_strategy epoch \
        --logging_steps 100 \
        --save_total_limit 2
    
    echo ""
    echo "✅ ${ratio} model training complete!"
    echo "   Saved to: $output_dir"
    echo ""
}

# Train all models
start_time=$(date +%s)

for ratio in "${RATIOS[@]}"; do
    train_model "$ratio"
done

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))

echo "=========================================="
echo "✅ All Models Trained Successfully!"
echo "=========================================="
echo ""
echo "Total training time: ${hours}h ${minutes}m"
echo ""

echo "Models created:"
ls -d ./trained_model_adversarial_* 2>/dev/null | grep -v augmented | awk '{print "  " NR ". " $1}'

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "Evaluate all models:"
echo "  bash scripts/evaluate_all_adversarial_models.sh"
echo ""
echo "Compare results:"
echo "  python3 scripts/compare_adversarial_models.py"
echo ""
