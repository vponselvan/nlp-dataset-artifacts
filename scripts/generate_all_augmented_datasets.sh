#!/bin/bash
# Automated script to generate all augmented datasets
# This creates augmented AddSent data and all 5 mixed training ratios

set -e

echo "=========================================="
echo "Automated Augmented Dataset Generation"
echo "=========================================="
echo ""

# Configuration
SQUAD_PATH="./data/squad.jsonl"
ADDSENT_TRAIN_PATH="./data/addsent_train.jsonl"
ADDSENT_AUGMENTED_PATH="./data/addsent_train_augmented.jsonl"
AUGMENTATION_RATIO=0.5
SEED=42

# Check if input files exist
if [ ! -f "$SQUAD_PATH" ]; then
    echo "❌ Error: SQuAD file not found: $SQUAD_PATH"
    exit 1
fi

if [ ! -f "$ADDSENT_TRAIN_PATH" ]; then
    echo "❌ Error: AddSent training file not found: $ADDSENT_TRAIN_PATH"
    exit 1
fi

echo "Input files verified ✅"
echo "  SQuAD: $SQUAD_PATH"
echo "  AddSent: $ADDSENT_TRAIN_PATH"
echo ""

# Step 1: Generate augmented AddSent data
echo "=========================================="
echo "Step 1: Augmenting AddSent Training Data"
echo "=========================================="
echo ""

if [ -f "$ADDSENT_AUGMENTED_PATH" ]; then
    echo "⚠️  Augmented AddSent file already exists: $ADDSENT_AUGMENTED_PATH"
    read -p "Do you want to regenerate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Regenerating augmented data..."
        python3 scripts/augment_adversarial_data.py \
            --input_path "$ADDSENT_TRAIN_PATH" \
            --output_path "$ADDSENT_AUGMENTED_PATH" \
            --augmentation_ratio $AUGMENTATION_RATIO \
            --seed $SEED
    else
        echo "Using existing augmented data."
    fi
else
    python3 scripts/augment_adversarial_data.py \
        --input_path "$ADDSENT_TRAIN_PATH" \
        --output_path "$ADDSENT_AUGMENTED_PATH" \
        --augmentation_ratio $AUGMENTATION_RATIO \
        --seed $SEED
fi

echo ""
echo "✅ Augmented AddSent data ready!"
echo ""

# Step 2: Generate all 5 mixed training datasets
echo "=========================================="
echo "Step 2: Generating Mixed Training Datasets"
echo "=========================================="
echo ""

# Define all ratios
declare -a RATIOS=("90:10" "80:20" "70:30" "60:40" "50:50")

for ratio in "${RATIOS[@]}"; do
    # Parse ratio
    squad_ratio=$(echo $ratio | cut -d: -f1)
    addsent_ratio=$(echo $ratio | cut -d: -f2)
    
    # Convert to decimal
    squad_decimal=$(echo "scale=2; $squad_ratio / 100" | bc)
    addsent_decimal=$(echo "scale=2; $addsent_ratio / 100" | bc)
    
    # Output path
    output_path="./data/mixed_training_${squad_ratio}_${addsent_ratio}_augmented.jsonl"
    
    echo "----------------------------------------"
    echo "Creating ${squad_ratio}-${addsent_ratio} augmented dataset"
    echo "----------------------------------------"
    
    python3 scripts/prepare_adversarial_training.py \
        --squad_path "$SQUAD_PATH" \
        --addsent_path "$ADDSENT_AUGMENTED_PATH" \
        --output_path "$output_path" \
        --squad_ratio $squad_decimal \
        --addsent_ratio $addsent_decimal
    
    echo ""
done

# Summary
echo "=========================================="
echo "✅ All Augmented Datasets Generated!"
echo "=========================================="
echo ""

echo "Files created:"
echo "  1. $ADDSENT_AUGMENTED_PATH"
ls -lh ./data/mixed_training_*_augmented.jsonl | awk '{print "  " NR+1 ". " $9 " (" $5 ")"}'

echo ""
echo "Dataset sizes:"
wc -l ./data/mixed_training_*_augmented.jsonl | grep -v total | awk '{print "  " $2 ": " $1 " examples"}'

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "Train all 5 augmented models:"
echo "  bash scripts/train_all_augmented_models.sh"
echo ""
echo "Or train individual models:"
echo "  bash scripts/train_adversarial_90_10_augmented.sh"
echo "  bash scripts/train_adversarial_80_20_augmented.sh"
echo "  bash scripts/train_adversarial_70_30_augmented.sh"
echo "  bash scripts/train_adversarial_60_40_augmented.sh"
echo "  bash scripts/train_adversarial_50_50_augmented.sh"
echo ""
