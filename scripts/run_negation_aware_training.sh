#!/bin/bash
#
# Negation-Aware Contrastive Training Pipeline
#
# This script implements the complete 3-step Negation-Aware training strategy:
# Step 1: Generate contrastive negation pairs
# Step 2: Train with weighted loss (3x for negation examples)
# Step 3: Evaluate on clean and adversarial data
#
# Goal: Reduce "Negation Confusion" from 40.4% of errors
# Expected: +10-15% improvement on negation-specific examples

set -e  # Exit on error

echo "========================================================================"
echo "Negation-Aware Contrastive Training Pipeline"
echo "========================================================================"
echo ""

# Configuration
INPUT_DATA="../data/mixed_training_80_20.jsonl"
OUTPUT_DATA="../data/mixed_training_80_20_negation_aware.jsonl"
MODEL_NAME="google/electra-base-discriminator"
OUTPUT_DIR="../trained_model_negation_aware"
NEGATION_WEIGHT=3.0
AUGMENTATION_RATIO=0.3
BATCH_SIZE=16
LEARNING_RATE=3e-5
NUM_EPOCHS=3

echo "Configuration:"
echo "  Input: ${INPUT_DATA}"
echo "  Output data: ${OUTPUT_DATA}"
echo "  Model: ${MODEL_NAME}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Negation weight: ${NEGATION_WEIGHT}x"
echo "  Augmentation ratio: ${AUGMENTATION_RATIO}"
echo ""

# Step 1: Generate Negation-Aware Contrastive Pairs
echo "========================================================================"
echo "STEP 1: Generating Negation-Aware Contrastive Pairs"
echo "========================================================================"
echo ""

python generate_negation_contrastive_pairs.py \
    --input "${INPUT_DATA}" \
    --output "${OUTPUT_DATA}" \
    --negation-weight ${NEGATION_WEIGHT} \
    --augmentation-ratio ${AUGMENTATION_RATIO} \
    --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate contrastive pairs"
    exit 1
fi

echo ""
echo "✓ Step 1 complete: Contrastive pairs generated"
echo ""

# Step 2: Train with Negation-Aware Weighted Loss
echo "========================================================================"
echo "STEP 2: Training with Negation-Aware Weighted Loss"
echo "========================================================================"
echo ""

python train_negation_aware.py \
    --train-data "${OUTPUT_DATA}" \
    --eval-squad "../data/squad.jsonl" \
    --eval-addsent "../data/addsent_eval.jsonl" \
    --model "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --num-epochs ${NUM_EPOCHS} \
    --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "✓ Step 2 complete: Model trained with negation awareness"
echo ""

# Step 3: Detailed Evaluation
echo "========================================================================"
echo "STEP 3: Detailed Evaluation"
echo "========================================================================"
echo ""

echo "Evaluating on SQuAD (clean)..."
cd ..
python run.py \
    --model "${OUTPUT_DIR}" \
    --task qa \
    --dataset data/squad.jsonl \
    --do_eval \
    --output_dir "${OUTPUT_DIR}/eval_squad" \
    --per_device_eval_batch_size 32

echo ""
echo "Evaluating on AddSent (adversarial)..."
python run.py \
    --model "${OUTPUT_DIR}" \
    --task qa \
    --dataset data/addsent_eval.jsonl \
    --do_eval \
    --output_dir "${OUTPUT_DIR}/eval_addsent" \
    --per_device_eval_batch_size 32

echo ""
echo "✓ Step 3 complete: Evaluation finished"
echo ""

# Print summary
echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Model saved to: ${OUTPUT_DIR}"
echo "Training data: ${OUTPUT_DATA}"
echo ""
echo "Next steps:"
echo "  1. Check results: ${OUTPUT_DIR}/negation_aware_results.json"
echo "  2. Compare with baseline (ELECTRA-base 80-20)"
echo "  3. Run error analysis to verify negation improvement"
echo "  4. Update report with findings"
echo ""
echo "Expected improvements:"
echo "  - Baseline AddSent EM: 68.90%"
echo "  - Target AddSent EM: 75-80% (+10-15% on negation errors)"
echo "  - Maintain SQuAD EM: ~85-90%"
echo ""
echo "========================================================================"
