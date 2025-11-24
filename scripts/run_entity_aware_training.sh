#!/bin/bash
#
# Entity-Aware Contrastive Training Pipeline
#
# This script implements the complete 3-step Entity-Aware training strategy:
# Step 1: Extract entities and identify hard negatives (NER)
# Step 2: Train with contrastive ranking loss
# Step 3: Evaluate on clean and adversarial data
#
# Goal: Reduce "Entity Substitution" from 29.9% of errors
# Expected: +8-12% improvement on entity-specific examples
#
# Checkpoints: This script creates checkpoint files to allow resuming

set -e  # Exit on error

echo "========================================================================"
echo "Entity-Aware Contrastive Training Pipeline"
echo "========================================================================"
echo ""

# Configuration
INPUT_DATA="./data/mixed_training_80_20.jsonl"
OUTPUT_DATA="./data/mixed_training_80_20_entity_aware.jsonl"
MODEL_NAME="google/electra-base-discriminator"
OUTPUT_DIR="./trained_model_entity_aware"
ENTITY_WEIGHT=2.5
AUGMENTATION_RATIO=0.2
CONTRASTIVE_WEIGHT=0.5
BATCH_SIZE=16
LEARNING_RATE=3e-5
NUM_EPOCHS=3

# Checkpoint files
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_STEP1="${CHECKPOINT_DIR}/step1_data_generation.done"
CHECKPOINT_STEP2="${CHECKPOINT_DIR}/step2_training.done"
CHECKPOINT_STEP3="${CHECKPOINT_DIR}/step3_evaluation.done"

echo "Configuration:"
echo "  Input: ${INPUT_DATA}"
echo "  Output data: ${OUTPUT_DATA}"
echo "  Model: ${MODEL_NAME}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Entity weight: ${ENTITY_WEIGHT}x"
echo "  Augmentation ratio: ${AUGMENTATION_RATIO}"
echo "  Contrastive weight: ${CONTRASTIVE_WEIGHT}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo ""

# Check for existing checkpoints
if [ -f "${CHECKPOINT_STEP3}" ]; then
    echo "✓ All steps already completed!"
    echo "  To re-run, delete: ${CHECKPOINT_DIR}"
    exit 0
fi

# Check spaCy installation
echo "Checking dependencies..."
python -c "import spacy; print('✓ spaCy installed')" 2>/dev/null || {
    echo "ERROR: spaCy not installed"
    echo "Install with: pip install spacy"
    echo "Download model: python -m spacy download en_core_web_sm"
    exit 1
}

python -c "import spacy; spacy.load('en_core_web_sm'); print('✓ en_core_web_sm model available')" 2>/dev/null || {
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
}

echo ""

# Step 1: Generate Entity-Aware Contrastive Pairs
if [ -f "${CHECKPOINT_STEP1}" ]; then
    echo "========================================================================"
    echo "STEP 1: SKIPPED (already completed)"
    echo "========================================================================"
    echo "  Checkpoint found: ${CHECKPOINT_STEP1}"
    echo "  Data file: ${OUTPUT_DATA}"
    echo ""
else
    echo "========================================================================"
    echo "STEP 1: Generating Entity-Aware Contrastive Pairs"
    echo "========================================================================"
    echo ""

    python3 scripts/generate_entity_contrastive_pairs.py \
        --input "${INPUT_DATA}" \
        --output "${OUTPUT_DATA}" \
        --entity-weight ${ENTITY_WEIGHT} \
        --augmentation-ratio ${AUGMENTATION_RATIO} \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate entity pairs"
        exit 1
    fi

    # Create checkpoint
    touch "${CHECKPOINT_STEP1}"
    echo "$(date)" > "${CHECKPOINT_STEP1}"

    echo ""
    echo "✓ Step 1 complete: Entity pairs with hard negatives generated"
    echo "  Checkpoint saved: ${CHECKPOINT_STEP1}"
    echo ""
fi

# Step 2: Train with Entity-Aware Contrastive Loss
if [ -f "${CHECKPOINT_STEP2}" ]; then
    echo "========================================================================"
    echo "STEP 2: SKIPPED (already completed)"
    echo "========================================================================"
    echo "  Checkpoint found: ${CHECKPOINT_STEP2}"
    echo "  Model dir: ${OUTPUT_DIR}"
    echo ""
else
    echo "========================================================================"
    echo "STEP 2: Training with Entity-Aware Contrastive Loss"
    echo "========================================================================"
    echo ""

    python3 scripts/train_entity_aware.py \
        --train-data "${OUTPUT_DATA}" \
        --eval-squad "./data/squad.jsonl" \
        --eval-addsent "./data/addsent_eval.jsonl" \
        --model "${MODEL_NAME}" \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size ${BATCH_SIZE} \
        --learning-rate ${LEARNING_RATE} \
        --num-epochs ${NUM_EPOCHS} \
        --contrastive-weight ${CONTRASTIVE_WEIGHT} \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed"
        exit 1
    fi

    # Create checkpoint
    touch "${CHECKPOINT_STEP2}"
    echo "$(date)" > "${CHECKPOINT_STEP2}"

    echo ""
    echo "✓ Step 2 complete: Model trained with entity awareness"
    echo "  Checkpoint saved: ${CHECKPOINT_STEP2}"
    echo ""
fi

# Step 3: Detailed Evaluation
if [ -f "${CHECKPOINT_STEP3}" ]; then
    echo "========================================================================"
    echo "STEP 3: SKIPPED (already completed)"
    echo "========================================================================"
    echo "  Checkpoint found: ${CHECKPOINT_STEP3}"
    echo "  Results: ${OUTPUT_DIR}/eval_squad and ${OUTPUT_DIR}/eval_addsent"
    echo ""
else
    echo "========================================================================"
    echo "STEP 3: Detailed Evaluation"
    echo "========================================================================"
    echo ""

    echo "Evaluating on SQuAD (clean)..."
    python3 run.py \
        --do_eval \
        --task qa \
        --dataset squad \
        --model "${OUTPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}/eval_squad" \
        --per_device_eval_batch_size 32

    echo ""
    echo "Evaluating on AddSent (adversarial)..."
    python3 run.py \
        --do_eval \
        --task qa \
        --dataset ./data/addsent_eval.jsonl \
        --model "${OUTPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}/eval_addsent" \
        --per_device_eval_batch_size 32

    # Create checkpoint
    touch "${CHECKPOINT_STEP3}"
    echo "$(date)" > "${CHECKPOINT_STEP3}"

    echo ""
    echo "✓ Step 3 complete: Evaluation finished"
    echo "  Checkpoint saved: ${CHECKPOINT_STEP3}"
    echo ""
fi

# Print summary
echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Model saved to: ${OUTPUT_DIR}"
echo "Training data: ${OUTPUT_DATA}"
echo ""
echo "Checkpoints:"
echo "  ✓ Step 1: ${CHECKPOINT_STEP1}"
echo "  ✓ Step 2: ${CHECKPOINT_STEP2}"
echo "  ✓ Step 3: ${CHECKPOINT_STEP3}"
echo ""
echo "To re-run from scratch, delete: ${CHECKPOINT_DIR}"
echo "To re-run from step 2, delete: ${CHECKPOINT_STEP2} and ${CHECKPOINT_STEP3}"
echo "To re-run from step 3, delete: ${CHECKPOINT_STEP3}"
echo ""
echo "Next steps:"
echo "  1. Check results: ${OUTPUT_DIR}/entity_aware_results.json"
echo "  2. Compare with baseline (ELECTRA-base 80-20)"
echo "  3. Run error analysis to verify entity improvement"
echo "  4. Update report with findings"
echo ""
echo "Expected improvements:"
echo "  - Baseline AddSent EM: 68.90%"
echo "  - Target AddSent EM: 74-78% (+8-12% on entity errors)"
echo "  - Maintain SQuAD EM: ~85-90%"
echo "  - Entity substitution errors: 29.9% → 20-23%"
echo ""
echo "========================================================================"
