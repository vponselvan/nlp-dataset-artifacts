#!/bin/bash
#
# Post-Processing for Partial Match Errors
#
# This script applies NER-based entity expansion to fix partial match errors.
# No training required - pure inference-time fix!
#
# Goal: Fix "Partial Match" errors (30.6% of errors)
# Example: "Broncos" → "Denver Broncos"
#
# Checkpoints: This script creates checkpoint files to allow resuming

set -e  # Exit on error

echo "========================================================================"
echo "Post-Processing for Partial Match Errors"
echo "========================================================================"
echo ""

# Configuration
MODEL_DIR="./trained_model_entity_aware"
TEST_DATA="./data/addsent_eval.jsonl"
OUTPUT_DIR="./postprocessing_results"
MIN_EXPANSION_RATIO=1.3
SPACY_MODEL="en_core_web_sm"

# Checkpoint files
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_STEP1="${CHECKPOINT_DIR}/step1_inference.done"
CHECKPOINT_STEP2="${CHECKPOINT_DIR}/step2_postprocessing.done"
CHECKPOINT_STEP3="${CHECKPOINT_DIR}/step3_evaluation.done"

echo "Configuration:"
echo "  Model: ${MODEL_DIR}"
echo "  Test data: ${TEST_DATA}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Min expansion ratio: ${MIN_EXPANSION_RATIO}"
echo "  spaCy model: ${SPACY_MODEL}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo ""

# Check for existing checkpoints
if [ -f "${CHECKPOINT_STEP3}" ]; then
    echo "✓ All steps already completed!"
    echo "  To re-run, delete: ${CHECKPOINT_DIR}"
    exit 0
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import spacy; print('✓ spaCy installed')" 2>/dev/null || {
    echo "ERROR: spaCy not installed"
    echo "Install with: pip install spacy"
    exit 1
}

python -c "import spacy; spacy.load('${SPACY_MODEL}'); print('✓ ${SPACY_MODEL} model available')" 2>/dev/null || {
    echo "Downloading spaCy model..."
    python -m spacy download ${SPACY_MODEL}
}

echo ""

# Step 1: Run inference
if [ -f "${CHECKPOINT_STEP1}" ]; then
    echo "========================================================================"
    echo "STEP 1: SKIPPED (already completed)"
    echo "========================================================================"
    echo "  Checkpoint found: ${CHECKPOINT_STEP1}"
    echo "  Predictions file: ${OUTPUT_DIR}/predictions_raw.jsonl"
    echo ""
else
    echo "========================================================================"
    echo "STEP 1: Running Inference"
    echo "========================================================================"
    echo ""

    echo "Running model inference on test set..."
    python3 run.py \
        --do_eval \
        --task qa \
        --dataset "${TEST_DATA}" \
        --model "${MODEL_DIR}" \
        --output_dir "${OUTPUT_DIR}/temp" \
        --per_device_eval_batch_size 32
    
    # Find predictions file
    PRED_FILE=$(find ${OUTPUT_DIR}/temp -name "*predictions*.jsonl" | head -n 1)
    if [ -f "$PRED_FILE" ]; then
        cp "$PRED_FILE" "${OUTPUT_DIR}/predictions_raw.jsonl"
        echo "✓ Predictions saved to: ${OUTPUT_DIR}/predictions_raw.jsonl"
    else
        echo "ERROR: Predictions file not found"
        exit 1
    fi

    # Create checkpoint
    touch "${CHECKPOINT_STEP1}"
    echo "$(date)" > "${CHECKPOINT_STEP1}"

    echo ""
    echo "✓ Step 1 complete: Inference finished"
    echo "  Checkpoint saved: ${CHECKPOINT_STEP1}"
    echo ""
fi

# Step 2: Apply post-processing
if [ -f "${CHECKPOINT_STEP2}" ]; then
    echo "========================================================================"
    echo "STEP 2: SKIPPED (already completed)"
    echo "========================================================================"
    echo "  Checkpoint found: ${CHECKPOINT_STEP2}"
    echo "  Postprocessed file: ${OUTPUT_DIR}/predictions_postprocessed.jsonl"
    echo ""
else
    echo "========================================================================"
    echo "STEP 2: Applying NER-Based Post-Processing"
    echo "========================================================================"
    echo ""

    python3 scripts/postprocess_partial_matches.py \
        --input "${OUTPUT_DIR}/predictions_raw.jsonl" \
        --output "${OUTPUT_DIR}/predictions_postprocessed.jsonl" \
        --spacy-model "${SPACY_MODEL}" \
        --min-expansion-ratio ${MIN_EXPANSION_RATIO}

    if [ $? -ne 0 ]; then
        echo "ERROR: Post-processing failed"
        exit 1
    fi

    # Create checkpoint
    touch "${CHECKPOINT_STEP2}"
    echo "$(date)" > "${CHECKPOINT_STEP2}"

    echo ""
    echo "✓ Step 2 complete: Post-processing finished"
    echo "  Checkpoint saved: ${CHECKPOINT_STEP2}"
    echo ""
fi

# Step 3: Evaluate and compare
if [ -f "${CHECKPOINT_STEP3}" ]; then
    echo "========================================================================"
    echo "STEP 3: SKIPPED (already completed)"
    echo "========================================================================"
    echo "  Checkpoint found: ${CHECKPOINT_STEP3}"
    echo "  Results: ${OUTPUT_DIR}/postprocessing_results.json"
    echo ""
else
    echo "========================================================================"
    echo "STEP 3: Evaluating Performance"
    echo "========================================================================"
    echo ""

    python3 scripts/evaluate_with_postprocessing.py \
        --model "${MODEL_DIR}" \
        --test-data "${TEST_DATA}" \
        --predictions-file "${OUTPUT_DIR}/predictions_raw.jsonl" \
        --gold-file "${TEST_DATA}" \
        --output-dir "${OUTPUT_DIR}" \
        --min-expansion-ratio ${MIN_EXPANSION_RATIO}

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
echo "POST-PROCESSING COMPLETE!"
echo "========================================================================"
echo ""
echo "Files generated:"
echo "  - Raw predictions: ${OUTPUT_DIR}/predictions_raw.jsonl"
echo "  - Post-processed: ${OUTPUT_DIR}/predictions_postprocessed.jsonl"
echo "  - Results: ${OUTPUT_DIR}/postprocessing_results.json"
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
echo "Key insights:"
echo "  - No training required (inference-time fix)"
echo "  - Fixes partial match errors (30.6% of errors)"
echo "  - Expected improvement: +2-4 points on EM"
echo "  - Example: 'Broncos' → 'Denver Broncos'"
echo ""
echo "Next steps:"
echo "  1. Review results in ${OUTPUT_DIR}/postprocessing_results.json"
echo "  2. Combine with other mitigation strategies"
echo "  3. Update report with findings"
echo ""
echo "========================================================================"
