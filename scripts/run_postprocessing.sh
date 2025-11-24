#!/bin/bash
#
# Post-Processing for Partial Match Errors
#
# This script applies NER-based entity expansion to fix partial match errors.
# No training required - pure inference-time fix!
#
# Goal: Fix "Partial Match" errors (30.6% of errors)
# Example: "Broncos" → "Denver Broncos"

set -e  # Exit on error

echo "========================================================================"
echo "Post-Processing for Partial Match Errors"
echo "========================================================================"
echo ""

# Configuration
MODEL_DIR="../trained_model_electra_80_20"
TEST_DATA="../data/addsent_eval.jsonl"
OUTPUT_DIR="../postprocessing_results"
MIN_EXPANSION_RATIO=1.3
SPACY_MODEL="en_core_web_sm"

echo "Configuration:"
echo "  Model: ${MODEL_DIR}"
echo "  Test data: ${TEST_DATA}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Min expansion ratio: ${MIN_EXPANSION_RATIO}"
echo "  spaCy model: ${SPACY_MODEL}"
echo ""

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

# Step 1: Run inference (if not already done)
echo "========================================================================"
echo "STEP 1: Running Inference"
echo "========================================================================"
echo ""

if [ ! -f "${OUTPUT_DIR}/predictions_raw.jsonl" ]; then
    echo "Running model inference on test set..."
    cd ..
    python ../run.py \
        --model "${MODEL_DIR}" \
        --task qa \
        --dataset "${TEST_DATA}" \
        --do_eval \
        --output_dir "${OUTPUT_DIR}" \
        --per_device_eval_batch_size 32
    
    cd scripts
    
    # Find predictions file
    PRED_FILE=$(find ${OUTPUT_DIR} -name "*predictions*.jsonl" | head -n 1)
    if [ -f "$PRED_FILE" ]; then
        cp "$PRED_FILE" "${OUTPUT_DIR}/predictions_raw.jsonl"
        echo "✓ Predictions saved to: ${OUTPUT_DIR}/predictions_raw.jsonl"
    else
        echo "ERROR: Predictions file not found"
        exit 1
    fi
else
    echo "✓ Using existing predictions: ${OUTPUT_DIR}/predictions_raw.jsonl"
fi

echo ""

# Step 2: Apply post-processing
echo "========================================================================"
echo "STEP 2: Applying NER-Based Post-Processing"
echo "========================================================================"
echo ""

python postprocess_partial_matches.py \
    --input "${OUTPUT_DIR}/predictions_raw.jsonl" \
    --output "${OUTPUT_DIR}/predictions_postprocessed.jsonl" \
    --spacy-model "${SPACY_MODEL}" \
    --min-expansion-ratio ${MIN_EXPANSION_RATIO}

if [ $? -ne 0 ]; then
    echo "ERROR: Post-processing failed"
    exit 1
fi

echo ""
echo "✓ Post-processing complete"
echo ""

# Step 3: Evaluate and compare
echo "========================================================================"
echo "STEP 3: Evaluating Performance"
echo "========================================================================"
echo ""

python evaluate_with_postprocessing.py \
    --predictions-file "${OUTPUT_DIR}/predictions_raw.jsonl" \
    --gold-file "${TEST_DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --min-expansion-ratio ${MIN_EXPANSION_RATIO}

echo ""
echo "✓ Evaluation complete"
echo ""

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
