#!/bin/bash
# Evaluate all 5 original (non-augmented) models on both AddSent and SQuAD
# This will take approximately 1 hour

set -e

echo "=========================================="
echo "Evaluating All 5 Original Models"
echo "=========================================="
echo ""

# Configuration
ADDSENT_EVAL="./data/addsent_eval.jsonl"
SQUAD_EVAL="squad"
BATCH_SIZE=32

echo "Test datasets:"
echo "  AddSent: $ADDSENT_EVAL"
echo "  SQuAD: $SQUAD_EVAL"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Check if test datasets exist
if [ ! -f "$ADDSENT_EVAL" ]; then
    echo "❌ Error: AddSent eval file not found: $ADDSENT_EVAL"
    exit 1
fi

# Check if models exist
echo "Checking for trained models..."
declare -a RATIOS=("90_10" "80_20" "70_30" "60_40" "50_50")
all_exist=true

for ratio in "${RATIOS[@]}"; do
    model_path="./trained_model_adversarial_${ratio}"
    if [ ! -d "$model_path" ]; then
        echo "❌ Missing: $model_path"
        all_exist=false
    else
        echo "✅ Found: $model_path"
    fi
done

echo ""

if [ "$all_exist" = false ]; then
    echo "❌ Error: Some trained models are missing!"
    echo ""
    echo "Please train models first:"
    echo "  bash scripts/train_all_adversarial_models.sh"
    echo ""
    exit 1
fi

echo "All models found! ✅"
echo ""

# Evaluation function
evaluate_model() {
    local ratio=$1
    local model_path="./trained_model_adversarial_${ratio}"
    local eval_dir="./evaluation/adversarial_${ratio}"
    
    echo "=========================================="
    echo "Evaluating ${ratio} Model"
    echo "=========================================="
    echo ""
    
    # Evaluate on AddSent
    echo "Evaluating on AddSent (adversarial)..."
    python3 run.py \
        --do_eval \
        --task qa \
        --dataset "$ADDSENT_EVAL" \
        --model "$model_path" \
        --output_dir "$eval_dir" \
        --per_device_eval_batch_size $BATCH_SIZE
    
    echo ""
    
    # Evaluate on SQuAD
    echo "Evaluating on SQuAD (clean)..."
    python3 run.py \
        --do_eval \
        --task qa \
        --dataset "$SQUAD_EVAL" \
        --model "$model_path" \
        --output_dir "$eval_dir/squad" \
        --per_device_eval_batch_size $BATCH_SIZE
    
    echo ""
    echo "✅ ${ratio} evaluation complete!"
    echo "   Results saved to: $eval_dir"
    echo ""
}

# Evaluate all models
start_time=$(date +%s)

for ratio in "${RATIOS[@]}"; do
    evaluate_model "$ratio"
done

end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "=========================================="
echo "✅ All Models Evaluated Successfully!"
echo "=========================================="
echo ""
echo "Total evaluation time: ${minutes}m ${seconds}s"
echo ""

echo "Results directories:"
ls -d ./evaluation/adversarial_* 2>/dev/null | grep -v augmented | awk '{print "  " NR ". " $1}'

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "Compare all results:"
echo "  python3 scripts/compare_adversarial_models.py"
echo ""
echo "View individual results:"
for ratio in "${RATIOS[@]}"; do
    eval_file="./evaluation/adversarial_${ratio}/eval_metrics.json"
    if [ -f "$eval_file" ]; then
        echo "  cat $eval_file"
    fi
done
echo ""
