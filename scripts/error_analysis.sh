#!/bin/bash
# Run error analysis on AddSent adversarial evaluation

echo "Running error analysis on AddSent adversarial dataset..."
echo ""

python3 analyze_adversarial_errors.py \
  ./data/addsent_adversarial.jsonl \
  ./eval_results_adversarial/eval_predictions.jsonl \
  ./error_analysis_addsent.json

echo ""
echo "Done! Check error_analysis_addsent.json for detailed results."
