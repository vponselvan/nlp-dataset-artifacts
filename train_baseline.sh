#!/bin/bash

# Fine-tune ELECTRA-small on SQuAD v1.1 (Baseline)
# This establishes the baseline performance for comparison with adversarial datasets

echo "Starting baseline ELECTRA-small fine-tuning on SQuAD v1.1..."

python3 run.py \
  --do_train \
  --do_eval \
  --task qa \
  --dataset squad \
  --model google/electra-small-discriminator \
  --output_dir ./experiments/baseline_squad \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 3 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 500 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --seed 42 \
  --load_best_model_at_end \
  --metric_for_best_model eval_f1

echo "Training complete! Results saved to ./experiments/baseline_squad"
echo "Check eval_metrics.json for performance metrics (EM and F1 scores)"
