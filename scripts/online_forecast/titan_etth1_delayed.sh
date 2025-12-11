#!/bin/bash

# Titan-Stream ETTh1 with delayed feedback enabled.

set -e

DATA_ROOT=./data/ETT/
DATA_FILE=ETTh1.csv
MODEL=TitanStream
MODEL_ID=titan_delayed

# Offline training (short)
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ${MODEL_ID} \
  --model ${MODEL} \
  --data ETTh1 \
  --root_path ${DATA_ROOT} \
  --data_path ${DATA_FILE} \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 256 --n_heads 4 --e_layers 2 --d_ff 512 \
  --train_epochs 1 --batch_size 8 --patience 1 \
  --learning_rate 1e-3 \
  --chunk_size 0 \
  --chunk_len 0 \
  --clip_grad 1.0 \
  --lambda_proxy 0.1 \
  --use_gpu 0 \
  --des delayed

# Online evaluation with delayed feedback
python -u run.py \
  --task_name online_forecast \
  --is_training 0 \
  --model_id ${MODEL_ID} \
  --model ${MODEL} \
  --data ETTh1 \
  --root_path ${DATA_ROOT} \
  --data_path ${DATA_FILE} \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 256 --n_heads 4 --e_layers 2 --d_ff 512 \
  --batch_size 8 \
  --online_strategy proxy_delayed \
  --use_delayed_feedback \
  --delayed_batch_size 8 \
  --delayed_max_wait_steps 10 \
  --delayed_weight_decay 0.05 \
  --delayed_supervised_weight 0.7 \
  --delayed_weight_temperature 1.0 \
  --delayed_anomaly_boost 1.5 \
  --delayed_min_ready 2 \
  --use_gpu 0 \
  --des delayed
