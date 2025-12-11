#!/bin/bash

# Titan-Stream ETTm1 with delayed feedback enabled.
# [修复] 统一离线/在线使用相同的 data 参数，确保 checkpoint 路径一致

set -e

DATA_ROOT=./data/ETT/
DATA_FILE=ETTm1.csv
MODEL=TitanStream
MODEL_ID=titan_delayed

DATA_NAME=ETTm1_Online

# Offline training
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ${MODEL_ID} \
  --model ${MODEL} \
  --data ${DATA_NAME} \
  --root_path ${DATA_ROOT} \
  --data_path ${DATA_FILE} \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 256 --n_heads 4 --e_layers 2 --d_ff 512 \
  --train_epochs 30 --batch_size 32 --patience 7 \
  --learning_rate 5e-4 \
  --lradj cosine \
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
  --data ${DATA_NAME} \
  --root_path ${DATA_ROOT} \
  --data_path ${DATA_FILE} \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
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
