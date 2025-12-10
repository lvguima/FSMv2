#!/bin/bash

# Quick Test for M-Stream
# 快速测试脚本 - 使用小数据集和少量 epoch 验证整个流程

echo "=========================================="
echo "M-Stream Quick Test (Small Scale)"
echo "=========================================="

# 使用小规模配置进行快速测试
python online_runner.py \
  --model MStream \
  --data ETTh1 \
  --mode train_and_test \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 8 \
  --d_model 64 \
  --memory_rank 16 \
  --momentum_beta 0.9 \
  --lr_ttt 0.001 \
  --surprise_thresh 3.0 \
  --warmup_steps 10 \
  --train_epochs 2 \
  --learning_rate 0.001 \
  --patience 1 \
  --use_gpu True \
  --gpu 0

echo ""
echo "=========================================="
echo "Quick test completed!"
echo "=========================================="
