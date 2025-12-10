#!/bin/bash

# M-Stream Online Learning Test Script for ETTh1
# 在 ETTh1 数据集上测试 M-Stream 的在线学习能力

echo "=========================================="
echo "M-Stream Online Learning Test on ETTh1"
echo "=========================================="

# 基础配置
MODEL="MStream"
DATA="ETTh1"
SEQ_LEN=96
PRED_LEN=96
BATCH_SIZE=32
D_MODEL=128

# Memory 配置
MEMORY_RANK=32
MOMENTUM_BETA=0.9
LR_TTT=0.001

# 在线学习配置
SURPRISE_THRESH=3.0
WARMUP_STEPS=50

# 训练配置
TRAIN_EPOCHS=10
LEARNING_RATE=0.0001

# ========================================
# Test 1: 训练 + 在线测试
# ========================================
echo ""
echo "[Test 1/4] Train (Offline) + Test (Online)"
echo "=========================================="

python online_runner.py \
  --model $MODEL \
  --data $DATA \
  --mode train_and_test \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --batch_size $BATCH_SIZE \
  --d_model $D_MODEL \
  --memory_rank $MEMORY_RANK \
  --momentum_beta $MOMENTUM_BETA \
  --lr_ttt $LR_TTT \
  --surprise_thresh $SURPRISE_THRESH \
  --warmup_steps $WARMUP_STEPS \
  --train_epochs $TRAIN_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --use_gpu True \
  --gpu 0

# ========================================
# Test 2: 仅在线测试 (使用已训练模型)
# ========================================
echo ""
echo "[Test 2/4] Test Only (Online)"
echo "=========================================="

python online_runner.py \
  --model $MODEL \
  --data $DATA \
  --mode test_only \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --batch_size $BATCH_SIZE \
  --d_model $D_MODEL \
  --memory_rank $MEMORY_RANK \
  --momentum_beta $MOMENTUM_BETA \
  --lr_ttt $LR_TTT \
  --surprise_thresh $SURPRISE_THRESH \
  --warmup_steps $WARMUP_STEPS \
  --use_gpu True \
  --gpu 0

# ========================================
# Test 3: 对比静态 vs 在线
# ========================================
echo ""
echo "[Test 3/4] Compare Static vs Online"
echo "=========================================="

python online_runner.py \
  --model $MODEL \
  --data $DATA \
  --mode compare \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --batch_size $BATCH_SIZE \
  --d_model $D_MODEL \
  --memory_rank $MEMORY_RANK \
  --momentum_beta $MOMENTUM_BETA \
  --lr_ttt $LR_TTT \
  --surprise_thresh $SURPRISE_THRESH \
  --warmup_steps $WARMUP_STEPS \
  --use_gpu True \
  --gpu 0

# ========================================
# Test 4: 消融实验
# ========================================
echo ""
echo "[Test 4/4] Ablation Study"
echo "=========================================="

python online_runner.py \
  --model $MODEL \
  --data $DATA \
  --mode ablation \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --batch_size $BATCH_SIZE \
  --d_model $D_MODEL \
  --memory_rank $MEMORY_RANK \
  --momentum_beta $MOMENTUM_BETA \
  --lr_ttt $LR_TTT \
  --surprise_thresh $SURPRISE_THRESH \
  --warmup_steps $WARMUP_STEPS \
  --use_gpu True \
  --gpu 0

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
