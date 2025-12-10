#!/bin/bash

# M-Stream v3.0: 延迟反馈在线学习实验脚本
# Delayed Feedback Online Learning for M-Stream

echo "=========================================="
echo "M-Stream v3.0 - Delayed Feedback 测试"
echo "=========================================="

# 实验配置
MODEL_NAME="MStream"
DATA="ETTh1"
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96
D_MODEL=128
MEMORY_RANK=32
MOMENTUM_BETA=0.9
LR_TTT=0.001

# 延迟反馈参数
USE_DELAYED_FEEDBACK=1
DELAYED_BATCH_SIZE=8
DELAYED_MAX_WAIT=20
DELAYED_WEIGHT_DECAY=0.05
DELAYED_SUPERVISED_WEIGHT=0.7

# ========== 实验 1: 基线 (仅 Proxy Loss) ==========
echo ""
echo "[实验 1/3] 基线: 仅使用 Proxy Loss (无延迟反馈)"
echo "------------------------------------------"

python online_runner.py \
    --model $MODEL_NAME \
    --data $DATA \
    --mode train_and_test \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --d_model $D_MODEL \
    --memory_rank $MEMORY_RANK \
    --momentum_beta $MOMENTUM_BETA \
    --lr_ttt $LR_TTT \
    --train_epochs 20 \
    --batch_size 32 \
    --use_delayed_feedback 0 \
    --des "baseline_proxy_only" \
    --itr 1

# ========== 实验 2: 延迟反馈 (Beta=0.7) ==========
echo ""
echo "[实验 2/3] 延迟反馈: 监督权重 = 0.7"
echo "------------------------------------------"

python online_runner.py \
    --model $MODEL_NAME \
    --data $DATA \
    --mode train_and_test \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --d_model $D_MODEL \
    --memory_rank $MEMORY_RANK \
    --momentum_beta $MOMENTUM_BETA \
    --lr_ttt $LR_TTT \
    --train_epochs 20 \
    --batch_size 32 \
    --use_delayed_feedback $USE_DELAYED_FEEDBACK \
    --delayed_batch_size $DELAYED_BATCH_SIZE \
    --delayed_max_wait_steps $DELAYED_MAX_WAIT \
    --delayed_weight_decay $DELAYED_WEIGHT_DECAY \
    --delayed_supervised_weight 0.7 \
    --des "delayed_feedback_beta0.7" \
    --itr 1

# ========== 实验 3: 延迟反馈 (Beta=0.5) ==========
echo ""
echo "[实验 3/3] 延迟反馈: 监督权重 = 0.5"
echo "------------------------------------------"

python online_runner.py \
    --model $MODEL_NAME \
    --data $DATA \
    --mode train_and_test \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --d_model $D_MODEL \
    --memory_rank $MEMORY_RANK \
    --momentum_beta $MOMENTUM_BETA \
    --lr_ttt $LR_TTT \
    --train_epochs 20 \
    --batch_size 32 \
    --use_delayed_feedback $USE_DELAYED_FEEDBACK \
    --delayed_batch_size $DELAYED_BATCH_SIZE \
    --delayed_max_wait_steps $DELAYED_MAX_WAIT \
    --delayed_weight_decay $DELAYED_WEIGHT_DECAY \
    --delayed_supervised_weight 0.5 \
    --des "delayed_feedback_beta0.5" \
    --itr 1

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="

