#!/bin/bash

# M-Stream v4.0: Attention Memory 在线学习实验脚本
# Attention-based Memory (Neural Dictionary) for M-Stream

echo "=========================================="
echo "M-Stream v4.0 - Attention Memory 测试"
echo "=========================================="

# 实验配置
MODEL_NAME="MStream"
DATA="ETTh1"
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96
D_MODEL=128
NUM_PROTOTYPES=32  # Memory 字典大小
MOMENTUM_BETA=0.9
LR_TTT=0.001

# 延迟反馈参数
USE_DELAYED_FEEDBACK=1
DELAYED_BATCH_SIZE=8
DELAYED_MAX_WAIT=20
DELAYED_WEIGHT_DECAY=0.05
DELAYED_SUPERVISED_WEIGHT=0.7

# ========== 实验 1: v3.0 Baseline (MLP Memory) ==========
echo ""
echo "[实验 1/3] v3.0 Baseline: MLP Residual Adapter"
echo "------------------------------------------"

python online_runner.py \
    --model $MODEL_NAME \
    --data $DATA \
    --mode train_and_test \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --d_model $D_MODEL \
    --memory_rank $NUM_PROTOTYPES \
    --memory_type mlp \
    --momentum_beta $MOMENTUM_BETA \
    --lr_ttt $LR_TTT \
    --train_epochs 20 \
    --batch_size 32 \
    --use_delayed_feedback $USE_DELAYED_FEEDBACK \
    --delayed_batch_size $DELAYED_BATCH_SIZE \
    --delayed_max_wait_steps $DELAYED_MAX_WAIT \
    --delayed_weight_decay $DELAYED_WEIGHT_DECAY \
    --delayed_supervised_weight $DELAYED_SUPERVISED_WEIGHT \
    --des "v3_mlp_memory" \
    --itr 1

# ========== 实验 2: v4.0 Attention Memory (32 Prototypes) ==========
echo ""
echo "[实验 2/3] v4.0: Attention Memory (M=32)"
echo "------------------------------------------"

python online_runner.py \
    --model $MODEL_NAME \
    --data $DATA \
    --mode train_and_test \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --d_model $D_MODEL \
    --memory_rank 32 \
    --memory_type attention \
    --momentum_beta $MOMENTUM_BETA \
    --lr_ttt $LR_TTT \
    --train_epochs 20 \
    --batch_size 32 \
    --use_delayed_feedback $USE_DELAYED_FEEDBACK \
    --delayed_batch_size $DELAYED_BATCH_SIZE \
    --delayed_max_wait_steps $DELAYED_MAX_WAIT \
    --delayed_weight_decay $DELAYED_WEIGHT_DECAY \
    --delayed_supervised_weight $DELAYED_SUPERVISED_WEIGHT \
    --des "v4_attention_m32" \
    --itr 1

# ========== 实验 3: v4.0 Attention Memory (64 Prototypes) ==========
echo ""
echo "[实验 3/3] v4.0: Attention Memory (M=64)"
echo "------------------------------------------"

python online_runner.py \
    --model $MODEL_NAME \
    --data $DATA \
    --mode train_and_test \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --d_model $D_MODEL \
    --memory_rank 64 \
    --memory_type attention \
    --momentum_beta $MOMENTUM_BETA \
    --lr_ttt $LR_TTT \
    --train_epochs 20 \
    --batch_size 32 \
    --use_delayed_feedback $USE_DELAYED_FEEDBACK \
    --delayed_batch_size $DELAYED_BATCH_SIZE \
    --delayed_max_wait_steps $DELAYED_MAX_WAIT \
    --delayed_weight_decay $DELAYED_WEIGHT_DECAY \
    --delayed_supervised_weight $DELAYED_SUPERVISED_WEIGHT \
    --des "v4_attention_m64" \
    --itr 1

echo ""
echo "=========================================="
echo "v4.0 实验完成！"
echo "=========================================="
echo ""
echo "数据集划分: 60/10/30 (Train/Val/Test)"
echo "Memory 类型: Attention-based Neural Dictionary"
echo "融合机制: Gate Alpha ∈ [0.1, 0.9] (带下限约束)"
echo ""

