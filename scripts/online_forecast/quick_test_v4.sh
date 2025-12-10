#!/bin/bash

# M-Stream v4.0: 快速测试脚本 (验证代码正确性)
# Quick Test for v4.0 Implementation

echo "=========================================="
echo "M-Stream v4.0 - 快速验证测试"
echo "=========================================="

MODEL_NAME="MStream"
DATA="ETTh1"
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96
D_MODEL=64  # 减小模型以加快测试
MOMENTUM_BETA=0.9
LR_TTT=0.001

# ========== 测试 1: MLP Memory (v2.0) ==========
echo ""
echo "[测试 1/2] MLP Memory (v2.0 向后兼容)"
echo "------------------------------------------"

python online_runner.py \
    --model $MODEL_NAME \
    --data $DATA \
    --mode train_and_test \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --d_model $D_MODEL \
    --memory_type mlp \
    --memory_rank 16 \
    --momentum_beta $MOMENTUM_BETA \
    --lr_ttt $LR_TTT \
    --train_epochs 2 \
    --batch_size 32 \
    --use_delayed_feedback 0 \
    --des "quick_test_mlp" \
    --itr 1

# ========== 测试 2: Attention Memory (v4.0) ==========
echo ""
echo "[测试 2/2] Attention Memory (v4.0 新功能)"
echo "------------------------------------------"

python online_runner.py \
    --model $MODEL_NAME \
    --data $DATA \
    --mode train_and_test \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --d_model $D_MODEL \
    --memory_type attention \
    --memory_rank 16 \
    --momentum_beta $MOMENTUM_BETA \
    --lr_ttt $LR_TTT \
    --train_epochs 2 \
    --batch_size 32 \
    --use_delayed_feedback 0 \
    --des "quick_test_attention" \
    --itr 1

echo ""
echo "=========================================="
echo "快速测试完成！"
echo "=========================================="
echo ""
echo "✅ 如果两个测试都成功运行，说明 v4.0 实现正确"
echo "🚀 可以运行完整实验: bash ETTh1_v4_attention_memory.sh"
echo ""

