"""
M-Stream V7: Residual-Driven Orthogonal Memory Forecaster
面向在线时间序列预测的深度重构版本

核心架构特性:
1.  **L0 (Backbone)**: Decomposition Backbone (Trend + Seasonal)。
    - 在线阶段：Trend 层允许微调，应对趋势漂移。
2.  **L1 (Perception)**: Patch Encoder。
    - 在线阶段：完全冻结，提供稳定的特征空间。
    - 输入：直接使用 Instance Norm 后的原始数据，保留高频突变信息。
3.  **L2 (Cognitive)**: Orthogonal Attention Memory。
    - 机制：Neural Dictionary 检索 + 投影。
    - 代理任务：Next Patch Prediction (利用 t-1 上下文预测 t 时刻的 Patch)。
4.  **L3 (Fusion)**: Residual Fusion。
    - 公式：Pred = Backbone + Sigma * Memory。
    - Sigma：可学习的标量系数，避免 Gate 网络死锁。

Author: AI Assistant & User
Date: 2025-12-05
"""

import torch
import torch.nn as nn
from layers.RevIN import RevIN
import torch.nn.functional as F
from layers.Embed import PatchEmbedding
from layers.MomentumMemory import AttentionMemoryCell
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Autoformer_EncDec import series_decomp


class DecompositionBackbone(nn.Module):
    """
    Trend + Seasonal 分解骨干
    """
    def __init__(self, seq_len, pred_len, moving_avg=25):
        super().__init__()
        self.decomp = series_decomp(moving_avg)
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        # x: [B, L, C]
        seasonal_init, trend_init = self.decomp(x)
        
        # 维度变换: [B, L, C] -> [B, C, L] -> Linear -> [B, C, P] -> [B, P, C]
        trend_part = self.trend_linear(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_part = self.seasonal_linear(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        return trend_part + seasonal_part, seasonal_init


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # ============ Patching 配置 ============
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        # 计算 Patch 数量: N = (L - P) / S + 2 (含 padding)
        self.patch_num = int((configs.seq_len - self.patch_len) / self.stride + 2)
        
        # ============ L0: 稳健骨干层 ============
        moving_avg = getattr(configs, 'moving_avg', 25)
        self.backbone = DecompositionBackbone(self.seq_len, self.pred_len, moving_avg)
        
        # ============ L1: 敏捷感知层 (Encoder) ============
        # [Change] 这里的 padding 策略需与 PatchEmbedding 内部一致
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, self.stride, configs.dropout
        )
        
        # 轻量级 Encoder (1层足以，保持特征提取效率)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout), 
                    configs.d_model, configs.n_heads
                ),
                configs.d_model, 
                configs.d_ff, 
                dropout=configs.dropout, 
                activation=configs.activation
            ) for _ in range(1)],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # ============ L2: 正交认知层 (Memory) ============
        self.memory = AttentionMemoryCell(
            d_model=configs.d_model,
            num_prototypes=getattr(configs, 'memory_rank', 32),
            momentum_beta=getattr(configs, 'momentum_beta', 0.9),
            lr_ttt=getattr(configs, 'lr_ttt', 0.001),
            init_scale=0.5 # 初始残差系数
        )
        
        # ============ 预测头 (Heads) ============
        # 1. 未来预测头: d_model -> pred_len
        self.head_memory = nn.Linear(configs.d_model, configs.pred_len)
        
        # 2. [New] 代理任务头: d_model -> patch_len
        # 用于将特征映射回原始数据空间，计算重建损失
        self.head_proxy = nn.Linear(configs.d_model, self.patch_len) 
        
        # ============ L3: 残差融合层 ============
        # [New] 使用可学习标量代替 Gate 网络，避免死锁
        # 初始化为 0.1，让模型初期主要信赖 Backbone
        self.memory_scale = nn.Parameter(torch.tensor(0.1)) 
        
        # 归一化配置
        self.use_norm = getattr(configs, 'use_norm', True)
        self.revin = RevIN(configs.enc_in, affine=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mode='offline', mask=None):
        # 兼容性检查
        if self.task_name not in ['long_term_forecast', 'short_term_forecast', 'online_forecast']:
            return self.backbone(x_enc)[0] # Fallback
            
        # 1. RevIN / Instance Norm
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        x_enc = self.revin(x_enc, 'norm')

        # 2. Backbone Prediction (L0)
        # Backbone 负责捕捉宏观趋势和周期
        pred_backbone, _ = self.backbone(x_enc)
        
        # 3. Memory Input (L1)
        # [Change] 使用原始归一化数据 x_enc 作为输入，而非 seasonal_init
        # 这样 Memory 能看到高频噪声和突变
        # x_enc: [B, L, C] -> permute -> [B, C, L]
        raw_input = x_enc.permute(0, 2, 1)
        patch_out, n_vars = self.patch_embedding(raw_input) # [B*C, N, D]
        
        # Encoder Forward
        enc_out, _ = self.encoder(patch_out) # [B*C, N, D]
        
        # 4. Memory Interaction (L2)
        mem_out = self.memory(enc_out) # [B*C, N, D]
        
        # 5. Forecasting (L3)
        # 使用 Average Pooling 汇聚所有 Patch 的信息进行预测
        mem_out_pooled = mem_out.mean(dim=1)
        pred_memory = self.head_memory(mem_out_pooled) # [B*C, PredLen]
        
        # 6. Residual Fusion
        # 重塑 Backbone 输出以匹配 Memory 输出维度
        # pred_backbone: [B, PredLen, C] -> [B, C, PredLen] -> [B*C, PredLen]
        pred_backbone_reshaped = pred_backbone.permute(0, 2, 1).reshape(-1, self.pred_len)
        
        # [Change] 残差连接: Pred = Backbone + Scale * Memory
        # 使用 sigmoid 确保 scale 在 (0, 1) 之间，保证数值稳定性
        scale = torch.sigmoid(self.memory_scale)
        pred = pred_backbone_reshaped + scale * pred_memory
        
        # 7. Final Reshape & Denorm
        pred = pred.reshape(-1, n_vars, self.pred_len).permute(0, 2, 1)
        if self.use_norm:
            pred = pred * stdev + means
        pred = self.revin(pred, 'denorm')
        # =========================================================
        # [New] V7 核心：计算 Next Patch Prediction Proxy Loss
        # =========================================================
        # 目标：利用 t-1 时刻及之前的上下文，预测 t 时刻的 Patch
        # 这迫使 Memory 学习时序推演能力，而非简单的填空
        
        # 1. 获取输入的最后一个 Patch (Target)
        # 这里的逻辑需与 PatchEmbedding 的 unfold 保持一致
        # 简单起见，直接从 raw_input 切片
        # raw_input: [B, C, L] -> reshape -> [B*C, L]
        flat_input = raw_input.reshape(-1, self.seq_len)
        target_patch = flat_input[:, -self.patch_len:] # [B*C, PatchLen]
        
        # 2. 获取 Memory 在倒数第二个位置的输出 (Source)
        # enc_out: [B*C, N, D]
        # 假设 N 个 Patch 对应时间轴。取 N-1 位置的特征预测 N 位置的 Patch
        # 注意：mem_out 的长度 N 包含 padding，需确保索引正确
        # 取 -2 是比较安全的策略 (代表最新的完整历史上下文)
        context_feat = mem_out[:, -2, :] # [B*C, D]
        
        # 3. 预测下一个 Patch
        rec_patch = self.head_proxy(context_feat) # [B*C, PatchLen]
        
        # 4. 计算 MSE
        loss_proxy = F.mse_loss(rec_patch, target_patch)
        
        # 返回预测值、Proxy Loss 和调试信息
        return pred, loss_proxy, {'gate_value': scale.item()}

    def freeze_backbone(self):
        """
        在线 TTT 阶段的冻结策略
        V7 改进：允许 Trend Layer 微调，应对趋势漂移
        """
        # 1. 冻结 Seasonal 分解部分 (保持对周期模式的记忆)
        for param in self.backbone.decomp.parameters(): 
            param.requires_grad = False
        for param in self.backbone.seasonal_linear.parameters(): 
            param.requires_grad = False
        
        # 2. [Change] 解冻 Trend Layer (允许适应趋势变化)
        for param in self.backbone.trend_linear.parameters(): 
            param.requires_grad = True
            
        # 3. 冻结 Perception 层 (Encoder)
        for param in self.patch_embedding.parameters(): 
            param.requires_grad = False
        for param in self.encoder.parameters(): 
            param.requires_grad = False
        
        # 4. 解冻 Cognitive 层 (Memory & Heads)
        for param in self.memory.parameters(): 
            param.requires_grad = True
        for param in self.head_memory.parameters(): 
            param.requires_grad = True
        for param in self.head_proxy.parameters(): 
            param.requires_grad = True # [Important] 必须解冻，否则 TTT 无效
        
        # 5. 解冻 Fusion Scale
        self.memory_scale.requires_grad = True
        
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
            
    def get_memory_statistics(self):
        return self.memory.get_statistics()
    
    def enable_backbone_grad(self):
        """
        [New] 在线阶段：临时开启 Backbone 和 Encoder 的梯度
        用于延迟监督更新 (Slow Learning)
        """
        # 1. 解冻 Seasonal 部分
        for param in self.backbone.seasonal_linear.parameters(): 
            param.requires_grad = True
        # 注意：decomp (Moving Avg) 通常没有参数，或者参数很少，也可以开启
        for param in self.backbone.decomp.parameters():
            param.requires_grad = True
            
        # 2. 解冻 Perception (Encoder)
        for param in self.patch_embedding.parameters(): 
            param.requires_grad = True
        for param in self.encoder.parameters(): 
            param.requires_grad = True
            
    def disable_backbone_grad(self):
        """
        [New] 在线阶段：重新冻结 Backbone 和 Encoder
        用于 TTT 阶段 (Fast Learning) 保护特征空间
        """
        # 恢复冻结
        for param in self.backbone.seasonal_linear.parameters(): 
            param.requires_grad = False
        for param in self.backbone.decomp.parameters():
            param.requires_grad = False
        for param in self.patch_embedding.parameters(): 
            param.requires_grad = False
        for param in self.encoder.parameters(): 
            param.requires_grad = False
            
        # 注意：Trend Linear, Memory, Heads 保持开启 (由 freeze_backbone 控制)

    def forecast_offline(self, x_enc):
        # 复用 forward 逻辑 (forward 已经实现了全流程)
        return self.forward(x_enc, mode='offline')[:2] # 返回 pred, enc_out(proxy_loss)

    def forecast_online(self, x_enc):
        # [Fix] 移除 with torch.no_grad()
        # 梯度由外部 requires_grad 控制，不在此处强行截断
        # 直接复用 forward 逻辑，避免代码重复带来的不一致风险
        return self.forward(x_enc, mode='online')