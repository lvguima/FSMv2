import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TitanStream(nn.Module):
    """
    Titan-Stream (M-Stream V9) minimal implementation.
    Three-stream structure:
    - Neural long-term memory (linear attention state with momentum + gate).
    - Persistent memory (static KV queried by core).
    - Core forecaster (Transformer encoder) fusing [persistent, neural memory, current tokens].

    Forward returns (pred, proxy_loss, stats).
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.n_heads = configs.n_heads
        self.n_persistent = getattr(configs, "n_persistent", 32)
        self.beta_momentum = getattr(configs, "beta_momentum", 0.9)
        self.lr_memory = getattr(configs, "lr_memory", 0.01)
        self.gate_hidden = getattr(configs, "gate_hidden", 128)
        self.dropout = getattr(configs, "dropout", 0.1)
        self.activation = getattr(configs, "activation", "gelu")
        self.use_norm = getattr(configs, "use_norm", 1)

        # Input projection
        self.input_proj = nn.Linear(configs.enc_in, self.d_model)

        # Q/K/V projections for memory + persistent queries
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)

        # Persistent memory (static KV)
        self.persistent_k = nn.Parameter(torch.randn(self.n_persistent, self.d_model) * 0.02)
        self.persistent_v = nn.Parameter(torch.randn(self.n_persistent, self.d_model) * 0.02)

        # Gate network controlling forgetting rate
        self.gate = nn.Sequential(
            nn.Linear(self.d_model, self.gate_hidden),
            nn.GELU(),
            nn.Linear(self.gate_hidden, 1),
            nn.Sigmoid(),
        )

        # Core forecaster: Transformer encoder over concatenated tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=getattr(configs, "d_ff", 2048),
            dropout=self.dropout,
            activation=self.activation,
        )
        self.core = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, getattr(configs, "e_layers", 2))
        )

        # [改进] Forecast head: Attention Pooling + 2-layer MLP
        # 使用可学习的 attention 来聚合 core_out，而非简单 mean pooling
        # 这样既保留了时序信息，又避免了 flatten 导致的参数爆炸
        forecast_hidden_dim_config = getattr(configs, "forecast_hidden_dim", 0)
        self.forecast_hidden_dim = forecast_hidden_dim_config if forecast_hidden_dim_config > 0 else self.d_model * 2

        # Attention pooling: 学习一个 query 向量来聚合所有 token
        self.forecast_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.forecast_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=False  # [S, B, D] format
        )

        # 2-layer MLP for final prediction
        self.forecast_head = nn.Sequential(
            nn.Linear(self.d_model, self.forecast_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.forecast_hidden_dim, self.pred_len * self.c_out)
        )

        # Learnable initial memory state; online state buffers are updated in-place
        self.memory_init = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        self.register_buffer("memory_state", torch.zeros(self.d_model, self.d_model))
        self.register_buffer("momentum_state", torch.zeros(self.d_model, self.d_model))

        self.reset_memory()

    def reset_memory(self):
        """Reset online memory and momentum to the learnable initializer."""
        with torch.no_grad():
            self.memory_state.copy_(self.memory_init.data)
            self.momentum_state.zero_()

    def get_persistent_params(self):
        """[修复 2.2] 返回 Persistent Memory 参数"""
        return [self.persistent_k, self.persistent_v]

    def freeze_persistent(self):
        """[修复 2.2] 冻结 Persistent Memory"""
        self.persistent_k.requires_grad = False
        self.persistent_v.requires_grad = False

    def unfreeze_persistent(self):
        """[修复 2.2] 解冻 Persistent Memory（用于极低学习率微调）"""
        self.persistent_k.requires_grad = True
        self.persistent_v.requires_grad = True

    def _select_memory(self, use_state: bool):
        """
        Choose which memory matrix to use.
        - use_state=False: use learnable initializer (allows gradient flow in offline training).
        - use_state=True: use online state buffer (no gradient, for online/update mode).
        """
        return self.memory_state if use_state else self.memory_init

    def _neural_memory(self, mem_matrix, q, k, v):
        """
        Neural memory retrieval + proxy reconstruction.
        mem_matrix: [D, D]; q/k/v: [B, L, D]
        """
        # Memory retrieval as context
        h_mem = torch.einsum("ij,bld->bld", mem_matrix, q)

        # Proxy reconstruction of V from K
        v_hat = torch.einsum("ij,bld->bld", mem_matrix, k)
        proxy_diff = v_hat - v
        proxy_loss = proxy_diff.pow(2).mean()

        # Gradient surrogate for memory update: (v_hat - v) @ k^T
        grad_mem = torch.einsum("bld,blf->df", proxy_diff, k) / max(1, q.size(0) * q.size(1))

        return h_mem, proxy_loss, grad_mem

    def _persistent_memory(self, q):
        """
        Cross-attend current queries to persistent static memory.
        q: [B, L, D]
        returns: context [B, L, D]
        """
        attn_scores = torch.matmul(q, self.persistent_k.transpose(0, 1)) / math.sqrt(self.d_model)
        attn = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn, self.persistent_v)
        return context

    def _update_memory_differentiable(self, mem_matrix, momentum_matrix, grad_mem, gate_value):
        """
        [修复 1.2] 可微分的记忆更新（用于训练时的二阶梯度）

        Args:
            mem_matrix: [D, D] 当前记忆矩阵
            momentum_matrix: [D, D] 当前动量矩阵
            grad_mem: [D, D] 梯度（保留梯度图）
            gate_value: scalar tensor in [0,1]

        Returns:
            new_memory: [D, D] 更新后的记忆
            new_momentum: [D, D] 更新后的动量
        """
        # 动量更新（保留梯度）
        new_momentum = self.beta_momentum * momentum_matrix + (1 - self.beta_momentum) * grad_mem

        # 梯度归一化（保留梯度）
        grad_norm = torch.norm(new_momentum)
        max_grad_norm = 10.0  # [修复 2.1] 降低阈值从 1e3 到 10
        new_momentum = torch.where(
            (torch.isfinite(grad_norm) & (grad_norm > max_grad_norm)).unsqueeze(-1).unsqueeze(-1),
            new_momentum * (max_grad_norm / (grad_norm + 1e-8)),
            new_momentum
        )

        # 门控更新（保留梯度）
        new_memory = (1 - gate_value) * mem_matrix - self.lr_memory * new_momentum

        return new_memory, new_momentum

    def _update_online_memory(self, grad_mem, gate_value):
        """
        In-place online memory update using momentum and gate.
        grad_mem: [D, D] (detached)
        gate_value: scalar tensor in [0,1]
        """
        with torch.no_grad():
            self.momentum_state.mul_(self.beta_momentum).add_(grad_mem, alpha=1 - self.beta_momentum)
            # Optional normalization for stability
            grad_norm = torch.norm(self.momentum_state)
            max_grad_norm = 10.0  # [修复 2.1] 降低阈值从 1e3 到 10
            if torch.isfinite(grad_norm) and grad_norm > max_grad_norm:
                self.momentum_state.mul_(max_grad_norm / grad_norm)

            self.memory_state.mul_(1 - gate_value).add_(self.momentum_state, alpha=-self.lr_memory)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, use_state: bool = False, update_state: bool = False, differentiable_update: bool = False, external_memory=None, external_momentum=None, **kwargs):
        """
        Args:
            x_enc: [B, L, C] input sequence
            use_state: if True, use mutable online memory buffers; else use learnable initializer
            update_state: if True and use_state, perform momentum+gate update in-place (no grad)
            differentiable_update: [修复 1.2] if True, use differentiable update (for meta-learning)
            external_memory: [修复 inplace] optional external memory state (for training with gradients)
            external_momentum: [修复 inplace] optional external momentum state (for training with gradients)
        Returns:
            pred: [B, pred_len, c_out]
            proxy_loss: scalar tensor
            stats: dict with debug info (no grad requirement)
        """

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-6)
            stdev = torch.clamp(stdev, min=1e-3)
            stdev = torch.nan_to_num(stdev, nan=1.0, posinf=1.0, neginf=1.0)
            x_enc = torch.nan_to_num(x_enc / stdev, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            means, stdev = None, None
        
        B, L, _ = x_enc.shape
        x_proj = self.input_proj(x_enc)  # [B, L, D]

        q = self.q_proj(x_proj)
        k = self.k_proj(x_proj)
        v = self.v_proj(x_proj)

        # [修复 inplace] 优先使用外部传入的记忆状态（避免 inplace 操作）
        if external_memory is not None:
            mem_matrix = external_memory
        else:
            mem_matrix = self._select_memory(use_state)

        h_mem, proxy_loss, grad_mem = self._neural_memory(mem_matrix, q, k, v)
        h_pers = self._persistent_memory(q)

        # Build core input: [persistent, memory, current]
        core_tokens = torch.cat([h_pers, h_mem, x_proj], dim=1)  # [B, 3L, D]
        core_in = core_tokens.transpose(0, 1)  # [S, B, D] for TransformerEncoder
        core_out = self.core(core_in)  # [S, B, D]

        # [改进] Attention Pooling + 2-layer MLP (保留时序信息，避免参数爆炸)
        # 使用可学习的 query 向量通过 attention 聚合所有 token
        query = self.forecast_query.expand(-1, B, -1)  # [1, B, D]
        pooled, _ = self.forecast_attn(query, core_out, core_out)  # [1, B, D]
        pooled = pooled.squeeze(0)  # [B, D]

        forecast = self.forecast_head(pooled)  # [B, pred_len * c_out]
        pred = forecast.view(B, self.pred_len, self.c_out)
        # Denorm back to original scale if normalized
        if self.use_norm and means is not None and stdev is not None:
            pred = torch.nan_to_num(pred * stdev + means, nan=0.0, posinf=0.0, neginf=0.0)
        # Gate computed from pooled input
        gate_input = x_proj.mean(dim=1)
        gate_value = self.gate(gate_input).mean()  # scalar in [0,1]

        # [修复 1.2] 记忆更新逻辑
        updated_memory = None
        updated_momentum = None

        if update_state:
            if differentiable_update:
                # 可微分更新（训练时使用，保留梯度图）
                # [修复 inplace] 使用外部传入的动量或零初始化
                if external_momentum is not None:
                    momentum_matrix = external_momentum
                else:
                    momentum_matrix = torch.zeros_like(mem_matrix)
                updated_memory, updated_momentum = self._update_memory_differentiable(
                    mem_matrix, momentum_matrix, grad_mem, gate_value
                )
            elif use_state:
                # 原地更新（推理时使用，无梯度）
                self._update_online_memory(grad_mem.detach(), gate_value.detach())

        # Nan safety
        proxy_loss = torch.nan_to_num(proxy_loss, nan=0.0, posinf=1e4, neginf=-1e4)
        grad_mem = torch.nan_to_num(grad_mem, nan=0.0, posinf=0.0, neginf=0.0)

        stats = {
            "gate": gate_value.detach(),
            "proxy_loss": proxy_loss.detach(),
            "grad_norm": torch.norm(grad_mem.detach()),
            "mean": means if means is None else means.detach(),
            "std": stdev if stdev is None else stdev.detach(),
            "updated_memory": updated_memory,  # 可微分更新的新记忆（如果有）
            "updated_momentum": updated_momentum,  # 可微分更新的新动量（如果有）
        }
        return pred, proxy_loss, stats
