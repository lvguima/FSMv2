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

        # Forecast head: pooled token -> horizon * channels
        self.forecast_head = nn.Linear(self.d_model, self.pred_len * self.c_out)

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
            if torch.isfinite(grad_norm) and grad_norm > 1e3:
                self.momentum_state.mul_(1e3 / grad_norm)

            self.memory_state.mul_(1 - gate_value).add_(self.momentum_state, alpha=-self.lr_memory)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, use_state: bool = False, update_state: bool = False, **kwargs):
        """
        Args:
            x_enc: [B, L, C] input sequence
            use_state: if True, use mutable online memory buffers; else use learnable initializer
            update_state: if True and use_state, perform momentum+gate update in-place (no grad)
        Returns:
            pred: [B, pred_len, c_out]
            proxy_loss: scalar tensor
            stats: dict with debug info (no grad requirement)
        """
        B, L, _ = x_enc.shape
        x_proj = self.input_proj(x_enc)  # [B, L, D]

        q = self.q_proj(x_proj)
        k = self.k_proj(x_proj)
        v = self.v_proj(x_proj)

        mem_matrix = self._select_memory(use_state)
        h_mem, proxy_loss, grad_mem = self._neural_memory(mem_matrix, q, k, v)
        h_pers = self._persistent_memory(q)

        # Build core input: [persistent, memory, current]
        core_tokens = torch.cat([h_pers, h_mem, x_proj], dim=1)  # [B, 3L, D]
        core_in = core_tokens.transpose(0, 1)  # [S, B, D] for TransformerEncoder
        core_out = self.core(core_in)  # [S, B, D]
        pooled = core_out.mean(dim=0)  # [B, D]

        forecast = self.forecast_head(pooled)  # [B, pred_len * c_out]
        pred = forecast.view(B, self.pred_len, self.c_out)

        # Gate computed from pooled input
        gate_input = x_proj.mean(dim=1)
        gate_value = self.gate(gate_input).mean()  # scalar in [0,1]

        if use_state and update_state:
            self._update_online_memory(grad_mem.detach(), gate_value.detach())

        stats = {
            "gate": gate_value.detach(),
            "proxy_loss": proxy_loss.detach(),
            "grad_norm": torch.norm(grad_mem.detach()),
        }
        return pred, proxy_loss, stats
