"""
Momentum Memory Cell for M-Stream (V7)
动量记忆单元 - 核心字典与更新机制

V7 变更:
1. 移除内部 Proxy Head 和 Loss 计算 (移交 Model 统一管理)
2. update_with_momentum 支持外部参数列表 (用于更新 Model 中的 Trend 和 Head)
3. 保持正交性约束逻辑

Author: AI Assistant & User
Date: 2025-12-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionMemoryCell(nn.Module):
    """
    注意力记忆单元 (Neural Dictionary)
    存储时序原型 (Keys) 和修正向量 (Values)
    """
    
    def __init__(
        self, 
        d_model: int,
        num_prototypes: int = 32,
        momentum_beta: float = 0.9,
        lr_ttt: float = 0.001,
        init_scale: float = 0.5
    ):
        super(AttentionMemoryCell, self).__init__()
        
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        self.beta = momentum_beta
        self.lr_ttt = lr_ttt
        
        # ========== 核心参数 ==========
        # Keys: [M, D] - 存储典型的错误/残差模式
        self.memory_keys = nn.Parameter(
            torch.randn(num_prototypes, d_model) / math.sqrt(d_model)
        )
        
        # Values: [M, D] - 存储对应的修正向量
        self.memory_values = nn.Parameter(
            torch.randn(num_prototypes, d_model) / math.sqrt(d_model)
        )
        
        # 输出投影: 调整维度适配残差
        self.output_proj = nn.Linear(d_model, d_model)
        
        # 动量缓冲 (Momentum Buffers)
        self.register_buffer('velocities_initialized', torch.tensor(False))
        self.velocities = {}
        
        # 统计信息
        self.register_buffer('update_count', torch.tensor(0))
        self.register_buffer('total_loss', torch.tensor(0.0))
    
    def _initialize_velocities(self, extra_params=None):
        """
        初始化动量缓冲
        不仅初始化自己的参数，也为传入的外部参数初始化动量
        """
        if not self.velocities_initialized:
            # 1. 内部参数
            self.velocities['memory_keys'] = torch.zeros_like(self.memory_keys.data)
            self.velocities['memory_values'] = torch.zeros_like(self.memory_values.data)
            for name, param in self.output_proj.named_parameters():
                self.velocities[f'output_proj.{name}'] = torch.zeros_like(param.data)
            self.velocities_initialized = torch.tensor(True)
            
        # 2. 外部参数 (每次检查，因为 extra_params 可能会变)
        if extra_params:
            for i, param in enumerate(extra_params):
                key = f'extra_param_{id(param)}' # 使用内存地址作为唯一ID
                if key not in self.velocities:
                    self.velocities[key] = torch.zeros_like(param.data)
    
    def forward(self, x):
        """
        检索与修正
        x: [B*C, N, D]
        """
        # 1. 计算相似度 (Query @ Keys^T)
        # [B*C, N, D] @ [D, M] -> [B*C, N, M]
        scores = torch.matmul(x, self.memory_keys.transpose(0, 1))
        scores = scores / math.sqrt(self.d_model)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 2. 检索 (Weights @ Values)
        # [B*C, N, M] @ [M, D] -> [B*C, N, D]
        retrieved = torch.matmul(attn_weights, self.memory_values)
        
        # 3. 投影
        out = self.output_proj(retrieved)
        
        return out
    
    def update_with_momentum(self, loss, extra_params=None):
        """
        执行 TTT 动量更新
        
        Args:
            loss: 计算好的 Proxy Loss 或 Supervised Loss
            extra_params: 需要一起更新的外部参数列表 (如 ProxyHead, TrendLayer)
        """
        if not loss.requires_grad:
            return False
        
        try:
            # 1. 准备参数列表
            my_params = [
                self.memory_keys,
                self.memory_values,
            ] + list(self.output_proj.parameters())
            
            all_params = my_params + (extra_params if extra_params else [])
            
            # 2. 初始化动量
            self._initialize_velocities(extra_params)
            
            # 3. 计算梯度
            grads = torch.autograd.grad(
                loss, 
                all_params, 
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
            
            # 4. 应用更新 (In-place SGD with Momentum)
            with torch.no_grad():
                for i, (param, grad) in enumerate(zip(all_params, grads)):
                    if grad is None:
                        continue
                    
                    # 确定 Velocity Key
                    if i == 0: key = 'memory_keys'
                    elif i == 1: key = 'memory_values'
                    elif i < 2 + len(list(self.output_proj.parameters())):
                        # output_proj params
                        idx = i - 2
                        name = list(self.output_proj.named_parameters())[idx][0]
                        key = f'output_proj.{name}'
                    else:
                        # extra params
                        key = f'extra_param_{id(param)}'
                    
                    # 获取 velocity
                    if key not in self.velocities:
                         self.velocities[key] = torch.zeros_like(param.data)
                    velocity = self.velocities[key]
                    
                    # V_t = beta * V_{t-1} + (1-beta) * grad
                    velocity.mul_(self.beta).add_(grad, alpha=1-self.beta)
                    
                    # Theta_t = Theta_{t-1} - lr * V_t
                    param.sub_(velocity, alpha=self.lr_ttt)
            
            # 5. 统计
            with torch.no_grad():
                self.update_count += 1
                self.total_loss += loss.item()
            
            return True
            
        except Exception as e:
            print(f"Momentum update failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset_momentum(self):
        with torch.no_grad():
            for name in self.velocities:
                self.velocities[name].zero_()
    
    def compute_orthogonal_loss(self):
        """
        计算 Keys 的正交性损失
        防止 Mode Collapse，保证字典多样性
        """
        # Normalize Keys
        keys_norm = F.normalize(self.memory_keys, p=2, dim=1)
        
        # Compute Gram Matrix
        correlation = torch.matmul(keys_norm, keys_norm.t())
        
        # Target: Identity Matrix
        identity = torch.eye(self.num_prototypes, device=keys_norm.device)
        
        # MSE Loss
        loss = F.mse_loss(correlation, identity)
        return loss
    
    def get_statistics(self):
        """获取监控指标"""
        stats = {
            'update_count': self.update_count.item(),
            'avg_loss': (self.total_loss / max(self.update_count, 1)).item(),
        }
        stats['keys_norm'] = torch.norm(self.memory_keys).item()
        stats['values_norm'] = torch.norm(self.memory_values).item()
        
        with torch.no_grad():
            keys_norm = F.normalize(self.memory_keys, p=2, dim=1)
            correlation = torch.matmul(keys_norm, keys_norm.t())
            mask = 1 - torch.eye(self.num_prototypes, device=correlation.device)
            # 计算非对角线元素的平均绝对值
            off_diagonal = (correlation * mask).abs().sum() / (self.num_prototypes * (self.num_prototypes - 1))
            stats['keys_orthogonality'] = off_diagonal.item()
        
        return stats


class MomentumMemoryCell(nn.Module):
    """
    Legacy MLP Adapter (保留以防需要 MLP 版本)
    """
    def __init__(self, d_model, hidden_dim=None, momentum_beta=0.9, lr_ttt=0.001):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.beta = momentum_beta
        self.lr_ttt = lr_ttt
        # ... (简化版，V7 主要用 AttentionMemoryCell)