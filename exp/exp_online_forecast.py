"""
Online Forecast Experiment Framework for M-Stream (V8)
在线预测实验框架 - 支持 Fast/Slow 双速学习架构

核心变更 (V8):
1. 延迟反馈阶段 (Delayed Feedback): 执行 Backbone 全量微调 (Slow Learning)。
2. 双速优化器: Backbone LR = 0.1 * Head LR。
3. 保持 Proxy TTT (Fast Learning) 仅更新 Memory。

Author: AI Assistant & User
Date: 2025-12-05
"""

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.online_utils import (
    OnlineMetrics, 
    SurpriseGate,
    DelayedFeedbackBuffer,
    visualize_online_results,
    visualize_online_results_enhanced,
    visualize_memory_keys_similarity,
    print_online_summary,
    save_online_results
)
import torch
import torch.nn as nn
from torch import optim
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import random
from collections import deque
from typing import Optional


class Exp_Online_Forecast(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_Online_Forecast, self).__init__(args)
        
        # 在线学习参数
        self.surprise_threshold_std = getattr(args, 'surprise_thresh', 3.0)
        self.warmup_steps = getattr(args, 'warmup_steps', 50)
        self.use_surprise_gate = getattr(args, 'use_surprise_gate', True)
        self.save_online_results = getattr(args, 'save_online_results', True)
        
        # 延迟反馈参数
        self.use_delayed_feedback = getattr(args, 'use_delayed_feedback', False)
        self.delayed_batch_size = getattr(args, 'delayed_batch_size', 8)
        self.delayed_max_wait_steps = getattr(args, 'delayed_max_wait_steps', 20)
        self.delayed_weight_decay = getattr(args, 'delayed_weight_decay', 0.05)
        self.delayed_supervised_weight = getattr(args, 'delayed_supervised_weight', 0.7)
        self.delayed_horizon = max(0, getattr(args, 'delayed_horizon', args.pred_len))
        
        self.online_strategy = getattr(args, 'online_strategy', 'proxy')
        self.baseline_strategies = getattr(args, 'baseline_strategies', ['static', 'proxy'])
        self.naive_ft_lr = getattr(args, 'naive_ft_lr', 1e-4)
        
        self.checkpoint_setting = getattr(args, 'checkpoint_setting', None)
        self._supervised_optimizer = None
        self.supervised_criterion = nn.MSELoss()
    
    def _strip_timestamp(self, setting: str) -> str:
        parts = setting.split('_')
        if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 6 \
                and parts[-2].isdigit() and len(parts[-2]) == 8:
            return '_'.join(parts[:-2])
        return setting
    
    def _resolve_checkpoint_path(self, setting: str, override_setting: Optional[str] = None) -> Optional[str]:
        checkpoint_dir = self.args.checkpoints
        def candidate_path(name: str) -> str:
            return os.path.join(checkpoint_dir, name, 'checkpoint.pth')
        if override_setting:
            path = candidate_path(override_setting)
            if os.path.exists(path): return path
        direct_path = candidate_path(setting)
        if os.path.exists(direct_path): return direct_path
        base_setting = self._strip_timestamp(setting)
        if not os.path.exists(checkpoint_dir): return None
        matches = []
        for item in os.listdir(checkpoint_dir):
            item_base = self._strip_timestamp(item)
            if item_base == base_setting:
                ckpt_path = candidate_path(item)
                if os.path.exists(ckpt_path):
                    matches.append((os.path.getmtime(ckpt_path), ckpt_path))
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            return matches[0][1]
        return None

    def _get_extra_params(self):
        """获取 TTT 阶段需要随 Memory 更新的参数 (ProxyHead, Scale, Trend)"""
        extra_params = []
        if hasattr(self.model, 'head_proxy'):
            extra_params.extend(list(self.model.head_proxy.parameters()))
        if hasattr(self.model, 'memory_scale'):
            extra_params.append(self.model.memory_scale)
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'trend_linear'):
            trend_params = [p for p in self.model.backbone.trend_linear.parameters() if p.requires_grad]
            extra_params.extend(trend_params)
        return extra_params

    def online_test(self, setting, load_checkpoint=True, checkpoint_path=None):
        print("\n" + "="*60)
        print("Starting Online Testing for M-Stream (V8: Fast/Slow)")
        print("="*60)
        if hasattr(self.model, "reset_memory"):
            self.model.reset_memory()
        
        test_data, test_loader = self._get_data(flag='test')
        print(f"✓ Test samples: {len(test_data)}")
        print(f"✓ Batch size: {self.args.batch_size}")
        print(f"✓ Online Strategy: {self.online_strategy}")
        print("\nStarting online inference...\n")
        resolved_ckpt = checkpoint_path or self._resolve_checkpoint_path(setting, self.checkpoint_setting)
        # Fallback: try corresponding long_term_forecast checkpoint if online setting not found
        if resolved_ckpt is None and self.args.task_name == 'online_forecast':
            offline_setting = setting.replace('online_forecast', 'long_term_forecast', 1)
            resolved_ckpt = self._resolve_checkpoint_path(offline_setting, self.checkpoint_setting)
        if load_checkpoint:
            if resolved_ckpt and os.path.exists(resolved_ckpt):
                print(f"\nLoading pretrained model from: {resolved_ckpt}")
                self.model.load_state_dict(torch.load(resolved_ckpt))
            else:
                print(f"\n⚠️  Warning: Checkpoint not found.")
        
        # 3. 初始冻结 (Backbone Seasonal/Encoder 冻结, Trend/Memory 开启)
        if hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()
        
        self.model.eval()
        self._reset_supervised_state()
        self._prepare_strategy()
        
        metrics = OnlineMetrics()
        
        if self.use_surprise_gate:
            surprise_gate = SurpriseGate(self.surprise_threshold_std, self.warmup_steps, adaptive=True)
            print(f"\n✓ Surprise Gate enabled (threshold: {self.surprise_threshold_std} std)")
        else:
            surprise_gate = None
            print("\n✓ Surprise Gate disabled")
        
        if self.use_delayed_feedback:
            delayed_buffer = DelayedFeedbackBuffer(
                pred_horizon=self.delayed_horizon,
                batch_size=self.delayed_batch_size,
                max_buffer_size=200,
                max_wait_steps=self.delayed_max_wait_steps,
                weight_decay=self.delayed_weight_decay,
                supervised_weight=self.delayed_supervised_weight,
                weight_temperature=getattr(self.args, "delayed_weight_temperature", 1.0),
                anomaly_boost=getattr(self.args, "delayed_anomaly_boost", 1.0),
                min_ready_for_anomaly=getattr(self.args, "delayed_min_ready", 1)
            )
            print(f"\n✓ Delayed Feedback enabled (Batch: {self.delayed_batch_size}, Horizon: {self.delayed_horizon})")
            print(f"  -> Backbone will be fine-tuned during delayed updates (Slow Learning).")
        else:
            delayed_buffer = None
            print("\n✓ Delayed Feedback disabled")
        
        all_predictions = []
        all_targets = []
        enable_proxy_updates = self.online_strategy in ['proxy', 'proxy_delayed', 'proxy_supervised']
        extra_params = self._get_extra_params()
        is_titan_stream = (self.args.model == 'TitanStream') or (self.model.__class__.__name__ == 'TitanStream')
        
        # 检测输出是否重定向到文件，如果是则禁用进度条以避免日志文件过大
        # 方法1: 通过参数控制
        disable_pbar = getattr(self.args, 'disable_progress_bar', False)
        if not disable_pbar:
            # 方法2: 检查 stdout 是否被 tee 包装（说明输出到文件）
            # run.py 中的 tee_stdout_stderr 会创建 _TeeStream 类
            if hasattr(sys.stdout, 'file_handle') or type(sys.stdout).__name__ == '_TeeStream':
                disable_pbar = True
            # 方法3: 检查原始流是否是终端
            else:
                try:
                    original_stdout = getattr(sys.stdout, 'stream', sys.stdout)
                    if hasattr(original_stdout, 'isatty'):
                        disable_pbar = not original_stdout.isatty()
                except:
                    pass
        
        with tqdm(total=len(test_loader), desc="Online Testing", disable=disable_pbar, file=sys.stdout) as pbar:
            for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                t_update_start = time.time()
                
                # ========== Step 1: Fast Learning (Pre-Hoc TTT) ==========
                should_update = False
                proxy_loss_value = 0.0
                is_anomaly = False

                if enable_proxy_updates:
                    if is_titan_stream:
                        # Titan-Stream: 先计算代理损失做门控判断，再按需更新内部记忆状态
                        with torch.no_grad():
                            _, proxy_loss, _ = self.model(batch_x, use_state=True, update_state=False)
                            proxy_loss_value = proxy_loss.item()

                        if self.use_surprise_gate:
                            should_update, gate_info = surprise_gate.should_update(proxy_loss_value)
                            is_anomaly = gate_info.get('is_anomaly', False)
                        else:
                            should_update = True

                        if should_update:
                            with torch.no_grad():
                                # 使用内部 momentum + gate 更新在线记忆
                                self.model(batch_x, use_state=True, update_state=True)
                    else:
                        with torch.enable_grad():
                            # 确保此时 Backbone 是冻结的 (除了 Trend)
                            _, proxy_loss, _ = self.model(batch_x, mode='online')
                            proxy_loss_value = proxy_loss.item()
                            
                            if self.use_surprise_gate:
                                should_update, gate_info = surprise_gate.should_update(proxy_loss_value)
                                is_anomaly = gate_info.get('is_anomaly', False)
                            else:
                                should_update = True
                            
                            if should_update and hasattr(self.model, 'memory'):
                                # 使用 Momentum SGD 更新 Memory (快速适应)
                                self.model.memory.update_with_momentum(proxy_loss, extra_params=extra_params)

                t_update = time.time() - t_update_start

                # ========== Step 2: Prediction ==========
                t_start = time.time()
                with torch.no_grad():
                    if is_titan_stream:
                        pred, _, debug_info = self.model(batch_x, use_state=True, update_state=False)
                    else:
                        pred, _, debug_info = self.model(batch_x, mode='online')
                t_inference = time.time() - t_start
                
                if 'gate_value' in debug_info:
                    metrics.record_gate(debug_info['gate_value'])
                elif 'gate' in debug_info:
                    metrics.record_gate(debug_info['gate'])
                
                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                pred = pred[:, -self.args.pred_len:, f_dim:]
                
                all_predictions.append(pred.detach().cpu().numpy())
                all_targets.append(target.detach().cpu().numpy())
                metrics.record_prediction(pred, target)

                # ========== Step 3: Slow Learning (Delayed Feedback) ==========
                supervised_updated = False
                supervised_loss_value = 0.0
                t_supervised_start = time.time()

                if self.use_delayed_feedback and delayed_buffer is not None:
                    delayed_buffer.add_sample(step, batch_x, None) # enc_out not needed for supervised
                    delayed_buffer.add_label(step, target)
                    delayed_buffer.advance_time(step)
                    
                    if delayed_buffer.should_update(step, is_anomaly):
                        batch_data, weights = delayed_buffer.get_batch()
                        if batch_data is not None:
                            # [V8 核心] 临时解冻 Backbone 进行全量微调（如可用）
                            if hasattr(self.model, 'enable_backbone_grad'):
                                self.model.enable_backbone_grad()
                            
                            # 使用 Adam 优化器更新所有参数
                            supervised_loss_value = self._supervised_update_adam(batch_data, weights)
                            
                            # 恢复冻结状态
                            if hasattr(self.model, 'disable_backbone_grad'):
                                self.model.disable_backbone_grad()
                            
                            supervised_updated = True
                
                elif self.online_strategy == 'naive_ft':
                    # 传入 batch_x
                    supervised_loss_value = self._naive_finetune_step(batch_x, target)
                    supervised_updated = supervised_loss_value > 0

                t_supervised = time.time() - t_supervised_start
                
                metrics.record_update(should_update, proxy_loss_value, supervised_updated, supervised_loss_value)
                metrics.record_time(t_inference, t_update, t_supervised)
                
                # 只在进度条启用时更新，避免日志文件过大
                if not disable_pbar:
                    pbar.update(1)
                    if (step + 1) % 100 == 0:
                        current_metrics = metrics.compute()
                        pbar.set_postfix({
                            'MSE': f"{current_metrics['mse']:.4f}",
                            'P_Loss': f"{proxy_loss_value:.4f}",
                            'S_Loss': f"{supervised_loss_value:.4f}"
                        })
                elif (step + 1) % 1000 == 0:
                    # 即使进度条禁用，也定期打印进度信息到日志
                    current_metrics = metrics.compute()
                    print(f"Progress: {step + 1}/{len(test_loader)} steps | "
                          f"MSE: {current_metrics['mse']:.4f} | "
                          f"P_Loss: {proxy_loss_value:.4f} | "
                          f"S_Loss: {supervised_loss_value:.4f}")
        
        print("\nOnline Testing Completed!")
        final_metrics = metrics.compute()
        print_online_summary(final_metrics)
        
        # 额外打印记忆模块统计信息
        if hasattr(self.model, 'get_memory_statistics'):
            mem_stats = self.model.get_memory_statistics()
            if isinstance(mem_stats, dict):
                print("\n🧠 Memory Statistics:")
                for key, value in mem_stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")
        
        if self.save_online_results:
            self._save_results(setting, metrics, all_predictions, all_targets)
        
        # 打印延迟反馈缓冲区统计
        if self.use_delayed_feedback and 'delayed_buffer' in locals() and delayed_buffer is not None:
            buffer_stats = delayed_buffer.get_statistics()
            if buffer_stats:
                print("\n📦 Delayed Feedback Buffer Statistics:")
                for key, value in buffer_stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
        
        return final_metrics

    def _save_results(self, setting, metrics, preds, targets):
        save_tag = f"{setting}_{self.online_strategy}"
        save_dir = os.path.join(self.args.result_dir, save_tag)
        os.makedirs(save_dir, exist_ok=True)
        
        final_metrics = metrics.compute()
        trajectory = metrics.get_trajectory()
        channel_names = [self.args.target] if self.args.features == 'MS' else None
        
        # 保存指标与轨迹（包含 CSV）
        save_online_results(
            final_metrics, 
            trajectory, 
            save_dir, 
            'online_test', 
            predictions=preds, 
            targets=targets, 
            channel_names=channel_names
        )
        
        # 增强可视化（多图）
        visualize_online_results_enhanced(
            metrics, 
            save_dir=save_dir, 
            setting='online_test', 
            show=False
        )
        
        # 单图可视化（向后兼容）
        fig_path = os.path.join(save_dir, 'online_test_visualization.png')
        visualize_online_results(metrics, save_path=fig_path, show=False)
        
        # Memory Keys 相似度分析
        if hasattr(self.model, 'memory'):
            memory_vis_path = os.path.join(save_dir, 'memory_keys_similarity.png')
            sim_stats = visualize_memory_keys_similarity(
                self.model, 
                save_path=memory_vis_path, 
                show=False
            )
            if sim_stats:
                print(f"\n🔍 Memory Similarity Analysis:")
                if 'orthogonality_score' in sim_stats:
                    print(f"  Orthogonality Score: {sim_stats['orthogonality_score']:.4f}")
                if 'off_diagonal_mean' in sim_stats:
                    print(f"  Off-Diagonal Mean: {sim_stats['off_diagonal_mean']:.4f}")
        
        print(f"\n✓ Results saved to {save_dir}")

    def _reset_supervised_state(self):
        self._supervised_optimizer = None

    def _prepare_strategy(self):
        strategy = getattr(self, 'online_strategy', 'proxy')
        self.use_delayed_feedback = (strategy == 'proxy_delayed')
        
        # 确保 Memory 模块可训练 (TTT需要)
        if strategy in ['proxy', 'proxy_delayed', 'proxy_supervised']:
            if hasattr(self.model, 'memory'):
                for param in self.model.memory.parameters(): param.requires_grad = True
            if hasattr(self.model, 'head_proxy'): 
                for param in self.model.head_proxy.parameters(): param.requires_grad = True
            if hasattr(self.model, 'memory_scale'): 
                self.model.memory_scale.requires_grad = True

    def _get_supervised_optimizer(self):
        """
        [V8] 获取全量微调的优化器 (双速学习率)
        [修复] 针对 TitanStream 更新参数分组规则
        """
        if self._supervised_optimizer is None:
            # 临时解冻所有层以获取参数列表
            if hasattr(self.model, 'enable_backbone_grad'):
                self.model.enable_backbone_grad()

            # 分组参数
            backbone_params = []
            head_params = []

            # [修复] 更新参数分组规则，兼容 TitanStream 和 MStream
            # Backbone (慢速学习): core/encoder, input_proj, q/k/v_proj
            # Head/Memory (快速学习): forecast_head, gate, memory, persistent
            backbone_keywords = [
                'backbone', 'encoder', 'embedding',  # MStream 原有
                'core', 'input_proj', 'q_proj', 'k_proj', 'v_proj'  # TitanStream
            ]
            head_keywords = [
                'memory', 'head', 'gate', 'persistent'  # TitanStream/MStream
            ]

            for name, param in self.model.named_parameters():
                if not param.requires_grad: continue

                # 优先检查 head 关键词
                is_head = any(kw in name for kw in head_keywords)
                is_backbone = any(kw in name for kw in backbone_keywords)

                if is_head and not is_backbone:
                    head_params.append(param)
                elif is_backbone:
                    backbone_params.append(param)
                else:
                    # 默认归为 head (快速学习)
                    head_params.append(param)

            # 恢复冻结状态 (如果不是 Naive FT 全开模式)
            if self.online_strategy != 'naive_ft' and hasattr(self.model, 'disable_backbone_grad'):
                self.model.disable_backbone_grad()

            # 定义双速优化器
            # Backbone LR: 0.1 * naive_ft_lr (更稳)
            # Head/Memory LR: naive_ft_lr
            if len(backbone_params) + len(head_params) == 0:
                return None
            if len(backbone_params) == 0:
                self._supervised_optimizer = optim.Adam(head_params, lr=self.naive_ft_lr)
            else:
                self._supervised_optimizer = optim.Adam([
                    {'params': backbone_params, 'lr': self.naive_ft_lr * 0.1},
                    {'params': head_params, 'lr': self.naive_ft_lr}
                ])

            print(f"\n🔧 Initialized Dual-Speed Optimizer:")
            print(f"  Backbone Params: {len(backbone_params)} (LR: {self.naive_ft_lr * 0.1:.2e})")
            print(f"  Head/Mem Params: {len(head_params)} (LR: {self.naive_ft_lr:.2e})")
            
        return self._supervised_optimizer

    def _supervised_update_adam(self, batch_data, weights):
        """
        使用 Adam 进行监督更新 (替代 update_with_momentum)
        """
        if not batch_data: return 0.0
        
        optimizer = self._get_supervised_optimizer()
        if optimizer is None:
            return 0.0
        criterion = nn.MSELoss(reduction='none')
        total_loss = 0.0
        
        # 简单起见，这里逐样本计算梯度并累积 (Gradient Accumulation)
        # 或者是将 batch_data 拼成一个大 Batch 一次 forward
        # 为了效率，拼成 Batch 更好
        
        # 1. 拼接 Batch
        # batch_data list of tuples: (step, x, y, enc_out)
        xs = torch.cat([item[1] for item in batch_data], dim=0) # [B_total, ...]
        ys = torch.cat([item[2] for item in batch_data], dim=0) # [B_total, ...]
        # 权重与样本对齐：若样本数超出权重，重复/截断；若权重不足则广播
        ws = torch.tensor(weights, device=self.device).float()  # [N_weights]
        if ws.numel() == 1:
            ws = ws.expand(xs.size(0))
        elif ws.numel() != xs.size(0):
            repeat_factor = int(torch.ceil(torch.tensor(xs.size(0) / ws.numel())).item())
            ws = ws.repeat(repeat_factor)[:xs.size(0)]
        
        # 2. Forward
        # TitanStream 兼容：无 mode 参数
        if hasattr(self.model, 'forward') and 'mode' in self.model.forward.__code__.co_varnames:
            pred, _, _ = self.model(xs, mode='online')
        else:
            pred, _, _ = self.model(xs, use_state=True, update_state=False)
        f_dim = -1 if self.args.features == 'MS' else 0
        target = ys[:, -self.args.pred_len:, f_dim:]
        pred = pred[:, -self.args.pred_len:, f_dim:]
        
        # 3. Weighted Loss
        # [B_total, PredLen, C] -> mean over PredLen, C -> [B_total]
        losses = criterion(pred, target).mean(dim=(1, 2))
        weighted_loss = (losses * ws).mean()
        
        # 4. Update
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        
        return weighted_loss.item()

    def _naive_finetune_step(self, batch_x, target): # <--- 增加 batch_x 参数
        optimizer = self._get_supervised_optimizer()
        if optimizer is None: return 0.0
        
        with torch.enable_grad():
            # 重新计算带梯度的 pred
            if hasattr(self.model, 'forward') and 'mode' in self.model.forward.__code__.co_varnames:
                pred_grad, _, _ = self.model(batch_x, mode='online')
            else:
                pred_grad, _, _ = self.model(batch_x, use_state=True, update_state=False)
            f_dim = -1 if self.args.features == 'MS' else 0
            pred_grad = pred_grad[:, -self.args.pred_len:, f_dim:]
            
            loss = self.supervised_criterion(pred_grad, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return loss.item()
