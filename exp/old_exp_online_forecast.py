"""
Online Forecast Experiment Framework for M-Stream
在线预测实验框架

核心功能:
1. 离线预训练 (复用 Exp_Long_Term_Forecast)
2. 在线测试 (逐步推理 + 动量更新)
3. 惊奇度门控 (过滤异常数据)
4. 性能监控和可视化

Author: AI Assistant & User
Date: 2025-12-05 (Fixed v6.2)
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
import time
import numpy as np
from tqdm import tqdm
import random
from collections import deque
from typing import Optional


class Exp_Online_Forecast(Exp_Long_Term_Forecast):
    """
    在线预测实验类
    
    继承自 Exp_Long_Term_Forecast，添加在线学习功能
    """
    
    def __init__(self, args):
        super(Exp_Online_Forecast, self).__init__(args)
        
        # 在线学习特有参数
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
        self.replay_buffer_size = getattr(args, 'replay_buffer_size', 256)
        self.replay_sample_size = getattr(args, 'replay_sample_size', 32)
        self.refresh_interval = getattr(args, 'refresh_interval', 200)
        self.refresh_epochs = getattr(args, 'refresh_epochs', 1)
        self.refresh_sample_limit = getattr(args, 'refresh_sample_limit', 256)
        self.checkpoint_setting = getattr(args, 'checkpoint_setting', None)
        self._supervised_optimizer = None
        self._replay_storage = deque(maxlen=self.replay_buffer_size)
        self._refresh_storage = deque(maxlen=self.refresh_sample_limit)
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
            if os.path.exists(path):
                return path
        
        direct_path = candidate_path(setting)
        if os.path.exists(direct_path):
            return direct_path
        
        base_setting = self._strip_timestamp(setting)
        if not os.path.exists(checkpoint_dir):
            return None
        
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

    def online_test(self, setting, load_checkpoint=True, checkpoint_path=None):
        """
        在线测试主流程
        """
        print("\n" + "="*60)
        print("Starting Online Testing for M-Stream")
        print("="*60)
        
        # 1. 加载数据
        test_data, test_loader = self._get_data(flag='test')
        
        # 2. 加载预训练模型
        resolved_ckpt = checkpoint_path or self._resolve_checkpoint_path(setting, self.checkpoint_setting)
        if load_checkpoint:
            if resolved_ckpt and os.path.exists(resolved_ckpt):
                print(f"\nLoading pretrained model from: {resolved_ckpt}")
                self.model.load_state_dict(torch.load(resolved_ckpt))
            else:
                print(f"\n⚠️  Warning: Checkpoint not found for setting '{setting}'")
                print("   You may specify --checkpoint_setting <folder_name> to reuse an existing model.")
                print("   Continuing with randomly initialized weights (not recommended).")
        
        # 3. 冻结 Backbone (只更新 Memory)
        self.model.freeze_backbone()
        self.model.eval()  # 设置为评估模式
        self._reset_supervised_state()
        self._prepare_strategy()
        
        # 4. 初始化在线学习组件
        metrics = OnlineMetrics()
        
        if self.use_surprise_gate:
            surprise_gate = SurpriseGate(
                threshold_std=self.surprise_threshold_std,
                warmup_steps=self.warmup_steps,
                adaptive=True,
                window_size=100
            )
            print(f"\n✓ Surprise Gate enabled (threshold: {self.surprise_threshold_std} std)")
        else:
            surprise_gate = None
            print("\n✓ Surprise Gate disabled (always update)")
        
        # 初始化延迟反馈缓冲区
        if self.use_delayed_feedback:
            delayed_buffer = DelayedFeedbackBuffer(
                pred_horizon=self.delayed_horizon,
                batch_size=self.delayed_batch_size,
                max_buffer_size=200,
                max_wait_steps=self.delayed_max_wait_steps,
                weight_decay=self.delayed_weight_decay,
                supervised_weight=self.delayed_supervised_weight
            )
            print(f"\n✓ Delayed Feedback enabled:")
            print(f"    Batch size: {self.delayed_batch_size}")
            print(f"    Max wait steps: {self.delayed_max_wait_steps}")
            print(f"    Weight decay: {self.delayed_weight_decay}")
            print(f"    Supervised weight: {self.delayed_supervised_weight}")
        else:
            delayed_buffer = None
            print("\n✓ Delayed Feedback disabled (Proxy Loss only)")
        
        print(f"✓ Test samples: {len(test_data)}")
        print(f"✓ Batch size: {self.args.batch_size}")
        print(f"✓ Online Strategy: {self.online_strategy}")
        print("\nStarting online inference...\n")
        
        # 保存预测值和真实值用于 CSV 导出
        all_predictions = []
        all_targets = []
        
        # 5. 逐步在线推理
        enable_proxy_updates = self.online_strategy in ['proxy', 'proxy_delayed', 'proxy_supervised']
        
        with tqdm(total=len(test_loader), desc="Online Testing") as pbar:
            for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # 移动到设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # [Fix 1] 初始化更新计时器 (防止 NameError)
                t_update_start = time.time()
                
                # ========== Step 1: Pre-Hoc TTT (先自监督更新) ==========
                if enable_proxy_updates:
                    # 1. 临时前向传播获取特征 (开启梯度)
                    with torch.enable_grad():
                        # 调用 model 获取 enc_out，此时不计算最终预测
                        # 注意：这里返回的 enc_out 是带有梯度的
                        _, enc_out_tmp, _ = self.model(batch_x, mode='online')
                        
                        # 2. 计算 Proxy Loss
                        proxy_loss = self.model.memory.compute_proxy_loss(enc_out_tmp)
                        proxy_loss_value = proxy_loss.item()
                        
                        # 3. Surprise Gate 判断
                        if self.use_surprise_gate:
                            should_update, gate_info = surprise_gate.should_update(proxy_loss_value)
                            is_anomaly = gate_info.get('is_anomaly', False)
                        else:
                            should_update = True
                            is_anomaly = False
                        
                        # 4. 动量更新
                        if should_update:
                            self.model.memory.update_with_momentum(proxy_loss)
                else:
                    proxy_loss_value = 0.0
                    should_update = False
                    is_anomaly = False

                # [Fix 2] 记录 Proxy 更新耗时
                t_update = time.time() - t_update_start

                # ========== Step 2: 正式前向预测 (使用更新后的参数) ==========
                # 注意：如果是 Proxy 模式，这里使用的是刚刚更新过的 Memory
                t_start = time.time()
                with torch.no_grad():
                    pred, enc_out, debug_info = self.model(batch_x, mode='online')
                t_inference = time.time() - t_start
                
                # 记录 Gate Value
                if 'gate_value' in debug_info:
                    metrics.record_gate(debug_info['gate_value'])
                
                # 提取目标
                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                pred = pred[:, -self.args.pred_len:, f_dim:]
                
                # 处理监督更新状态 (仅用于统计)
                supervised_updated = False
                supervised_loss_value = 0.0
                
                # ========== Step 3: 延迟反馈或其他策略 ==========
                t_supervised_start = time.time()
                
                if self.use_delayed_feedback and delayed_buffer is not None:
                    # 添加样本和标签到缓冲区
                    delayed_buffer.add_sample(step, batch_x, enc_out)
                    delayed_buffer.add_label(step, target)
                    delayed_buffer.advance_time(step)
                    
                    if delayed_buffer.should_update(step, is_anomaly):
                        batch_data, weights = delayed_buffer.get_batch()
                        if batch_data is not None:
                            delayed_loss = self._supervised_update(
                                batch_data, 
                                weights,
                                self.delayed_supervised_weight
                            )
                            supervised_loss_value = delayed_loss
                            supervised_updated = True
                
                # [Fix 3] 修复即时监督模式 (proxy_supervised)
                elif self.online_strategy == 'proxy_supervised':
                    # 必须重新计算带梯度的预测，因为 Step 2 是 no_grad 的
                    with torch.enable_grad():
                        pred_grad, _, _ = self.model(batch_x, mode='online')
                        pred_grad = pred_grad[:, -self.args.pred_len:, f_dim:]
                        
                        # 计算监督 Loss 并更新 Memory
                        supervised_loss = self.supervised_criterion(pred_grad, target)
                        self.model.memory.update_with_momentum(supervised_loss)
                        
                    supervised_loss_value = supervised_loss.item()
                    supervised_updated = True

                elif self.online_strategy == 'naive_ft':
                    supervised_loss_value = self._naive_finetune_step(pred, target)
                    supervised_updated = supervised_loss_value > 0
                
                elif self.online_strategy == 'replay':
                     self._store_replay_sample(batch_x, target)
                     replay_loss = self._replay_update()
                     if replay_loss > 0:
                        supervised_loss_value = replay_loss
                        supervised_updated = True

                elif self.online_strategy == 'refresh':
                    self._store_refresh_sample(batch_x, target)
                    # Refresh 逻辑通常是每 N 步执行一次，这里简化处理
                    if (step + 1) % max(1, self.refresh_interval) == 0:
                        refresh_loss = self._offline_refresh_update()
                        if refresh_loss > 0:
                            supervised_loss_value = refresh_loss
                            supervised_updated = True

                t_supervised = time.time() - t_supervised_start

                # 保存预测结果
                all_predictions.append(pred.detach().cpu().numpy())
                all_targets.append(target.detach().cpu().numpy())
                
                # 记录所有指标
                metrics.record_prediction(pred, target)
                metrics.record_update(should_update, proxy_loss_value, supervised_updated, supervised_loss_value)
                metrics.record_time(t_inference, t_update, t_supervised)
                
                pbar.update(1)
                if (step + 1) % 100 == 0:
                    current_metrics = metrics.compute()
                    postfix_dict = {
                        'MSE': f"{current_metrics['mse']:.4f}",
                        'Proxy%': f"{current_metrics['update_rate']*100:.1f}",
                        'ProxyL': f"{proxy_loss_value:.4f}"
                    }
                    pbar.set_postfix(postfix_dict)
        
        # 6. 计算最终指标
        print("\n" + "="*60)
        print("Online Testing Completed!")
        print("="*60)
        
        final_metrics = metrics.compute()
        print_online_summary(final_metrics)
        
        # 7. 获取记忆模块统计
        memory_stats = self.model.get_memory_statistics()
        print("\n📊 Memory Module Statistics:")
        for key, value in memory_stats.items():
            print(f"  {key}: {value:.6f}")
        
        # 8. 保存结果
        if self.save_online_results:
            save_suffix = []
            if self.args.mode == 'train_and_test':
                save_suffix.append('train')
            elif self.args.mode in ['test_only', 'baseline']:
                save_suffix.append(self.online_strategy)
            elif self.args.mode in ['compare', 'ablation']:
                save_suffix.append(self.online_strategy)
            save_tag = f"{setting}_{'_'.join(save_suffix)}" if save_suffix else setting
            save_dir = os.path.join('./results', save_tag)
            os.makedirs(save_dir, exist_ok=True)
            
            trajectory = metrics.get_trajectory()
            
            # 保存指标和轨迹（包含 CSV）
            channel_names = None
            if self.args.features == 'MS':
                channel_names = [self.args.target]
            save_online_results(
                final_metrics, 
                trajectory, 
                save_dir, 
                'online_test',
                predictions=all_predictions,
                targets=all_targets,
                channel_names=channel_names
            )
            
            # 使用增强版可视化（生成多个图表）
            visualize_online_results_enhanced(
                metrics, 
                save_dir=save_dir, 
                setting='online_test',
                show=False
            )
            
            # 同时保留原有的单一图表（向后兼容）
            fig_path = os.path.join(save_dir, 'online_test_visualization.png')
            visualize_online_results(metrics, save_path=fig_path, show=False)
            
            # v6.0: 可视化 Memory Keys 相似度矩阵
            if hasattr(self.model, 'memory'):
                memory_vis_path = os.path.join(save_dir, 'memory_keys_similarity.png')
                sim_stats = visualize_memory_keys_similarity(
                    self.model, 
                    save_path=memory_vis_path, 
                    show=False
                )
                if sim_stats:
                    print(f"\n🔍 Memory Similarity Analysis:")
                    print(f"  Orthogonality Score: {sim_stats['orthogonality_score']:.4f}")
                    print(f"  Off-Diagonal Mean: {sim_stats['off_diagonal_mean']:.4f}")
            
            print(f"\n✓ Results saved to {save_dir}")
        
        # 9. v6.0: 打印 Memory 统计信息
        if hasattr(self.model, 'memory'):
            mem_stats = self.model.get_memory_statistics()
            print("\n🧠 Memory Statistics (v6.0):")
            print(f"  Total Updates: {mem_stats.get('update_count', 0)}")
            print(f"  Avg Proxy Loss: {mem_stats.get('avg_proxy_loss', 0.0):.6f}")
            print(f"  Alpha (Gate): {mem_stats.get('alpha_clamped', 0.0):.4f}")
            print(f"  Keys Norm: {mem_stats.get('keys_norm', 0.0):.4f}")
            print(f"  Values Norm: {mem_stats.get('values_norm', 0.0):.4f}")
            
            # v6.0: 正交性统计
            if 'keys_orthogonality' in mem_stats:
                orth_val = mem_stats['keys_orthogonality']
                print(f"  Keys Orthogonality: {orth_val:.6f} (closer to 0 is better)")
                # 评估正交性质量
                if orth_val < 0.1:
                    quality = "✓ Excellent"
                elif orth_val < 0.3:
                    quality = "○ Good"
                elif orth_val < 0.5:
                    quality = "△ Fair"
                else:
                    quality = "✗ Poor (Mode Collapse Risk)"
                print(f"  Orthogonality Quality: {quality}")
        
        # 10. 打印延迟反馈统计 (如果启用)
        if self.use_delayed_feedback and delayed_buffer is not None:
            buffer_stats = delayed_buffer.get_statistics()
            print("\n📦 Delayed Feedback Buffer Statistics:")
            for key, value in buffer_stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        return final_metrics
    
    def _supervised_update(
        self, 
        batch_data: list, 
        weights: np.ndarray,
        supervised_weight: float
    ) -> float:
        """
        使用延迟到达的真实标签进行监督更新
        """
        if not batch_data:
            return 0.0
        
        total_loss = 0.0
        criterion = nn.MSELoss(reduction='none')
        
        for idx, (sample_step, batch_x, batch_y, enc_out) in enumerate(batch_data):
            # 1. 前向传播 (重新计算预测, 开启梯度)
            with torch.enable_grad():
                pred, _, _ = self.model(batch_x, mode='online')
                
                # 提取目标
                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                pred = pred[:, -self.args.pred_len:, f_dim:]
                
                # 2. 计算监督损失 (加权)
                sample_loss = criterion(pred, target).mean()
                weighted_loss = sample_loss * weights[idx]
                
                # 3. 计算 Proxy Loss (用于混合梯度)
                # 注意：enc_out 也是从带梯度的 forward 来的
                # 但为了简单，我们这里主要关注 supervised
                # 如果想混合 proxy，需要保证 enc_out 也有梯度
                # 这里简化处理：只用 supervised
                
                # 4. 动量更新
                self.model.memory.update_with_momentum(weighted_loss)
            
            total_loss += sample_loss.item()
        
        avg_supervised_loss = total_loss / len(batch_data)
        return avg_supervised_loss
    
    def _reset_supervised_state(self):
        self._supervised_optimizer = None
        self._replay_storage.clear()
        self._refresh_storage.clear()
    
    def _prepare_strategy(self):
        strategy = getattr(self, 'online_strategy', 'proxy')
        
        if strategy == 'proxy_delayed':
            self.use_delayed_feedback = True
        elif strategy in ['naive_ft', 'replay', 'refresh', 'static']:
            self.use_delayed_feedback = False
        elif strategy == 'proxy_supervised':  # <--- 新增
            self.use_delayed_feedback = False
            
        if strategy in ['naive_ft', 'replay', 'refresh']:
            self.use_surprise_gate = False
            for param in self.model.memory.parameters():
                param.requires_grad = False
            for param in self.model.head_memory.parameters():
                param.requires_grad = True
        elif strategy == 'static':
            self.use_surprise_gate = False
            for param in self.model.head_memory.parameters():
                param.requires_grad = False
        else: # proxy, proxy_delayed, proxy_supervised
            for param in self.model.memory.parameters():
                param.requires_grad = True
            for param in self.model.head_memory.parameters():
                param.requires_grad = True
    
    def _get_supervised_optimizer(self):
        if self._supervised_optimizer is None:
            params = [p for p in self.model.head_memory.parameters() if p.requires_grad]
            if not params:
                return None
            self._supervised_optimizer = optim.Adam(params, lr=self.naive_ft_lr)
        return self._supervised_optimizer
    
    def _naive_finetune_step(self, pred, target):
        optimizer = self._get_supervised_optimizer()
        if optimizer is None:
            return 0.0
        loss = self.supervised_criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def _store_replay_sample(self, batch_x, target):
        self._replay_storage.append((
            batch_x.detach().cpu(),
            target.detach().cpu()
        ))
    
    def _replay_update(self):
        if len(self._replay_storage) < max(2, self.replay_sample_size):
            return 0.0
        sample_size = min(self.replay_sample_size, len(self._replay_storage))
        indices = np.random.choice(len(self._replay_storage), sample_size, replace=False)
        batch_x = torch.cat([self._replay_storage[i][0] for i in indices], dim=0).to(self.device)
        batch_y = torch.cat([self._replay_storage[i][1] for i in indices], dim=0).to(self.device)
        pred, _, _ = self.model(batch_x, mode='online')
        f_dim = -1 if self.args.features == 'MS' else 0
        target = batch_y[:, -self.args.pred_len:, f_dim:]
        pred = pred[:, -self.args.pred_len:, f_dim:]
        optimizer = self._get_supervised_optimizer()
        if optimizer is None:
            return 0.0
        loss = self.supervised_criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def _store_refresh_sample(self, batch_x, target):
        self._refresh_storage.append((
            batch_x.detach().cpu(),
            target.detach().cpu()
        ))
    
    def _offline_refresh_update(self):
        if not self._refresh_storage:
            return 0.0
        optimizer = self._get_supervised_optimizer()
        if optimizer is None:
            return 0.0
        samples = list(self._refresh_storage)
        losses = []
        for _ in range(max(1, self.refresh_epochs)):
            random.shuffle(samples)
            for batch_x_cpu, batch_y_cpu in samples:
                batch_x = batch_x_cpu.to(self.device)
                batch_y = batch_y_cpu.to(self.device)
                pred, _, _ = self.model(batch_x, mode='online')
                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                pred = pred[:, -self.args.pred_len:, f_dim:]
                loss = self.supervised_criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0
    
    def compare_static_vs_online(self, setting):
        """
        对比静态模型 vs 在线学习模型
        """
        print("\n" + "="*60)
        print("Comparing Static Model vs Online Learning Model")
        print("="*60)
        
        checkpoint_path = self._resolve_checkpoint_path(setting, self.checkpoint_setting)
        if not checkpoint_path:
            print(f"\n❌ Error: checkpoint not found for setting '{setting}'.")
            print("   Please pass --checkpoint_setting <existing_folder> or rerun training.")
            return {}
        
        # 1. 测试静态模型 (不更新)
        print("\n[1/2] Testing Static Model (No Updates)...")
        static_setting = f"{setting}_static"
        self.online_strategy = 'static'
        self.args.online_strategy = 'static'
        static_metrics = self.online_test(static_setting, load_checkpoint=True, checkpoint_path=checkpoint_path)
        
        # 2. 测试在线学习模型
        print("\n[2/2] Testing Online Learning Model (With Updates)...")
        online_setting = f"{setting}_proxy"
        self.online_strategy = 'proxy'
        self.args.online_strategy = 'proxy'
        online_metrics = self.online_test(online_setting, load_checkpoint=True, checkpoint_path=checkpoint_path)
        
        # 3. 对比结果
        print("\n" + "="*60)
        print("Comparison Results")
        print("="*60)
        
        comparison = {
            'static': static_metrics,
            'online': online_metrics,
            'improvement': {}
        }
        
        # 计算改进百分比
        for key in ['mse', 'mae', 'rmse']:
            if key in static_metrics and key in online_metrics:
                static_val = static_metrics[key]
                online_val = online_metrics[key]
                improvement = (static_val - online_val) / static_val * 100
                comparison['improvement'][key] = improvement
        
        print("\n📊 Performance Comparison:")
        print(f"{'Metric':<10} {'Static':<12} {'Online':<12} {'Improvement':<12}")
        print("-" * 50)
        for key in ['mse', 'mae', 'rmse']:
            static_val = static_metrics[key]
            online_val = online_metrics[key]
            improvement = comparison['improvement'][key]
            print(f"{key.upper():<10} {static_val:<12.6f} {online_val:<12.6f} {improvement:>+10.2f}%")
        
        print("\n" + "="*60)
        
        return comparison
    
    def ablation_study(self, setting):
        """
        消融实验: 测试不同配置的影响
        """
        print("\n" + "="*60)
        print("Ablation Study for M-Stream")
        print("="*60)
        
        results = {}
        
        # 保存原始配置
        original_beta = self.model.memory.beta
        original_use_gate = self.use_surprise_gate
        
        checkpoint_path = self._resolve_checkpoint_path(setting, self.checkpoint_setting)
        if not checkpoint_path:
            print(f"\n❌ Error: No checkpoint available for setting '{setting}'.")
            print("   Please provide --checkpoint_setting <folder> or run training first.")
            return results
        
        # 测试不同的动量因子
        beta_values = [0.0, 0.5, 0.9, 0.95]
        
        for beta in beta_values:
            print(f"\n[Testing] Momentum Beta = {beta}")
            
            try:
                # 重新加载模型并重置状态
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()
                
                # 重置 momentum buffer
                self.model.memory.reset_momentum()
                
                # 设置动量因子
                self.model.memory.beta = float(beta)
                
                # 运行测试
                metrics = self.online_test(setting, load_checkpoint=False)
                results[f'beta_{beta}'] = metrics
                
            except Exception as e:
                print(f"\n❌ Error during testing with beta={beta}: {str(e)}")
                results[f'beta_{beta}'] = {'mse': float('inf')}
        
        # 测试 Surprise Gate
        try:
            print(f"\n[Testing] Without Surprise Gate")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.model.memory.reset_momentum()
            self.model.memory.beta = original_beta
            self.use_surprise_gate = False
            metrics_no_gate = self.online_test(setting, load_checkpoint=False)
            results['no_surprise_gate'] = metrics_no_gate
        except Exception as e:
            print(f"\n❌ Error during testing without surprise gate: {str(e)}")
            results['no_surprise_gate'] = {'mse': float('inf')}
        
        try:
            print(f"\n[Testing] With Surprise Gate")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.model.memory.reset_momentum()
            self.use_surprise_gate = True
            metrics_with_gate = self.online_test(setting, load_checkpoint=False)
            results['with_surprise_gate'] = metrics_with_gate
        except Exception as e:
            print(f"\n❌ Error during testing with surprise gate: {str(e)}")
            results['with_surprise_gate'] = {'mse': float('inf')}
        
        # 恢复原始配置
        self.model.memory.beta = original_beta
        self.use_surprise_gate = original_use_gate
        
        print("\n" + "="*60)
        return results
    
    def run_baselines(self, setting):
        """
        按照 baseline 策略列表依次运行在线测试
        """
        checkpoint_path = self._resolve_checkpoint_path(setting, self.checkpoint_setting)
        if not checkpoint_path:
            print(f"\n❌ Error: checkpoint not found for setting '{setting}'.")
            return {}
        
        results = {}
        for strategy in self.baseline_strategies:
            tag = strategy.strip()
            if not tag:
                continue
            print("\n" + "="*40)
            print(f"Running baseline strategy: {tag}")
            print("="*40)
            self.online_strategy = tag
            self.args.online_strategy = tag
            strategy_setting = f"{setting}_{tag}"
            metrics = self.online_test(strategy_setting, load_checkpoint=True, checkpoint_path=checkpoint_path)
            results[tag] = metrics
        return results
