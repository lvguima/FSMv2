"""
Online Learning Utilities for M-Stream
在线学习工具函数

包含:
1. OnlineMetrics - 在线评估指标收集
2. SurpriseGate - 惊奇度门控
3. 可视化工具
4. 其他辅助函数

Author: AI Assistant & User
Date: 2025-11-19
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
import os


class OnlineMetrics:
    """
    在线评估指标收集器
    
    跟踪:
    - 预测性能 (MSE, MAE, RMSE)
    - 更新统计 (更新率, 异常率)
    - 时间性能 (推理时间, 更新时间)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有统计"""
        # 预测误差
        self.predictions = []
        self.targets = []
        self.mse_list = []
        self.mae_list = []
        
        # 更新统计
        self.update_flags = []  # True: 更新, False: 跳过
        self.proxy_losses = []
        self.anomaly_indices = []
        
        # 监督更新统计 (Delayed Feedback)
        self.supervised_update_flags = []  # True: 监督更新, False: 仅 Proxy 更新
        self.supervised_losses = []  # 监督损失值
        
        # 时间统计
        self.inference_times = []
        self.update_times = []
        self.supervised_update_times = []
        self.total_times = []
        
        # 门控统计
        self.gate_values = []
        
        # 步数
        self.total_steps = 0
    
    def record_prediction(self, pred: torch.Tensor, target: torch.Tensor):
        """
        记录预测结果
        
        Args:
            pred: [B, H, C]
            target: [B, H, C]
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        self.predictions.append(pred_np)
        self.targets.append(target_np)
        
        # 计算当前步的误差
        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        
        self.mse_list.append(mse)
        self.mae_list.append(mae)
    
    def record_update(self, updated: bool, proxy_loss: float, supervised: bool = False, supervised_loss: float = 0.0):
        """
        记录更新状态
        
        Args:
            updated: 是否执行了 Proxy 更新
            proxy_loss: 代理损失值
            supervised: 是否执行了监督更新
            supervised_loss: 监督损失值
        """
        self.update_flags.append(updated)
        self.proxy_losses.append(proxy_loss)
        self.supervised_update_flags.append(supervised)
        self.supervised_losses.append(supervised_loss)
        
        if not updated:
            self.anomaly_indices.append(self.total_steps)
        
        self.total_steps += 1
    
    def record_time(self, inference_time: float, update_time: float = 0.0, supervised_update_time: float = 0.0):
        """
        记录时间消耗
        
        Args:
            inference_time: 推理时间 (秒)
            update_time: Proxy 更新时间 (秒)
            supervised_update_time: 监督更新时间 (秒)
        """
        self.inference_times.append(inference_time)
        self.update_times.append(update_time)
        self.supervised_update_times.append(supervised_update_time)
        self.total_times.append(inference_time + update_time + supervised_update_time)
        
    def record_gate(self, gate_value: float):
        """
        记录门控值
        
        Args:
            gate_value: 门控值 (0~1)
        """
        if hasattr(gate_value, "detach"):
            try:
                gate_value = gate_value.detach().cpu().item()
            except Exception:
                gate_value = float(gate_value.detach().cpu().numpy())
        self.gate_values.append(float(gate_value))
    
    def compute(self) -> Dict[str, float]:
        """
        计算最终指标
        
        Returns:
            metrics: 包含所有指标的字典
        """
        # 预测性能
        mse = np.mean(self.mse_list)
        mae = np.mean(self.mae_list)
        rmse = np.sqrt(mse)

        # 额外评估指标（基于全量预测）
        rse = 0.0
        r2 = 0.0
        mape = 0.0
        if len(self.predictions) > 0 and len(self.targets) > 0:
            preds_all = np.concatenate(self.predictions, axis=0)
            targets_all = np.concatenate(self.targets, axis=0)

            residual = preds_all - targets_all
            sse = np.sum(residual ** 2)
            sst = np.sum((targets_all - np.mean(targets_all)) ** 2) + 1e-8  # 避免除零

            rse = float(np.sqrt(sse) / np.sqrt(sst)) if sst > 0 else 0.0
            r2 = float(1.0 - sse / sst) if sst > 0 else 0.0

            # 与 utils/metrics.py 保持一致：MAPE = mean(|true - pred| / true)
            denom = targets_all + 1e-6  # 避免除零，保留符号一致性
            mape = float(np.mean(np.abs(residual / denom)))
        
        # 更新统计
        update_rate = np.mean(self.update_flags) if self.update_flags else 0.0
        anomaly_rate = 1.0 - update_rate
        avg_proxy_loss = np.mean(self.proxy_losses) if self.proxy_losses else 0.0
        
        # 监督更新统计
        supervised_update_rate = np.mean(self.supervised_update_flags) if self.supervised_update_flags else 0.0
        supervised_losses_filtered = [l for l in self.supervised_losses if l > 0]
        avg_supervised_loss = np.mean(supervised_losses_filtered) if supervised_losses_filtered else 0.0
        
        # 时间统计
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0.0
        avg_update_time = np.mean(self.update_times) if self.update_times else 0.0
        avg_supervised_update_time = np.mean([t for t in self.supervised_update_times if t > 0]) if any(t > 0 for t in self.supervised_update_times) else 0.0
        total_time = avg_inference_time + avg_update_time + avg_supervised_update_time
        latency_p95 = np.percentile(self.total_times, 95) if self.total_times else 0.0
        throughput = 1.0 / np.mean(self.total_times) if self.total_times and np.mean(self.total_times) > 0 else 0.0
        
        # 稳定性指标
        mse_volatility = float(np.std(self.mse_list)) if len(self.mse_list) > 1 else 0.0
        drift_window = max(5, len(self.mse_list) // 10)
        if drift_window > 0 and len(self.mse_list) >= 2 * drift_window:
            start_mean = np.mean(self.mse_list[:drift_window])
            end_mean = np.mean(self.mse_list[-drift_window:])
            mse_drift = float(end_mean - start_mean)
        else:
            mse_drift = 0.0
        stability_index = float(abs(mse_drift))
        
        metrics = {
            # 预测性能
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'rse': float(rse),
            'r2': float(r2),
            'mape': float(mape),
            
            # Proxy 更新统计
            'update_rate': float(update_rate),
            'anomaly_rate': float(anomaly_rate),
            'avg_proxy_loss': float(avg_proxy_loss),
            'total_updates': int(sum(self.update_flags)),
            'total_anomalies': len(self.anomaly_indices),
            
            # 监督更新统计
            'supervised_update_rate': float(supervised_update_rate),
            'avg_supervised_loss': float(avg_supervised_loss),
            'total_supervised_updates': int(sum(self.supervised_update_flags)),
            
            # 时间统计
            'avg_inference_time_ms': float(avg_inference_time * 1000),
            'avg_update_time_ms': float(avg_update_time * 1000),
            'avg_supervised_update_time_ms': float(avg_supervised_update_time * 1000),
            'avg_total_time_ms': float(total_time * 1000),
            'latency_p95_ms': float(latency_p95 * 1000),
            'throughput_steps_per_sec': float(throughput),
            
            # 门控统计
            'avg_gate_value': float(np.mean(self.gate_values)) if self.gate_values else 0.0,
            
            # 稳定性/波动
            'mse_volatility': float(mse_volatility),
            'mse_drift': float(mse_drift),
            'stability_index': float(stability_index),
            
            # 总步数
            'total_steps': self.total_steps
        }
        
        return metrics
    
    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """
        获取完整的轨迹数据 (用于可视化)
        
        Returns:
            trajectory: 包含时间序列数据的字典
        """
        return {
            'mse': np.array(self.mse_list),
            'mae': np.array(self.mae_list),
            'proxy_loss': np.array(self.proxy_losses),
            'update_flags': np.array(self.update_flags),
            'anomaly_indices': np.array(self.anomaly_indices),
            'supervised_update_flags': np.array(self.supervised_update_flags),
            'supervised_losses': np.array(self.supervised_losses),
            'inference_times': np.array(self.inference_times),
            'update_times': np.array(self.update_times),
            'supervised_update_times': np.array(self.supervised_update_times),
            'gate_values': np.array(self.gate_values)
        }


class SurpriseGate:
    """
    惊奇度门控 (Surprise Gate)
    
    根据 Proxy Loss 判断是否应该更新模型:
    - Loss 正常: 更新 (适应概念漂移)
    - Loss 异常: 跳过 (避免异常数据污染)
    
    使用统计方法确定阈值:
        threshold = mean + k * std
    """
    
    def __init__(
        self, 
        threshold_std: float = 3.0,
        warmup_steps: int = 50,
        adaptive: bool = True,
        window_size: int = 100
    ):
        """
        Args:
            threshold_std: 阈值系数 (几倍标准差)
            warmup_steps: 预热步数 (用于估计初始分布)
            adaptive: 是否自适应调整阈值
            window_size: 滑动窗口大小 (用于自适应)
        """
        self.threshold_std = threshold_std
        self.warmup_steps = warmup_steps
        self.adaptive = adaptive
        self.window_size = window_size
        
        # 统计信息
        self.loss_history = []
        self.threshold = None
        self.mean = None
        self.std = None
        
        # 计数器
        self.step_count = 0
    
    def update_statistics(self, loss: float):
        """更新统计信息"""
        self.loss_history.append(loss)
        
        # 保持窗口大小
        if self.adaptive and len(self.loss_history) > self.window_size:
            self.loss_history = self.loss_history[-self.window_size:]
        
        # 计算均值和标准差
        if len(self.loss_history) >= self.warmup_steps:
            self.mean = np.mean(self.loss_history)
            self.std = np.std(self.loss_history)
            self.threshold = self.mean + self.threshold_std * self.std
    
    def should_update(self, proxy_loss: float) -> Tuple[bool, Dict[str, float]]:
        """
        判断是否应该更新
        
        Args:
            proxy_loss: 当前的代理损失
        
        Returns:
            should_update: True 表示应该更新
            info: 包含统计信息的字典
        """
        self.step_count += 1
        
        # 预热阶段: 总是更新
        if self.step_count <= self.warmup_steps:
            self.update_statistics(proxy_loss)
            return True, {
                'proxy_loss': proxy_loss,
                'threshold': None,
                'is_warmup': True
            }
        
        # 正常阶段: 根据阈值判断
        self.update_statistics(proxy_loss)
        
        should_update = proxy_loss <= self.threshold
        
        info = {
            'proxy_loss': proxy_loss,
            'threshold': self.threshold,
            'mean': self.mean,
            'std': self.std,
            'is_warmup': False,
            'is_anomaly': not should_update
        }
        
        return should_update, info
    
    def reset(self):
        """重置门控状态"""
        self.loss_history = []
        self.threshold = None
        self.mean = None
        self.std = None
        self.step_count = 0


def compute_threshold_from_validation(
    model,
    val_loader,
    device,
    percentile: float = 95.0
) -> float:
    """
    从验证集计算惊奇度阈值
    
    Args:
        model: M-Stream 模型
        val_loader: 验证集 DataLoader
        device: 设备
        percentile: 百分位数 (例如 95 表示取 95% 分位数)
    
    Returns:
        threshold: 阈值
    """
    model.eval()
    proxy_losses = []
    
    with torch.no_grad():
        for batch_x, _, _, _ in val_loader:
            batch_x = batch_x.float().to(device)
            
            # 获取 embedding
            _, enc_out = model(batch_x, mode='online')
            
            # 计算 proxy loss
            proxy_loss = model.memory.compute_proxy_loss(enc_out)
            proxy_losses.append(proxy_loss.item())
    
    # 计算阈值
    threshold = np.percentile(proxy_losses, percentile)
    
    print(f"Validation Proxy Loss Statistics:")
    print(f"  Mean: {np.mean(proxy_losses):.6f}")
    print(f"  Std: {np.std(proxy_losses):.6f}")
    print(f"  {percentile}th percentile: {threshold:.6f}")
    
    return threshold


def visualize_online_results(
    metrics: OnlineMetrics,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    可视化在线学习结果
    
    Args:
        metrics: OnlineMetrics 对象
        save_path: 保存路径 (可选)
        show: 是否显示图像
    """
    trajectory = metrics.get_trajectory()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. 预测误差随时间变化
    ax1 = axes[0]
    steps = np.arange(len(trajectory['mse']))
    ax1.plot(steps, trajectory['mse'], label='MSE', alpha=0.7)
    ax1.plot(steps, trajectory['mae'], label='MAE', alpha=0.7)
    
    # 标记异常点
    if len(trajectory['anomaly_indices']) > 0:
        ax1.scatter(
            trajectory['anomaly_indices'],
            trajectory['mse'][trajectory['anomaly_indices']],
            color='red',
            marker='x',
            s=50,
            label='Anomaly (No Update)',
            zorder=5
        )
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Prediction Error')
    ax1.set_title('Prediction Error Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Proxy Loss 和更新状态
    ax2 = axes[1]
    ax2.plot(steps, trajectory['proxy_loss'], label='Proxy Loss', color='blue', alpha=0.7)
    
    # 用颜色区分更新和跳过
    update_steps = steps[trajectory['update_flags']]
    skip_steps = steps[~trajectory['update_flags']]
    
    if len(update_steps) > 0:
        ax2.scatter(
            update_steps,
            trajectory['proxy_loss'][trajectory['update_flags']],
            color='green',
            marker='o',
            s=20,
            label='Updated',
            alpha=0.5
        )
    
    if len(skip_steps) > 0:
        ax2.scatter(
            skip_steps,
            trajectory['proxy_loss'][~trajectory['update_flags']],
            color='red',
            marker='x',
            s=30,
            label='Skipped',
            alpha=0.7
        )
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Proxy Loss')
    ax2.set_title('Proxy Loss and Update Status')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. 时间性能
    ax3 = axes[2]
    ax3.plot(steps, trajectory['inference_times'] * 1000, label='Inference Time', alpha=0.7)
    ax3.plot(steps, trajectory['update_times'] * 1000, label='Update Time', alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Time Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_online_summary(metrics_dict: Dict[str, float]):
    """
    打印在线学习结果摘要
    
    Args:
        metrics_dict: compute() 返回的指标字典
    """
    print("\n" + "="*60)
    print("Online Learning Results Summary")
    print("="*60)
    
    print("\n📊 Prediction Performance:")
    print(f"  MSE:  {metrics_dict['mse']:.6f}")
    print(f"  MAE:  {metrics_dict['mae']:.6f}")
    print(f"  RMSE: {metrics_dict['rmse']:.6f}")
    print(f"  RSE:  {metrics_dict['rse']:.6f}")
    print(f"  R2:   {metrics_dict['r2']:.6f}")
    print(f"  MAPE: {metrics_dict['mape']:.6f}")
    
    print("\n🔄 Update Statistics:")
    print(f"  Total Steps:    {metrics_dict['total_steps']}")
    print(f"  Proxy Updates:  {metrics_dict['total_updates']} ({metrics_dict['update_rate']*100:.2f}%)")
    print(f"  Anomalies:      {metrics_dict['total_anomalies']} ({metrics_dict['anomaly_rate']*100:.2f}%)")
    print(f"  Avg Proxy Loss: {metrics_dict['avg_proxy_loss']:.6f}")
    
    if 'avg_gate_value' in metrics_dict:
        print(f"  Avg Gate Value: {metrics_dict['avg_gate_value']:.4f}")
    
    # 监督更新统计（如果存在）
    if 'total_supervised_updates' in metrics_dict and metrics_dict['total_supervised_updates'] > 0:
        print("\n🎯 Supervised Update Statistics (Delayed Feedback):")
        print(f"  Total Supervised Updates: {metrics_dict['total_supervised_updates']}")
        print(f"  Supervised Update Rate:   {metrics_dict['supervised_update_rate']*100:.2f}%")
        print(f"  Avg Supervised Loss:      {metrics_dict['avg_supervised_loss']:.6f}")
    
    print("\n⏱️  Time Performance:")
    print(f"  Avg Inference Time: {metrics_dict['avg_inference_time_ms']:.2f} ms/step")
    print(f"  Avg Proxy Update:   {metrics_dict['avg_update_time_ms']:.2f} ms/step")
    if 'avg_supervised_update_time_ms' in metrics_dict and metrics_dict['avg_supervised_update_time_ms'] > 0:
        print(f"  Avg Supervised Update: {metrics_dict['avg_supervised_update_time_ms']:.2f} ms/step")
    print(f"  Avg Total Time:     {metrics_dict['avg_total_time_ms']:.2f} ms/step")
    print(f"  Latency P95:        {metrics_dict['latency_p95_ms']:.2f} ms")
    print(f"  Throughput:         {metrics_dict['throughput_steps_per_sec']:.2f} steps/s")
    
    print("\n📉 Stability Metrics:")
    print(f"  MSE Volatility:     {metrics_dict['mse_volatility']:.6f}")
    print(f"  MSE Drift:          {metrics_dict['mse_drift']:.6f}")
    print(f"  Stability Index:    {metrics_dict['stability_index']:.6f}")
    
    print("="*60 + "\n")


def save_online_results(
    metrics_dict: Dict[str, float],
    trajectory: Dict[str, np.ndarray],
    save_dir: str,
    setting: str,
    predictions: Optional[List[np.ndarray]] = None,
    targets: Optional[List[np.ndarray]] = None,
    channel_names: Optional[List[str]] = None
):
    """
    保存在线学习结果
    
    Args:
        metrics_dict: 指标字典
        trajectory: 轨迹数据
        save_dir: 保存目录
        setting: 实验设置名称
        predictions: 预测值列表 (可选)
        targets: 真实值列表 (可选)
    """
    import os
    import json
    import pandas as pd
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存指标 (JSON)
    metrics_path = os.path.join(save_dir, f'{setting}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # 保存预测结果为 CSV (如果提供)
    if predictions is not None and targets is not None:
        save_predictions_to_csv(predictions, targets, save_dir, setting, channel_names=channel_names)


def save_predictions_to_csv(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    save_dir: str,
    setting: str,
    reduce_overlap: bool = True,
    channel_names: Optional[List[str]] = None
):
    """
    将预测结果保存为 CSV 文件
    
    Args:
        predictions: 预测值列表，每个元素形状为 [B, H, C]
        targets: 真实值列表，每个元素形状为 [B, H, C]
        save_dir: 保存目录
        setting: 实验设置名称
        reduce_overlap: 是否减少重叠 (只保存 Horizon=1 的点，以及少量的完整预测)
    """
    import pandas as pd
    import os
    
    # 合并所有批次
    all_preds = np.concatenate(predictions, axis=0)  # [N, H, C]
    all_targets = np.concatenate(targets, axis=0)    # [N, H, C]
    
    N, H, C = all_preds.shape
    
    # 1. 保存 "Streaming View" (Horizon=1) - 最常用
    # 这代表了模型在每个时间步对"下一步"的预测
    stream_rows = []
    channel_labels = [f'channel_{c}' for c in range(C)] if not channel_names else channel_names
    for t in range(N):
        for c in range(C):
            pred_val = all_preds[t, 0, c]
            true_val = all_targets[t, 0, c]
            stream_rows.append({
                'step': t,
                'channel': channel_labels[c],
                'pred': pred_val,
                'true': true_val,
                'error': pred_val - true_val,
                'abs_error': abs(pred_val - true_val)
            })
    
    stream_df = pd.DataFrame(stream_rows)
    stream_path = os.path.join(save_dir, f'{setting}_predictions_stream.csv')
    stream_df.to_csv(stream_path, index=False, float_format='%.6f')
    print(f"Streaming predictions (H=1) saved to {stream_path}")
    
    # 2. 生成可视化图表 (True vs Pred)
    visualize_predictions_aligned(stream_df, save_dir, setting)


def visualize_predictions_aligned(stream_df, save_dir, setting):
    """
    可视化真实值与预测值曲线 (基于 H=1 的流式预测)
    """
    import matplotlib.pyplot as plt
    # import seaborn as sns
    
    # 获取通道列表
    channels = stream_df['channel'].unique()
    n_channels = len(channels)
    
    # 最多画 4 个通道，避免太拥挤
    plot_channels = channels[:min(4, n_channels)]
    
    fig, axes = plt.subplots(len(plot_channels), 1, figsize=(15, 4 * len(plot_channels)), sharex=True)
    if len(plot_channels) == 1:
        axes = [axes]
        
    for ax, c in zip(axes, plot_channels):
        data = stream_df[stream_df['channel'] == c]
        
        ax.plot(data['step'], data['true'], label='Ground Truth', color='black', alpha=0.6, linewidth=1.5)
        ax.plot(data['step'], data['pred'], label='Prediction (H=1)', color='dodgerblue', alpha=0.8, linewidth=1.5)
        
        # 计算该通道的 MSE
        mse = (data['error'] ** 2).mean()
        ax.set_title(f'Channel {c} - MSE: {mse:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 局部放大插图 (Zoom-in)
        # 选取中间 100 个点
        if len(data) > 200:
            start = len(data) // 2
            end = start + 100
            zoom_data = data.iloc[start:end]
            
            # 创建插图
            ins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
            ins.plot(zoom_data['step'], zoom_data['true'], color='black', alpha=0.6)
            ins.plot(zoom_data['step'], zoom_data['pred'], color='dodgerblue', alpha=0.8)
            ins.set_title('Zoom (100 steps)', fontsize=8)
            ins.set_xticks([])
            ins.set_yticks([])
            
            # 指示插图位置
            ax.indicate_inset_zoom(ins, edgecolor="black")
            
    plt.xlabel('Time Step')
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, f'{setting}_forecast_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Forecast comparison figure saved to {fig_path}")
    plt.close()


def visualize_online_results_enhanced(
    metrics: OnlineMetrics,
    save_dir: str,
    setting: str,
    show: bool = False
):
    """
    增强版可视化在线学习结果（生成多个图表）
    
    Args:
        metrics: OnlineMetrics 对象
        save_dir: 保存目录
        setting: 实验设置名称
        show: 是否显示图像
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    trajectory = metrics.get_trajectory()
    metrics_summary = metrics.compute()
    steps = np.arange(len(trajectory['mse']))
    
    # ========== 图1: 综合视图 (4子图) ==========
    fig1, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # 1.1 预测误差随时间变化
    ax1 = axes[0]
    ax1.plot(steps, trajectory['mse'], label='MSE', alpha=0.7, linewidth=2)
    ax1.plot(steps, trajectory['mae'], label='MAE', alpha=0.7, linewidth=2)
    
    # 标记异常点
    if len(trajectory['anomaly_indices']) > 0:
        ax1.scatter(
            trajectory['anomaly_indices'],
            trajectory['mse'][trajectory['anomaly_indices']],
            color='red',
            marker='x',
            s=100,
            label='Anomaly (No Update)',
            zorder=5
        )
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Prediction Error', fontsize=12)
    ax1.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 1.2 Proxy Loss 和更新状态
    ax2 = axes[1]
    ax2.plot(steps, trajectory['proxy_loss'], label='Proxy Loss', color='blue', alpha=0.7, linewidth=2)
    
    # 用颜色区分更新和跳过
    update_steps = steps[trajectory['update_flags']]
    skip_steps = steps[~trajectory['update_flags']]
    
    if len(update_steps) > 0:
        ax2.scatter(
            update_steps,
            trajectory['proxy_loss'][trajectory['update_flags']],
            color='green',
            marker='o',
            s=30,
            label='Updated',
            alpha=0.5
        )
    
    if len(skip_steps) > 0:
        ax2.scatter(
            skip_steps,
            trajectory['proxy_loss'][~trajectory['update_flags']],
            color='red',
            marker='x',
            s=50,
            label='Skipped (Anomaly)',
            alpha=0.8
        )
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Proxy Loss (log scale)', fontsize=12)
    ax2.set_title('Proxy Loss and Update Status', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 1.3 时间性能
    ax3 = axes[2]
    ax3.plot(steps, trajectory['inference_times'] * 1000, label='Inference Time', alpha=0.7, linewidth=2)
    ax3.plot(steps, trajectory['update_times'] * 1000, label='Update Time', alpha=0.7, linewidth=2)
    total_times = (trajectory['inference_times'] + trajectory['update_times']) * 1000
    ax3.plot(steps, total_times, label='Total Time', alpha=0.7, linewidth=2, linestyle='--')
    
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Time (ms)', fontsize=12)
    ax3.set_title('Time Performance', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 1.4 门控值变化 (Gate Value)
    ax4 = axes[3]
    if len(trajectory['gate_values']) > 0:
        ax4.plot(steps, trajectory['gate_values'], label='Gate Value (Memory Weight)', color='purple', alpha=0.7, linewidth=2)
        ax4.set_ylabel('Gate Value (0-1)', fontsize=12)
        ax4.set_ylim([0, 1.1])
        ax4.set_title('Dynamic Gate Evolution', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, f'{setting}_overview.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"Overview figure saved to {fig1_path}")
    if not show:
        plt.close()
    
    # ========== 图2: 误差分布直方图 ==========
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    ax_mse = axes2[0]
    ax_mse.hist(trajectory['mse'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax_mse.axvline(np.mean(trajectory['mse']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(trajectory["mse"]):.4f}')
    ax_mse.axvline(np.median(trajectory['mse']), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(trajectory["mse"]):.4f}')
    ax_mse.set_xlabel('MSE', fontsize=12)
    ax_mse.set_ylabel('Frequency', fontsize=12)
    ax_mse.set_title('MSE Distribution', fontsize=14, fontweight='bold')
    ax_mse.legend(fontsize=10)
    ax_mse.grid(True, alpha=0.3)
    
    ax_mae = axes2[1]
    ax_mae.hist(trajectory['mae'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax_mae.axvline(np.mean(trajectory['mae']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(trajectory["mae"]):.4f}')
    ax_mae.axvline(np.median(trajectory['mae']), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(trajectory["mae"]):.4f}')
    ax_mae.set_xlabel('MAE', fontsize=12)
    ax_mae.set_ylabel('Frequency', fontsize=12)
    ax_mae.set_title('MAE Distribution', fontsize=14, fontweight='bold')
    ax_mae.legend(fontsize=10)
    ax_mae.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, f'{setting}_error_distribution.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"Error distribution figure saved to {fig2_path}")
    if not show:
        plt.close()
    
    # ========== 图3: 滑动窗口统计 ==========
    window_size = min(50, len(steps) // 10)
    if window_size > 1:
        fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8))
        
        # 计算滑动窗口平均
        mse_rolling = np.convolve(trajectory['mse'], np.ones(window_size)/window_size, mode='valid')
        mae_rolling = np.convolve(trajectory['mae'], np.ones(window_size)/window_size, mode='valid')
        rolling_steps = steps[:len(mse_rolling)]
        
        ax_rolling1 = axes3[0]
        ax_rolling1.plot(rolling_steps, mse_rolling, label=f'MSE (window={window_size})', linewidth=2)
        ax_rolling1.plot(rolling_steps, mae_rolling, label=f'MAE (window={window_size})', linewidth=2)
        ax_rolling1.set_xlabel('Time Step', fontsize=12)
        ax_rolling1.set_ylabel('Error', fontsize=12)
        ax_rolling1.set_title(f'Rolling Average Error (Window Size: {window_size})', fontsize=14, fontweight='bold')
        ax_rolling1.legend(fontsize=10)
        ax_rolling1.grid(True, alpha=0.3)
        
        # 更新率滑动窗口
        update_rolling = np.convolve(trajectory['update_flags'].astype(float), np.ones(window_size)/window_size, mode='valid')
        
        ax_rolling2 = axes3[1]
        ax_rolling2.plot(rolling_steps, update_rolling * 100, linewidth=2, color='green')
        ax_rolling2.axhline(np.mean(trajectory['update_flags']) * 100, color='red', linestyle='--', linewidth=2, label=f'Overall: {np.mean(trajectory["update_flags"])*100:.1f}%')
        ax_rolling2.set_xlabel('Time Step', fontsize=12)
        ax_rolling2.set_ylabel('Update Rate (%)', fontsize=12)
        ax_rolling2.set_title(f'Rolling Update Rate (Window Size: {window_size})', fontsize=14, fontweight='bold')
        ax_rolling2.legend(fontsize=10)
        ax_rolling2.grid(True, alpha=0.3)
        ax_rolling2.set_ylim([0, 105])
        
        plt.tight_layout()
        fig3_path = os.path.join(save_dir, f'{setting}_rolling_statistics.png')
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"Rolling statistics figure saved to {fig3_path}")
        if not show:
            plt.close()
    
    # ========== 图4: 指标总览表 ==========
    summary_fig, ax_summary = plt.subplots(figsize=(10, 4))
    ax_summary.axis('off')
    table_data = [
        ['Metric', 'Value', 'Metric', 'Value'],
        ['MSE', f"{metrics_summary['mse']:.4f}", 'MAE', f"{metrics_summary['mae']:.4f}"],
        ['RMSE', f"{metrics_summary['rmse']:.4f}", 'MSE Drift', f"{metrics_summary['mse_drift']:.4f}"],
        ['MSE Volatility', f"{metrics_summary['mse_volatility']:.4f}", 'Stability Index', f"{metrics_summary['stability_index']:.4f}"],
        ['Avg total time (ms)', f"{metrics_summary['avg_total_time_ms']:.2f}", 'Latency P95 (ms)', f"{metrics_summary['latency_p95_ms']:.2f}"],
        ['Throughput (step/s)', f"{metrics_summary['throughput_steps_per_sec']:.2f}", 'Proxy update rate (%)', f"{metrics_summary['update_rate']*100:.2f}"],
    ]
    summary_table = ax_summary.table(cellText=table_data, cellLoc='center', loc='center')
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10)
    summary_table.scale(1, 1.5)
    summary_path = os.path.join(save_dir, f'{setting}_metrics_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Metrics summary figure saved to {summary_path}")
    if not show:
        plt.close()
    
    if show:
        plt.show()


class MovingAverage:
    """移动平均工具"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float) -> float:
        """更新并返回移动平均"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return np.mean(self.values)
    
    def get(self) -> float:
        """获取当前移动平均"""
        return np.mean(self.values) if self.values else 0.0


class DelayedFeedbackBuffer:
    """
    延迟反馈缓冲区 (Delayed Feedback Buffer)
    
    用于在线学习中处理延迟到达的真实标签：
    1. Replay Buffer: 存储 (X_t, t) 等待标签到达
    2. Ready Queue: 存储标签已到达的 (X_t, Y_t, t, emb_t)
    3. 自适应触发: 累积到阈值或检测到异常时批量更新
    
    工作流程:
    - 时刻 t: 添加样本 (X_t, emb_t) 到 Replay Buffer
    - 时刻 t+H: 标签 Y_t 到达，移动到 Ready Queue
    - 触发条件满足时: 从 Ready Queue 取出批量样本进行监督更新
    """
    
    def __init__(
        self,
        pred_horizon: int,
        batch_size: int = 8,
        max_buffer_size: int = 200,
        max_wait_steps: int = 20,
        weight_decay: float = 0.05,
        supervised_weight: float = 0.7,
        weight_temperature: float = 1.0,
        anomaly_boost: float = 1.0,
        min_ready_for_anomaly: int = 1
    ):
        """
        Args:
            pred_horizon: 预测视野 (H)，标签延迟到达的步数
            batch_size: 批量更新的触发阈值
            max_buffer_size: 最大缓冲区大小
            max_wait_steps: 强制更新的最大等待步数
            weight_decay: 样本时间衰减率 (λ)
            supervised_weight: 监督损失的权重 (β)
        """
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.max_wait_steps = max_wait_steps
        self.weight_decay = weight_decay
        self.weight_temperature = max(weight_temperature, 1e-6)
        self.anomaly_boost = anomaly_boost
        self.min_ready_for_anomaly = max(1, min_ready_for_anomaly)
        self.supervised_weight = supervised_weight
        
        # Replay Buffer: 存储等待标签的样本
        # 格式: {step: {'batch_x': tensor, 'enc_out': tensor, 'label': tensor or None, 'available_step': int}}
        self.replay_buffer = {}
        
        # Ready Queue: 存储可用于监督更新的样本
        self.ready_queue = []
        
        # 标签可能先于样本或样本被移除时暂存
        self.pending_labels = {}
        
        # 统计信息
        self.current_step = 0
        self.last_update_step = 0
        self.total_supervised_updates = 0
        self.total_samples_used = 0
        self.anomaly_trigger_count = 0
        self.last_update_reason = ""
        self.last_is_anomaly = False
    
    def add_sample(self, step: int, batch_x: torch.Tensor, enc_out: Optional[torch.Tensor]):
        """
        添加样本到 Replay Buffer (等待标签)
        
        Args:
            step: 样本所属的时间步
            batch_x: 输入数据 [B, L, C]
            enc_out: Encoder 输出 [B*C, N, D]
        """
        sample = {
            'batch_x': batch_x.detach().clone(),
            'enc_out': enc_out.detach().clone() if enc_out is not None else None,
            'label': None,
            'available_step': step + self.pred_horizon
        }
        
        if step in self.pending_labels:
            sample['label'] = self.pending_labels.pop(step)
        
        self.replay_buffer[step] = sample
        
        # 限制缓冲区大小
        if len(self.replay_buffer) > self.max_buffer_size:
            oldest_step = min(self.replay_buffer.keys())
            removed = self.replay_buffer.pop(oldest_step)
            # 如果移除的样本已经带有标签，避免丢失标签信息
            if removed['label'] is not None:
                self.pending_labels[oldest_step] = removed['label']
    
    def add_label(self, step: int, batch_y: torch.Tensor):
        """
        为指定样本添加真实标签
        
        Args:
            step: 样本所属的时间步
            batch_y: 真实标签 [B, L, C]
        """
        label = batch_y.detach().clone()
        
        if step in self.replay_buffer:
            self.replay_buffer[step]['label'] = label
        else:
            # 样本可能因容量限制被移除，暂存标签等待样本补回
            self.pending_labels[step] = label
    
    def advance_time(self, current_step: int):
        """
        根据当前时间推进，将满足延迟条件的样本移动到 Ready Queue
        """
        ready_steps = [
            step for step, data in self.replay_buffer.items()
            if data['label'] is not None and current_step >= data['available_step']
        ]
        
        ready_steps.sort()
        
        for step in ready_steps:
            data = self.replay_buffer.pop(step)
            batch_x = data['batch_x']
            batch_y = data['label']
            enc_out = data['enc_out']
            
            # 尺寸对齐（处理末尾批次）
            if len(batch_y) < len(batch_x):
                new_batch_size = len(batch_y)
                batch_x = batch_x[:new_batch_size]
                if enc_out is not None:
                    old_batch_size = len(batch_x)
                    C = batch_x.shape[2] if len(batch_x.shape) == 3 else 1
                    if enc_out.shape[0] == old_batch_size * C:
                        enc_out = enc_out[:new_batch_size * C]
                    elif enc_out.shape[0] == old_batch_size:
                        enc_out = enc_out[:new_batch_size]
            
            if len(batch_x) < len(batch_y):
                batch_y = batch_y[:len(batch_x)]
            
            self.ready_queue.append((step, batch_x, batch_y, enc_out))
            
            if len(self.ready_queue) > self.max_buffer_size:
                self.ready_queue.pop(0)
    
    def should_update(
        self, 
        current_step: int, 
        is_anomaly: bool = False
    ) -> bool:
        """
        判断是否应该触发监督更新
        
        触发条件 (满足任一即可):
        1. Ready Queue 达到批量大小
        2. 检测到异常 (紧急更新)
        3. 距离上次更新超过最大等待步数
        
        Args:
            current_step: 当前时间步
            is_anomaly: 是否检测到异常
        
        Returns:
            should_update: 是否应该更新
        """
        self.current_step = current_step
        
        # 条件 1: 批量大小达到阈值
        if len(self.ready_queue) >= self.batch_size:
            return True
        
        # 条件 2: 检测到异常且有可用样本
        if is_anomaly and len(self.ready_queue) >= self.min_ready_for_anomaly:
            self.anomaly_trigger_count += 1
            self.last_is_anomaly = True
            self.last_update_reason = "anomaly"
            return True
        
        # 条件 3: 超时强制更新
        steps_since_update = current_step - self.last_update_step
        if steps_since_update >= self.max_wait_steps and len(self.ready_queue) > 0:
            self.last_is_anomaly = False
            self.last_update_reason = "timeout"
            return True
        
        self.last_is_anomaly = False
        self.last_update_reason = ""
        return False
    
    def get_batch(self) -> Optional[Tuple[List, np.ndarray]]:
        """
        从 Ready Queue 获取一批样本进行更新
        
        Returns:
            batch_data: [(step, batch_x, batch_y, enc_out), ...] 或 None
            weights: 时间衰减权重 [N] 或 None
        """
        if len(self.ready_queue) == 0:
            return None, None
        
        # 取出全部或部分样本（最多 batch_size 个）
        # v5.0 Fix: 如果是最后一批（且数量较少），允许取出所有样本
        num_samples = min(len(self.ready_queue), self.batch_size * 2)  # 允许稍微多一点
        batch_data = self.ready_queue[:num_samples]
        self.ready_queue = self.ready_queue[num_samples:]
        
        # 计算时间衰减权重 + 温度 + 异常加权
        # weight_i = softmax(-λ * Δt / T)
        weights = []
        for step, _, _, _ in batch_data:
            time_diff = self.current_step - step
            weight = np.exp(-self.weight_decay * time_diff / self.weight_temperature)
            weights.append(weight)
        
        weights = np.array(weights, dtype=np.float64)
        if self.last_is_anomaly and self.anomaly_boost != 1.0:
            weights = weights * self.anomaly_boost
        
        # 归一化权重 (避免除零错误)
        weights_sum = weights.sum()
        if weights_sum > 0:
            weights = weights / weights_sum
        else:
            weights = np.ones_like(weights) / len(weights)
        
        # 更新统计
        self.last_update_step = self.current_step
        self.total_supervised_updates += 1
        self.total_samples_used += len(batch_data)
        
        return batch_data, weights
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取缓冲区统计信息
        
        Returns:
            stats: 统计信息字典
        """
        return {
            'replay_buffer_size': len(self.replay_buffer),
            'ready_queue_size': len(self.ready_queue),
            'total_supervised_updates': self.total_supervised_updates,
            'total_samples_used': self.total_samples_used,
            'avg_samples_per_update': (
                self.total_samples_used / self.total_supervised_updates
                if self.total_supervised_updates > 0 else 0.0
            ),
            'steps_since_last_update': self.current_step - self.last_update_step,
            'anomaly_trigger_count': self.anomaly_trigger_count,
            'last_update_reason': self.last_update_reason,
            'last_is_anomaly': self.last_is_anomaly
        }
    
    def reset(self):
        """重置缓冲区"""
        self.replay_buffer.clear()
        self.ready_queue.clear()
        self.pending_labels.clear()
        self.current_step = 0
        self.last_update_step = 0
        self.total_supervised_updates = 0
        self.total_samples_used = 0


def visualize_memory_keys_similarity(
    model,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    v6.0: 可视化 Memory Keys 的相似度矩阵热力图
    
    用于监控正交性约束是否生效，检测 Mode Collapse
    
    Args:
        model: M-Stream 模型实例
        save_path: 保存路径（可选）
        show: 是否显示图像
    """
    if not hasattr(model, 'memory'):
        print("⚠️  Model does not have a memory module.")
        return
    
    memory = model.memory
    if not hasattr(memory, 'memory_keys'):
        print("⚠️  Memory module does not have memory_keys parameter.")
        return
    
    # 计算归一化后的相似度矩阵
    with torch.no_grad():
        keys = memory.memory_keys  # [M, D]
        keys_norm = torch.nn.functional.normalize(keys, p=2, dim=1)
        similarity_matrix = torch.matmul(keys_norm, keys_norm.t())  # [M, M]
        similarity_np = similarity_matrix.cpu().numpy()
    
    num_prototypes = similarity_np.shape[0]
    
    # 计算统计信息
    # 对角线元素应该接近 1，非对角线应该接近 0
    diag_mean = np.diag(similarity_np).mean()
    off_diag_mask = ~np.eye(num_prototypes, dtype=bool)
    off_diag_mean = np.abs(similarity_np[off_diag_mask]).mean()
    off_diag_max = np.abs(similarity_np[off_diag_mask]).max()
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarity_np, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 添加色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    # 设置标题和标签
    ax.set_title(
        f'Memory Keys Similarity Matrix (v6.0)\n'
        f'Diag: {diag_mean:.3f} | Off-Diag Mean: {off_diag_mean:.3f} | Off-Diag Max: {off_diag_max:.3f}',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Memory Prototype Index', fontsize=12)
    ax.set_ylabel('Memory Prototype Index', fontsize=12)
    
    # 添加网格
    ax.set_xticks(np.arange(0, num_prototypes, max(1, num_prototypes // 10)))
    ax.set_yticks(np.arange(0, num_prototypes, max(1, num_prototypes // 10)))
    ax.grid(False)
    
    # 添加文本注释（仅在原型数量较少时）
    if num_prototypes <= 16:
        for i in range(num_prototypes):
            for j in range(num_prototypes):
                text = ax.text(j, i, f'{similarity_np[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Memory similarity matrix saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # 返回统计信息
    return {
        'diagonal_mean': float(diag_mean),
        'off_diagonal_mean': float(off_diag_mean),
        'off_diagonal_max': float(off_diag_max),
        'orthogonality_score': 1.0 - float(off_diag_mean)  # 越接近 1 越好
    }
