"""SNN控制器模块

实现了脉冲神经网络的监控、可视化和控制功能，包括：
- 神经元活动实时监控
- 突触权重分布可视化
- 网络性能指标统计
- 紧急控制机制
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from .snn import SpikingNeuron
from .synapse import Synapse

class SNNController:
    def __init__(self, trainer=None, update_interval: int = 10):
        """初始化SNN控制器
        
        Args:
            trainer: SNN训练器实例
            update_interval: 可视化更新间隔(步数)
        """
        self.trainer = trainer
        self.update_interval = update_interval
        self.spike_history = []
        self.membrane_history = []
        self.weight_history = []
        self.performance_metrics = {
            'spike_rate': [],
            'response_latency': [],
            'energy_consumption': []
        }
        self.is_paused = False
        self.is_terminated = False
        
    def record_activity(self, 
                        spikes: torch.Tensor,
                        membrane_potential: torch.Tensor,
                        synapse: Synapse):
        """记录神经元活动
        
        Args:
            spikes: 神经元发放状态
            membrane_potential: 膜电位
            synapse: 突触对象
        """
        self.spike_history.append(spikes.detach().cpu().numpy())
        self.membrane_history.append(membrane_potential.detach().cpu().numpy())
        self.weight_history.append(synapse.weight.detach().cpu().numpy())
        
        # 计算性能指标
        self._update_metrics(spikes)
        
    def visualize_activity(self, time_window: int = 100):
        """可视化神经元活动
        
        Args:
            time_window: 显示的时间窗口大小
        """
        if len(self.spike_history) == 0:
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # 绘制脉冲发放图
        spikes = np.array(self.spike_history[-time_window:])
        ax1.imshow(spikes.T, aspect='auto', cmap='binary')
        ax1.set_title('神经元发放活动')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('神经元编号')
        
        # 绘制膜电位变化
        membrane = np.array(self.membrane_history[-time_window:])
        for i in range(membrane.shape[1]):
            ax2.plot(membrane[:, i], label=f'神经元 {i}')
        ax2.set_title('膜电位动态')
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('膜电位 (mV)')
        ax2.legend()
        
        # 绘制权重分布
        weights = self.weight_history[-1]
        ax3.hist(weights.flatten(), bins=50)
        ax3.set_title('突触权重分布')
        ax3.set_xlabel('权重值')
        ax3.set_ylabel('频数')
        
        plt.tight_layout()
        plt.show()
        
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息
        
        Returns:
            stats: 包含各项性能指标的字典
        """
        return {
            'avg_spike_rate': np.mean(self.performance_metrics['spike_rate']),
            'avg_latency': np.mean(self.performance_metrics['response_latency']),
            'energy_efficiency': np.mean(self.performance_metrics['energy_consumption'])
        }
        
    def _update_metrics(self, spikes: torch.Tensor):
        """更新性能指标
        
        Args:
            spikes: 当前时间步的脉冲状态
        """
        # 计算发放率
        spike_rate = torch.mean(spikes.float()).item()
        self.performance_metrics['spike_rate'].append(spike_rate)
        
        # 计算响应延迟
        if torch.any(spikes):
            latency = torch.where(spikes)[0][0].item()
            self.performance_metrics['response_latency'].append(latency)
            
        # 估算能耗(基于脉冲数量)
        energy = torch.sum(spikes.float()).item() * 0.1 # 假设每个脉冲消耗0.1单位能量
        self.performance_metrics['energy_consumption'].append(energy)
        
    def reset(self):
        """重置控制器状态"""
        self.spike_history.clear()
        self.membrane_history.clear()
        self.weight_history.clear()
        for key in self.performance_metrics:
            self.performance_metrics[key].clear()
            
    def emergency_reset(self, snn: SpikingNeuron, synapse: Synapse):
        """紧急重置网络
        
        Args:
            snn: 需要重置的神经元
            synapse: 需要重置的突触
        """
        # 重置神经元状态
        if snn.membrane_potential is not None:
            snn.reset_state(snn.membrane_potential.shape[0], 
                          snn.membrane_potential.device)
            
        # 重置突触痕迹
        if synapse.pre_spike_trace is not None:
            synapse.reset_trace(synapse.pre_spike_trace.shape[0],
                              synapse.pre_spike_trace.device)
            
        # 重置控制器状态
        self.reset()