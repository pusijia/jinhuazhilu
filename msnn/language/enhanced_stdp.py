"""增强型STDP学习规则模块

实现了改进的脉冲时序依赖可塑性(STDP)学习规则，包括：
- 多层网络的权重更新机制
- 动态学习率调整
- 权重正则化
- 稳定性控制
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from .synapse import Synapse

class EnhancedSTDP:
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0001,
                 stability_factor: float = 0.1):
        """初始化增强型STDP
        
        Args:
            learning_rate: 基础学习率
            momentum: 动量因子
            weight_decay: 权重衰减系数
            stability_factor: 稳定性因子
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.stability_factor = stability_factor
        
        # 动量状态
        self.velocity = None
        
    def initialize_state(self, synapses: List[Synapse]):
        """初始化优化器状态
        
        Args:
            synapses: 需要优化的突触列表
        """
        self.velocity = [torch.zeros_like(synapse.weight) for synapse in synapses]
        
    def compute_weight_updates(self,
                              pre_spikes: torch.Tensor,
                              post_spikes: torch.Tensor,
                              current_weight: torch.Tensor) -> torch.Tensor:
        """计算权重更新
        
        Args:
            pre_spikes: 前神经元脉冲
            post_spikes: 后神经元脉冲
            current_weight: 当前权重
            
        Returns:
            weight_update: 权重更新量
        """
        # 计算STDP时间窗
        pre_times = torch.where(pre_spikes)[1]
        post_times = torch.where(post_spikes)[1]
        
        # 计算时间差
        delta_t = post_times.unsqueeze(1) - pre_times.unsqueeze(0)
        
        # STDP核函数
        positive_window = torch.exp(-torch.abs(delta_t) / 20.0) * (delta_t > 0)
        negative_window = -torch.exp(-torch.abs(delta_t) / 20.0) * (delta_t < 0)
        stdp_window = positive_window + negative_window
        
        # 计算权重更新
        weight_update = torch.zeros_like(current_weight)
        weight_update[pre_spikes.bool(), post_spikes.bool()] = stdp_window
        
        # 添加权重正则化
        weight_update = weight_update - self.weight_decay * current_weight
        
        return weight_update
        
    def update_synapses(self, synapses: List[Synapse], layer_idx: int):
        """更新突触权重
        
        Args:
            synapses: 突触列表
            layer_idx: 当前层索引
        """
        if self.velocity is None:
            self.initialize_state(synapses)
            
        # 计算层次化学习率
        layer_lr = self.learning_rate * (0.8 ** layer_idx)  # 深层学习率递减
            
        for i, synapse in enumerate(synapses):
            # 计算权重更新
            weight_update = self.compute_weight_updates(
                synapse.pre_spike_trace,
                synapse.post_spike_trace,
                synapse.weight
            )
            
            # 计算权重稳定性因子
            weight_stability = torch.exp(-torch.abs(synapse.weight - 0.5) * 2)
            
            # 应用动量和稳定性控制
            self.velocity[i] = self.momentum * self.velocity[i] + \
                layer_lr * weight_update * weight_stability
                
            # 更新权重
            synapse.weight.data += self.velocity[i]
            
            # 应用权重限制和软约束
            weight_center = 0.5 + 0.4 * torch.tanh(synapse.weight - 0.5)
            synapse.weight.data = 0.9 * synapse.weight.data + 0.1 * weight_center
            synapse.weight.data.clamp_(0, 1)
            
    def adjust_learning_rate(self, weight_changes: List[float]):
        """动态调整学习率
        
        Args:
            weight_changes: 权重变化历史
        """
        if len(weight_changes) > 1:
            # 根据权重变化趋势调整学习率
            recent_changes = weight_changes[-10:]
            change_std = np.std(recent_changes)
            
            if change_std > self.stability_factor:
                self.learning_rate *= 0.9  # 降低学习率以提高稳定性
            elif change_std < self.stability_factor * 0.1:
                self.learning_rate *= 1.1  # 提高学习率以加速学习
                
            # 限制学习率范围
            self.learning_rate = max(0.0001, min(0.1, self.learning_rate))