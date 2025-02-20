"""突触连接机制实现

实现了基于STDP(Spike-Timing-Dependent Plasticity)的突触可塑性模型，包括：
- 突触权重初始化
- STDP学习规则
- 权重归一化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class Synapse(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 learning_rate: float = 0.01,
                 w_min: float = 0.0,
                 w_max: float = 1.0):
        """初始化突触连接

        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            learning_rate: STDP学习率
            w_min: 最小权重
            w_max: 最大权重
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.w_min = w_min
        self.w_max = w_max

        # 初始化突触权重
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features).uniform_(w_min, w_max)
        )
        
        # 记录神经元发放历史
        self.pre_spike_trace = None
        self.post_spike_trace = None
        
    def reset_trace(self, batch_size: int, device: torch.device):
        """重置突触痕迹"""
        self.pre_spike_trace = torch.zeros(
            batch_size, self.weight.shape[1], device=device
        )
        self.post_spike_trace = torch.zeros(
            batch_size, self.weight.shape[0], device=device
        )
        
    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            pre_spikes: 突触前神经元的脉冲 [batch_size, in_features]
            
        Returns:
            post_current: 突触后电流 [batch_size, out_features]
        """
        return torch.mm(pre_spikes, self.weight.t())
    
    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
               tau_pre: float = 20.0, tau_post: float = 20.0):
        """更新突触权重
        
        Args:
            pre_spikes: 突触前脉冲 [batch_size, in_features]
            post_spikes: 突触后脉冲 [batch_size, out_features]
            tau_pre: 突触前时间常数
            tau_post: 突触后时间常数
        """
        if self.pre_spike_trace is None:
            self.reset_trace(pre_spikes.shape[0], pre_spikes.device)
            
        # 更新突触痕迹
        decay_pre = torch.exp(torch.tensor(-1.0/tau_pre, device=pre_spikes.device))
        decay_post = torch.exp(torch.tensor(-1.0/tau_post, device=post_spikes.device))
        self.pre_spike_trace = self.pre_spike_trace * decay_pre + pre_spikes
        self.post_spike_trace = self.post_spike_trace * decay_post + post_spikes
        
        # 计算权重更新
        pre_contribution = torch.mm(post_spikes.t(), self.pre_spike_trace)
        post_contribution = torch.mm(self.post_spike_trace.t(), pre_spikes)
        dw = self.learning_rate * (pre_contribution - post_contribution)
        
        # 更新权重并限制范围
        self.weight.data += dw
        self.weight.data.clamp_(self.w_min, self.w_max)