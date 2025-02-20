"""卷积脉冲神经网络实现

实现了基于STDP的卷积SNN模型，包括：
- 卷积层
- 池化层
- 脉冲编码层
- 时空特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..language.synapse import Synapse

class ConvSNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 learning_rate: float = 0.01):
        """初始化卷积SNN层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            learning_rate: STDP学习率
        """
        super().__init__()
        
        # 卷积层参数
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 突触连接
        self.synapse = Synapse(
            in_features=in_channels * kernel_size * kernel_size,
            out_features=out_channels,
            learning_rate=learning_rate
        )
        
        # 神经元状态
        self.membrane_potential = None
        self.firing_threshold = 1.0
        self.reset_potential = 0.0
        
    def reset_state(self, batch_size: int, height: int, width: int,
                    device: torch.device):
        """重置神经元状态"""
        self.membrane_potential = torch.zeros(
            batch_size, self.synapse.weight.shape[0],
            height, width, device=device
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, in_channels, height, width]
            
        Returns:
            spikes: 输出脉冲 [batch_size, out_channels, out_height, out_width]
        """
        batch_size, _, height, width = x.shape
        
        if self.membrane_potential is None:
            self.reset_state(batch_size, height, width, x.device)
            
        # 提取局部特征
        unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        
        # 计算突触后电流
        batch_size = unfold.size(0)
        n_locations = unfold.size(2)
        unfold = unfold.transpose(1, 2).contiguous()
        unfold = unfold.view(batch_size * n_locations, -1)
        post_current = self.synapse.forward(unfold)
        post_current = post_current.view(batch_size, n_locations, -1).transpose(1, 2)
        
        # 重塑post_current以匹配membrane_potential的维度
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        post_current = post_current.view(batch_size, -1, out_height, out_width)
        
        # 更新膜电位
        self.membrane_potential = self.membrane_potential + post_current
        
        # 生成脉冲
        spikes = (self.membrane_potential >= self.firing_threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spikes) + \
                                self.reset_potential * spikes
                                
        return spikes
    
    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """更新突触权重
        
        Args:
            pre_spikes: 突触前脉冲 [batch_size, in_channels, height, width]
            post_spikes: 突触后脉冲 [batch_size, out_channels, out_height, out_width]
        """
        # 展开特征图
        pre_unfold = F.unfold(
            pre_spikes,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        ).transpose(1, 2)
        
        post_unfold = post_spikes.view(post_spikes.shape[0], -1, 1)
        
        # 更新突触权重
        self.synapse.update(pre_unfold, post_unfold)