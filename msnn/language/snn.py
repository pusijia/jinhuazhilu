"""脉冲神经网络基础模型

实现了基于PyTorch的脉冲神经网络核心组件，包括：
- 脉冲神经元模型
- 突触连接机制
- 时序信息处理
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class SpikingNeuron(nn.Module):
    def __init__(self, 
                 threshold: float = 1.0,
                 tau_mem: float = 10.0,
                 tau_syn: float = 5.0):
        """初始化脉冲神经元

        Args:
            threshold: 发放阈值
            tau_mem: 膜电位时间常数
            tau_syn: 突触电位时间常数
        """
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.tau_mem = nn.Parameter(torch.tensor(tau_mem))
        self.tau_syn = nn.Parameter(torch.tensor(tau_syn))
        
        # 初始化状态变量
        self.membrane_potential = None
        self.synaptic_current = None
        
    def reset_state(self, batch_size: int, device: torch.device):
        """重置神经元状态"""
        self.membrane_potential = torch.zeros(batch_size, device=device)
        self.synaptic_current = torch.zeros(batch_size, device=device)
        
    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            input_current: 输入电流 [batch_size]
            
        Returns:
            spike_out: 输出脉冲
            membrane_potential: 膜电位
        """
        if self.membrane_potential is None:
            self.reset_state(input_current.shape[0], input_current.device)
            
        # 更新突触电流
        self.synaptic_current = self.synaptic_current + \
            (-self.synaptic_current / self.tau_syn + input_current)
            
        # 更新膜电位
        self.membrane_potential = self.membrane_potential + \
            (-self.membrane_potential / self.tau_mem + self.synaptic_current)
            
        # 生成脉冲
        spike_out = (self.membrane_potential >= self.threshold).float()
        
        # 重置膜电位
        self.membrane_potential = self.membrane_potential * (1 - spike_out)
        
        return spike_out, self.membrane_potential