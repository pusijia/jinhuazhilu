"""脉冲神经网络训练框架

实现了基于STDP的训练机制，包括：
- 训练循环控制
- 学习参数管理
- 性能指标统计
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from .snn import SpikingNeuron
from .synapse import Synapse

class SNNTrainer:
    def __init__(self,
                 neurons: List[SpikingNeuron],
                 synapses: List[Synapse],
                 learning_rate: float = 0.01):
        """初始化训练器

        Args:
            neurons: 神经元列表
            synapses: 突触列表
            learning_rate: 学习率
        """
        self.neurons = neurons
        self.synapses = synapses
        self.learning_rate = learning_rate
        
        # 训练统计
        self.spike_counts = []
        self.weight_changes = []
        
    def reset_state(self):
        """重置网络状态"""
        batch_size = 1
        device = next(self.neurons[0].parameters()).device
        
        for neuron in self.neurons:
            neuron.reset_state(batch_size, device)
        for synapse in self.synapses:
            synapse.reset_trace(batch_size, device)
            
    def train_step(self, input_spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 前向传播
        current = input_spikes
        layer_spikes = []
        membrane_potentials = []
        
        for i, (neuron, synapse) in enumerate(zip(self.neurons, self.synapses)):
            current = synapse.forward(current)
            spike, membrane = neuron.forward(current)
            layer_spikes.append(spike)
            membrane_potentials.append(membrane)
            current = spike
            
            # 及时清理中间状态
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # STDP学习
        weight_changes = []
        for i in range(len(self.synapses)):
            pre_spikes = input_spikes if i == 0 else layer_spikes[i-1]
            post_spikes = layer_spikes[i]
            
            initial_weight = self.synapses[i].weight.data.clone()
            self.synapses[i].update(pre_spikes, post_spikes)
            weight_changes.append(
                torch.mean(torch.abs(self.synapses[i].weight.data - initial_weight))
            )
            
            # 及时释放不需要的张量
            del initial_weight
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # 计算统计指标
        spike_rate = torch.mean(torch.stack([s.float().mean() for s in layer_spikes]))
        weight_change = torch.mean(torch.stack(weight_changes))
        
        self.spike_counts.append(spike_rate.item())
        self.weight_changes.append(weight_change.item())
        
        # 返回更多的状态信息
        return {
            'spike_rate': spike_rate,
            'weight_change': weight_change,
            'membrane_potentials': membrane_potentials,
            'layer_spikes': layer_spikes
        }
        
    def train(self, 
              input_data: torch.Tensor,
              num_epochs: int,
              batch_size: int = 1) -> Dict[str, List[float]]:
        """训练网络
        
        Args:
            input_data: 输入数据 [num_samples, num_inputs]
            num_epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            history: 训练历史记录
        """
        num_samples = input_data.shape[0]
        history = {'spike_rates': [], 'weight_changes': []}
        
        for epoch in range(num_epochs):
            self.reset_state()
            epoch_metrics = []
            
            # 按批次训练
            for i in range(0, num_samples, batch_size):
                batch_input = input_data[i:i+batch_size]
                metrics = self.train_step(batch_input)
                epoch_metrics.append(metrics)
            
            # 记录每轮统计
            avg_spike_rate = torch.mean(torch.tensor(
                [m['spike_rate'] for m in epoch_metrics]
            ))
            avg_weight_change = torch.mean(torch.tensor(
                [m['weight_change'] for m in epoch_metrics]
            ))
            
            history['spike_rates'].append(avg_spike_rate.item())
            history['weight_changes'].append(avg_weight_change.item())
            
        return history