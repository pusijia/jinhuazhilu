"""在线学习和适应模块

实现了实时学习和环境适应功能，包括：
- 在线权重更新机制
- 短期记忆和遗忘机制
- 动态阈值调整
- 适应性学习控制
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from .snn import SpikingNeuron
from .synapse import Synapse
from .enhanced_stdp import EnhancedSTDP

class OnlineLearning:
    def __init__(self,
                 neurons: List[SpikingNeuron],
                 synapses: List[Synapse],
                 stdp: EnhancedSTDP,
                 memory_decay: float = 0.99,
                 threshold_adapt_rate: float = 0.01):
        """初始化在线学习系统
        
        Args:
            neurons: 神经元列表
            synapses: 突触列表
            stdp: 增强型STDP实例
            memory_decay: 记忆衰减率
            threshold_adapt_rate: 阈值适应速率
        """
        self.neurons = neurons
        self.synapses = synapses
        self.stdp = stdp
        self.memory_decay = memory_decay
        self.threshold_adapt_rate = threshold_adapt_rate
        
        # 初始化短期记忆
        self.short_term_memory = {
            'spike_patterns': [],
            'importance_scores': []
        }
        
        # 性能监控
        self.performance_history = {
            'adaptation_rate': [],
            'memory_usage': [],
            'learning_efficiency': []
        }
        
    def update_short_term_memory(self, spike_pattern: torch.Tensor, importance: float):
        """更新短期记忆
        
        Args:
            spike_pattern: 当前的脉冲模式
            importance: 重要性得分
        """
        # 应用记忆衰减
        for i in range(len(self.short_term_memory['importance_scores'])):
            self.short_term_memory['importance_scores'][i] *= self.memory_decay
            
        # 添加新的记忆
        self.short_term_memory['spike_patterns'].append(spike_pattern)
        self.short_term_memory['importance_scores'].append(importance)
        
        # 移除不重要的记忆
        self._cleanup_memory()
        
    def adapt_thresholds(self, recent_activity: List[float]):
        """动态调整神经元阈值
        
        Args:
            recent_activity: 最近的神经元活动水平
        """
        for neuron in self.neurons:
            avg_activity = np.mean(recent_activity)
            target_activity = 0.2  # 目标活动水平
            
            # 根据活动水平调整阈值
            if avg_activity > target_activity:
                neuron.threshold.data *= (1 + self.threshold_adapt_rate)
            else:
                neuron.threshold.data *= (1 - self.threshold_adapt_rate)
                
            # 限制阈值范围
            neuron.threshold.data.clamp_(0.1, 2.0)
            
    def online_train_step(self, input_spikes: torch.Tensor) -> Dict[str, float]:
        """执行单步在线学习
        
        Args:
            input_spikes: 输入脉冲序列
            
        Returns:
            metrics: 学习性能指标
        """
        # 前向传播
        current = input_spikes
        layer_activities = []
        
        for neuron, synapse in zip(self.neurons, self.synapses):
            current = synapse.forward(current)
            spike, _ = neuron.forward(current)
            layer_activities.append(torch.mean(spike.float()).item())
            current = spike
            
        # 计算当前模式的重要性
        importance = self._compute_pattern_importance(layer_activities)
        
        # 更新短期记忆
        self.update_short_term_memory(input_spikes, importance)
        
        # 适应性阈值调整
        self.adapt_thresholds(layer_activities)
        
        # 应用STDP学习
        self.stdp.update_synapses(self.synapses, 0)
        
        # 计算性能指标
        metrics = {
            'adaptation_rate': np.mean(layer_activities),
            'memory_usage': len(self.short_term_memory['spike_patterns']),
            'learning_efficiency': importance
        }
        
        # 记录性能历史
        for key, value in metrics.items():
            self.performance_history[key].append(value)
            
        return metrics
        
    def _compute_pattern_importance(self, activities: List[float]) -> float:
        """计算输入模式的重要性得分
        
        Args:
            activities: 各层的活动水平
            
        Returns:
            importance: 重要性得分
        """
        # 计算活动水平统计特征
        activity_mean = np.mean(activities)
        activity_std = np.std(activities)
        activity_max = np.max(activities)
        
        # 计算时间相关性
        temporal_correlation = 0.0
        if len(activities) > 1:
            temporal_correlation = np.corrcoef(activities[:-1], activities[1:])[0,1]
        
        # 综合评估重要性
        importance = (
            0.4 * activity_mean +  # 活动强度
            0.3 * activity_std +   # 活动变化
            0.2 * activity_max +   # 峰值响应
            0.1 * (1 + temporal_correlation)  # 时间相关性
        )
        
        return importance
        
    def _cleanup_memory(self):
        """清理短期记忆中不重要的内容"""
        max_capacity = 100  # 最大记忆容量
        min_capacity = 30   # 最小保留容量
        current_size = len(self.short_term_memory['importance_scores'])
        
        if current_size > max_capacity:
            # 计算遗忘阈值
            importance_threshold = np.percentile(
                self.short_term_memory['importance_scores'], 
                (1 - min_capacity/current_size) * 100
            )
            
            # 根据重要性和时间进行遗忘
            keep_indices = []
            for i, (score, pattern) in enumerate(zip(
                self.short_term_memory['importance_scores'],
                self.short_term_memory['spike_patterns']
            )):
                # 综合考虑重要性和时间衰减
                time_factor = np.exp(-0.1 * (current_size - i))  # 时间衰减
                effective_score = score * time_factor
                
                if effective_score > importance_threshold:
                    keep_indices.append(i)
                    
            # 更新记忆
            self.short_term_memory['spike_patterns'] = \
                [self.short_term_memory['spike_patterns'][i] for i in keep_indices]
            self.short_term_memory['importance_scores'] = \
                [self.short_term_memory['importance_scores'][i] for i in keep_indices]
                
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息
        
        Returns:
            stats: 性能统计指标
        """
        return {
            'avg_adaptation_rate': np.mean(self.performance_history['adaptation_rate']),
            'avg_memory_usage': np.mean(self.performance_history['memory_usage']),
            'avg_learning_efficiency': np.mean(self.performance_history['learning_efficiency'])
        }