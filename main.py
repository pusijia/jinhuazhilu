"""MSNN系统主程序

实现了系统的初始化和启动流程，包括：
- 核心模块初始化
- 屏幕控制器启动
- 自主学习循环
"""

import torch
import numpy as np
from msnn.language.snn import SpikingNeuron
from msnn.vision.conv_snn import ConvSNN
from msnn.logic.graph_snn import GraphSNN
from msnn.vision.screen_controller import ScreenController
from msnn.language.dx_interactive import DXController
from msnn.language.trainer import SNNTrainer

def init_language_module(input_size: int = 1024,
                      hidden_size: int = 512,
                      output_size: int = 256) -> SpikingNeuron:
    """初始化语言认知模块"""
    return SpikingNeuron(
        threshold=1.0,
        tau_mem=10.0,
        tau_syn=5.0
    )

def init_vision_module(in_channels: int = 3,
                     out_channels: int = 64) -> ConvSNN:
    """初始化视觉感知模块"""
    return ConvSNN(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1
    )

def init_logic_module(in_features: int = 256,
                    hidden_features: int = 128,
                    out_features: int = 64) -> GraphSNN:
    """初始化逻辑推理模块"""
    return GraphSNN(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features
    )

def main():
    # 初始化设备并配置GPU优化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # 启用自动混合精度训练
        scaler = torch.cuda.amp.GradScaler()
        # 设置GPU内存分配器
        torch.cuda.set_per_process_memory_fraction(0.8)  # 预留20%显存
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动优化
        print(f"使用GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("使用CPU设备")
    
    # 初始化核心模块
    language_module = init_language_module().to(device)
    vision_module = init_vision_module().to(device)
    logic_module = init_logic_module().to(device)
    
    # 初始化训练器
    trainer = SNNTrainer(
        neurons=[language_module],
        synapses=[vision_module.synapse],
        learning_rate=0.01
    )
    
    # 初始化控制器
    screen_controller = ScreenController()
    snn_controller = DXController(trainer=trainer, update_interval=10)
    
    print("MSNN系统初始化完成")
    print("启动DirectX可视化界面...")
    
    try:
        while True:
            # 定期清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 监控GPU使用情况
                if (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) > 0.9:
                    print("警告：GPU显存使用率超过90%")
            
            # 捕获屏幕内容
            screen_data = screen_controller.capture()
            
            # 检测显著变化
            if screen_controller.detect_significant_change(screen_data, threshold=0.5):
                # 使用自动混合精度进行视觉处理
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    vision_spikes = vision_module(screen_data)
                    
                    # 逻辑分析
                    graph_data = logic_module.forward(
                        vision_spikes.view(1, -1, vision_spikes.size(-1)),  # 调整为[batch_size, num_nodes, features]
                        torch.ones(1, vision_spikes.size(1), vision_spikes.size(1)).to(device)  # 生成对应的邻接矩阵
                    )
                    
                    # 语言理解
                    language_spikes, _ = language_module(graph_data.mean(dim=1))
                
                # 更新系统状态
                snn_controller.record_activity(
                    language_spikes,
                    language_module.membrane_potential,
                    vision_module.synapse
                )
                
                print("检测到显著变化，已完成一轮学习")
                if torch.cuda.is_available():
                    print(f"当前GPU显存使用: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                
    except KeyboardInterrupt:
        print("\n系统正常退出")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()