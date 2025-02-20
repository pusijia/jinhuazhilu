"""脉冲神经网络演示程序

展示了MSNN系统的基本功能，包括：
- 网络构建
- 数据准备
- 训练过程
- 可视化监控
"""

import torch
from msnn.language.snn import SpikingNeuron
from msnn.language.synapse import Synapse
from msnn.language.trainer import SNNTrainer
from msnn.language.interactive import SNNController

def main():
    # 构建简单的双层网络
    input_size = 10
    hidden_size = 5
    output_size = 2
    
    # 创建神经元
    hidden_neuron = SpikingNeuron(threshold=0.5, tau_mem=10.0, tau_syn=5.0)
    output_neuron = SpikingNeuron(threshold=0.5, tau_mem=10.0, tau_syn=5.0)
    neurons = [hidden_neuron, output_neuron]
    
    # 创建突触连接
    input_hidden = Synapse(input_size, hidden_size, learning_rate=0.01)
    hidden_output = Synapse(hidden_size, output_size, learning_rate=0.01)
    synapses = [input_hidden, hidden_output]
    
    # 创建训练器和控制器
    trainer = SNNTrainer(neurons, synapses, learning_rate=0.01)
    controller = SNNController(trainer, update_interval=10)
    
    # 准备随机输入数据
    num_samples = 100
    input_data = torch.rand(num_samples, input_size) > 0.8
    input_data = input_data.float()
    
    # 设置检查点保存路径
    checkpoint_path = "shuju/snn_checkpoint.pt"
    
    # 开始训练
    print("开始训练...")
    print("- 按Ctrl+C可以终止训练")
    print("- 关闭图形窗口可以结束程序")
    print(f"- 训练状态将自动保存到：{checkpoint_path}")
    
    try:
        history = controller.train_with_control(
            input_data=input_data,
            num_epochs=50,
            batch_size=1
        )
        print("训练完成！")
        # 保存最终状态
        controller.save_checkpoint(checkpoint_path)
        print(f"训练状态已保存到：{checkpoint_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 保存中断时的状态
        controller.save_checkpoint(checkpoint_path)
        print(f"中断状态已保存到：{checkpoint_path}")
        controller.terminate_training()

if __name__ == "__main__":
    main()