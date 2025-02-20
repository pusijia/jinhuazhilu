"""DirectX可视化界面控制器

实现了基于DirectX的神经网络可视化和交互功能，包括：
- 实时神经元活动显示
- 网络结构可视化
- 训练进度监控
- 性能指标图表
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from .trainer import SNNTrainer
from .controller import SNNController
import win32gui
import win32con
import win32api
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class DXController(SNNController):
    def __init__(self, trainer: SNNTrainer = None, update_interval: int = 10):
        """初始化DirectX控制器
        
        Args:
            trainer: SNN训练器实例
            update_interval: 可视化更新间隔(步数)
        """
        super().__init__(trainer, update_interval)
        self.dx_initialized = False
        self.window_handle = None
        self.gl_context = None
        self._init_dx_context()
        
    def _init_dx_context(self):
        try:
            # 注册窗口类
            hinst = win32gui.GetModuleHandle(None)
            wndclass = win32gui.WNDCLASS()
            wndclass.lpszClassName = 'SNNVisualization'
            wndclass.hInstance = hinst
            wndclass.hbrBackground = win32gui.GetStockObject(win32con.WHITE_BRUSH)
            wndclass.lpfnWndProc = win32gui.DefWindowProc  # 修改为正确的函数指针传递方式
            wndclass.style = win32con.CS_VREDRAW | win32con.CS_HREDRAW
            win32gui.RegisterClass(wndclass)
            
            # 创建窗口
            self.window_handle = win32gui.CreateWindow(
                wndclass.lpszClassName,
                'SNN可视化',
                win32con.WS_OVERLAPPEDWINDOW,
                win32con.CW_USEDEFAULT,
                win32con.CW_USEDEFAULT,
                800, 600,
                0, 0,
                hinst, None
            )
            
            # 初始化OpenGL上下文
            glutInit()
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
            glutCreateWindow('SNN可视化')
            
            # 基本OpenGL设置
            glEnable(GL_DEPTH_TEST)
            glClearColor(0.0, 0.0, 0.0, 0.0)
            
            self.dx_initialized = True
            win32gui.ShowWindow(self.window_handle, win32con.SW_SHOW)
            win32gui.UpdateWindow(self.window_handle)
            
            # 启动消息循环
            msg = win32gui.GetMessage(None, 0, 0)
            while msg[0]:
                win32gui.TranslateMessage(msg)
                win32gui.DispatchMessage(msg)
                msg = win32gui.GetMessage(None, 0, 0)
            
        except Exception as e:
            print(f"DirectX/OpenGL初始化失败: {e}")
            print("将回退到标准可视化模式")
            
    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        """窗口消息处理函数"""
        if msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        elif msg == win32con.WM_PAINT:
            self.visualize_activity()
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
            
    def visualize_activity(self, time_window: int = 100):
        if not self.dx_initialized:
            super().visualize_activity(time_window)
            return
            
        if len(self.spike_history) == 0:
            return
            
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # 设置3D视角
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, 800.0/600.0, 0.1, 100.0)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -6.0)
            
            # 绘制神经元网络
            self._draw_neurons()
            
            # 绘制性能指标
            self._draw_metrics()
            
            glutSwapBuffers()
            
        except Exception as e:
            print(f"渲染失败: {e}")
            
    def _draw_neurons(self):
        if not self.spike_history:
            return
            
        spikes = self.spike_history[-1]
        positions = np.linspace(-1, 1, len(spikes))
        
        glBegin(GL_POINTS)
        for i, (pos, spike) in enumerate(zip(positions, spikes)):
            if spike:
                glColor3f(1.0, 0.0, 0.0)  # 发放的神经元显示为红色
            else:
                glColor3f(0.5, 0.5, 0.5)  # 静息的神经元显示为灰色
            glVertex3f(pos, 0.0, 0.0)
        glEnd()
        
    def _draw_metrics(self):
        if not self.performance_metrics['spike_rate']:
            return
            
        # 绘制发放率曲线
        rates = np.array(self.performance_metrics['spike_rate'][-100:])
        positions = np.linspace(-1, 1, len(rates))
        
        glBegin(GL_LINE_STRIP)
        glColor3f(0.0, 1.0, 0.0)
        for pos, rate in zip(positions, rates):
            glVertex3f(pos, rate - 0.5, 0.0)
        glEnd()

    def train_with_control(self,
                          input_data: torch.Tensor,
                          num_epochs: int,
                          batch_size: int = 1) -> Dict[str, List[float]]:
        """带有DirectX可视化的训练过程
        
        Args:
            input_data: 输入数据
            num_epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            history: 训练历史记录
        """
        history = {'spike_rates': [], 'weight_changes': []}
        
        try:
            for epoch in range(num_epochs):
                if self.is_terminated:
                    break
                    
                self.trainer.reset_state()
                epoch_metrics = []
                
                for i in range(0, len(input_data), batch_size):
                    if self.is_terminated:
                        break
                        
                    batch = input_data[i:i+batch_size]
                    metrics = self.trainer.train_step(batch)
                    epoch_metrics.append(metrics)
                    
                    # 更新DirectX显示
                    if i % self.update_interval == 0:
                        self.visualize_activity()
                        
                # 记录每轮统计
                avg_spike_rate = torch.mean(torch.tensor(
                    [m['spike_rate'] for m in epoch_metrics]
                ))
                avg_weight_change = torch.mean(torch.tensor(
                    [m['weight_change'] for m in epoch_metrics]
                ))
                
                history['spike_rates'].append(avg_spike_rate.item())
                history['weight_changes'].append(avg_weight_change.item())
                
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            self.is_terminated = True
            
        return history
        
    def save_checkpoint(self, path: str):
        """保存训练检查点
        
        Args:
            path: 保存路径
        """
        if self.trainer is None:
            return
            
        checkpoint = {
            'neurons_state': [
                neuron.state_dict() for neuron in self.trainer.neurons
            ],
            'synapses_state': [
                synapse.state_dict() for synapse in self.trainer.synapses
            ],
            'performance_metrics': self.performance_metrics
        }
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """加载训练检查点
        
        Args:
            path: 检查点文件路径
        """
        if self.trainer is None:
            return
            
        checkpoint = torch.load(path)
        
        for neuron, state in zip(self.trainer.neurons, checkpoint['neurons_state']):
            neuron.load_state_dict(state)
            
        for synapse, state in zip(self.trainer.synapses, checkpoint['synapses_state']):
            synapse.load_state_dict(state)
            
        self.performance_metrics = checkpoint['performance_metrics']