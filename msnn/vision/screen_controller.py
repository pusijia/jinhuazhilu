"""屏幕控制器模块

实现了屏幕内容捕获和分析功能，包括：
- 屏幕截图
- 图像预处理
- 变化检测
- 区域分析
"""

import numpy as np
import torch
import pyautogui
from PIL import Image
from typing import Optional, Tuple

class ScreenController:
    def __init__(self,
                 capture_interval: float = 0.1,
                 memory_size: int = 10,
                 screen_width: int = 2560,
                 screen_height: int = 1600):
        """初始化屏幕控制器
        
        Args:
            capture_interval: 捕获间隔(秒)
            memory_size: 历史记忆大小
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
        """
        self.capture_interval = capture_interval
        self.memory_size = memory_size
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 历史记忆
        self.screen_memory = []
        
        # 设置PyAutoGUI的截图参数
        pyautogui.FAILSAFE = True
        
        # 初始化GPU内存优化配置
        if torch.cuda.is_available():
            # 根据分辨率动态调整GPU内存分配
            torch.cuda.empty_cache()
            self.device = torch.device('cuda')
            # 对于高分辨率，使用半精度浮点数以节省显存
            if screen_width * screen_height > 1920 * 1080:
                self.use_half_precision = True
            else:
                self.use_half_precision = False
        else:
            self.device = torch.device('cpu')
            self.use_half_precision = False
    
    def capture(self) -> torch.Tensor:
        """捕获屏幕内容
        
        Returns:
            screen_tensor: 屏幕张量 [1, 3, height, width]
        """
        # 截取屏幕
        screenshot = pyautogui.screenshot()
        
        # 转换为张量
        screen_array = np.array(screenshot)
        screen_tensor = torch.from_numpy(screen_array).float()
        screen_tensor = screen_tensor.permute(2, 0, 1).unsqueeze(0)
        screen_tensor = screen_tensor / 255.0
        
        # 移动到指定设备
        screen_tensor = screen_tensor.to(self.device)
        
        # 使用半精度（如果启用）
        if self.use_half_precision:
            screen_tensor = screen_tensor.half()
        
        # 更新历史记忆
        self.screen_memory.append(screen_tensor)
        if len(self.screen_memory) > self.memory_size:
            self.screen_memory.pop(0)
            
        return screen_tensor
    
    def detect_significant_change(self,
                                current_screen: torch.Tensor,
                                threshold: float = 0.5) -> bool:
        """检测显著变化
        
        Args:
            current_screen: 当前屏幕张量
            threshold: 变化阈值
            
        Returns:
            has_change: 是否有显著变化
        """
        if len(self.screen_memory) < 2:
            return False
            
        # 计算与上一帧的差异
        prev_screen = self.screen_memory[-2]
        diff = torch.abs(current_screen - prev_screen)
        change_ratio = torch.mean(diff)
        
        return change_ratio > threshold
    
    def analyze_region(self,
                      region: Tuple[int, int, int, int]) -> torch.Tensor:
        """分析特定区域
        
        Args:
            region: 区域坐标(x, y, width, height)
            
        Returns:
            region_tensor: 区域张量
        """
        x, y, width, height = region
        screenshot = pyautogui.screenshot(region=region)
        
        region_array = np.array(screenshot)
        region_tensor = torch.from_numpy(region_array).float()
        region_tensor = region_tensor.permute(2, 0, 1).unsqueeze(0)
        region_tensor = region_tensor / 255.0
        
        return region_tensor
    
    def reset_memory(self):
        """重置历史记忆"""
        self.screen_memory.clear()