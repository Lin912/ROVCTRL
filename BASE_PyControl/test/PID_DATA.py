# 创建示例传感器数据生成器
import numpy as np
import time

def generate_test_data():
    """生成测试用的传感器数据"""
    t = 0
    while True:
        # 模拟缓慢漂移后回到原点
        x = 0.1 * np.sin(0.1 * t)
        y = 0.1 * np.cos(0.1 * t)
        z = -5.0 + 0.2 * np.sin(0.2 * t)
        
        roll = 0.05 * np.sin(0.5 * t)
        pitch = 0.05 * np.cos(0.5 * t)
        yaw = 0.1 * np.sin(0.3 * t)
        
        # 速度