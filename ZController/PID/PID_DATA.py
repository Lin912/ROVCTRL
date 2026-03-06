# PID_DATA.py
import numpy as np
import time

def generate_test_data():
    """生成测试用的传感器数据"""
    t = 0
    dt = 1.00/100.0  # 生成频率100Hz
       
    with open('rov_sensor_data.txt', 'w') as f:
        # 写入表头
        header = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                 'u', 'v', 'w', 'p', 'q', 'r', 'depth', 'altitude']
        f.write(','.join(header) + '\n')
        
        while True:
            timestamp = time.time()
            
            # 模拟缓慢漂移后回到原点
            x = 0.1 * np.sin(0.1 * t)
            y = 0.1 * np.cos(0.1 * t)
            z = -5.0 + 0.2 * np.sin(0.2 * t)
            
            roll = 0.05 * np.sin(0.5 * t)
            pitch = 0.05 * np.cos(0.5 * t)
            yaw = 0.1 * np.sin(0.3 * t)
            
            # 速度（导数）
            u = 0.01 * np.cos(0.1 * t)
            v = -0.01 * np.sin(0.1 * t)
            w = 0.04 * np.cos(0.2 * t)
            
            p = 0.025 * np.cos(0.5 * t)
            q = -0.025 * np.sin(0.5 * t)
            r = 0.03 * np.cos(0.3 * t)
            
            depth = -z  # 深度（正值）
            altitude = 10.0 + z  # 假设水深10米
            
            # 写入数据
            data = [timestamp, x, y, z, roll, pitch, yaw, 
                   u, v, w, p, q, r, depth, altitude]
            f.write(','.join([str(val) for val in data]) + '\n')
            f.flush()  # 立即写入文件
            
            t += dt
            time.sleep(dt)

if __name__ == "__main__":
    print("开始生成传感器测试数据...")
    try:
        generate_test_data()
    except KeyboardInterrupt:
        print("\n数据生成已停止")