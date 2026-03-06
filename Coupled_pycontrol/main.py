# main.py
import time
import numpy as np

# 从我们拆分好的文件中导入类
from SofinLoop import CoSimInterface
from controller import ROVControlSystem

def main():
    print("=" * 80)
    print("Tether Vehicle 联合仿真节点 (Linux MMF 模式)")
    print("=" * 80)
    
    # 1. 实例化内存接口 
    # (修改 use_dummy=False 即可真正连接到你的 ControlDirect_SharedMemory 文件)
    cosim_interface = CoSimInterface(use_dummy=True, filename="ControlDirect_SharedMemory")
    
    # 2. 实例化控制器，并注入内存接口
    cs = ROVControlSystem(cosim_interface=cosim_interface)
    cs.set_desired_position(0.0, 0.0, -5.0)
    cs.set_desired_attitude(0.0, 0.0, 0.0)
    
    print("\n等待 CFD (STAR-CCM+) / Tethra 轮询计算中...")
    
    last_display_time = -1.0
    display_interval = 0.1 
    
    print("\n" + "=" * 135)
    print(f"{'仿真时间':>8} | {'位置(m)':^25} | {'控制力(N/Nm)':^18} | {'目标 RPM (T0-T5)':^30} | {'实际 RPM (延迟爬坡)':^30}")
    print("=" * 135)
    
    try:
        while True:
            # 驱动控制器步进
            sim_time, current_state, tau, target_rpm, actual_rpm = cs.step()
            
            # 刷新 UI
            if sim_time - last_display_time >= (display_interval - 1e-6):
                pos_str = f"({current_state['x']:>5.2f}, {current_state['y']:>5.2f}, {current_state['z']:>5.2f})"
                force_mag = np.linalg.norm(tau[:3])
                moment_mag = np.linalg.norm(tau[3:])
                control_str = f"F:{force_mag:>5.1f} M:{moment_mag:>4.1f}"
                
                target_str = f"[{target_rpm[0]:>4.0f} {target_rpm[1]:>4.0f} {target_rpm[2]:>4.0f}]..."
                actual_str = f"[{actual_rpm[0]:>4.0f} {actual_rpm[1]:>4.0f} {actual_rpm[2]:>4.0f}]..."
                
                print(f"{sim_time:>7.3f}s | {pos_str:^25} | {control_str:^18} | {target_str:^30} | {actual_str:^30}")
                last_display_time = sim_time
                
    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
        
    finally:
        # 确保安全关闭内存映射文件，释放 Linux 系统资源
        cosim_interface.close()
        print("内存映射解除，程序安全退出。")

if __name__ == "__main__":
    main()