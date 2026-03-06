# main.py
import time
import numpy as np

# 从我们拆分好的文件中导入类
from SofinLoop import CoSimInterface
from controller import ROVControlSystem

def main():
    print("StarCCM+ -> coupling -> ROVCTRL (BASED_ON Linux MMF)")
    
    # 1. 实例化内存接口 
    # (修改 use_dummy=False 即可真正连接到你的 ControlDirect_SharedMemory 文件)
    cosim_interface = CoSimInterface(
        use_dummy=False,
        filename="../HydroSimulation/ControlDirect_SharedMemory")
    
    # 2. 实例化控制器，并注入内存接口
    cs = ROVControlSystem(cosim_interface=cosim_interface)
    cs.set_desired_position(-0.5, 0.0, 0.0)
    cs.set_desired_attitude(0.0, 0.0, 0.0)
    
    print("\n Wating for StarCCM+ simulating...")
    
    last_display_time = -1.0    # 初始刷新时间
    display_interval = 1.0      # 屏幕刷新时间 
    
    print("\n" + "=" * 135)
    print(f"{'Simulating_Time':>8} | {'Position(m)':^25} | {'ControlForce(N/Nm)':^18} | {'Target RPM (T0-T5)':^30} | {'Real RPM ':^30}")
    print("=" * 135)
    
    try:
        while True:
            # 驱动控制器步进
            sim_time, current_state, tau, target_rpm, actual_rpm = cs.step()
            
            # 刷新 UI [打印信息]
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
        print("\n\n Stoping...")
        
    finally:
        # 确保安全关闭内存映射文件，释放 Linux 系统资源
        cosim_interface.close()
        print("Memory mapping released, program exits safely.")

if __name__ == "__main__":
    main()