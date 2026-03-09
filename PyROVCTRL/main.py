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
    cs.set_desired_position(-0.02, 0.0, 0.0)
    cs.set_desired_attitude(0.0, 0.0, 0.0)
    
    print("\n Wating for StarCCM+ simulating...")
    
    last_display_time = -1.0    # 初始刷新时间
    display_interval = 0.01      # 屏幕刷新时间 
    
    # 打印包含姿态的终极表头 (宽度增加到 180)
    print("=" * 180)
    print(f"{'Time(s)':^8} | {'Position (X,Y,Z)':^20} | {'Attitude (R,P,Y)°':^20} | {'Force (Fx,Fy,Fz)':^22} | {'Moment (Mx,My,Mz)':^22} | {'Target RPM (T0~T5)':^32} | {'Real RPM (T0~T5)':^32}")
    print("=" * 180)

    try:
        while True:
            # 驱动控制器步进
            sim_time, current_state, tau, target_rpm, actual_rpm = cs.step()
            
            # 刷新 UI [打印信息]
            if sim_time - last_display_time >= (display_interval - 1e-6):
                # 1. 位置
                pos_str = f"({current_state['x']:>5.2f}, {current_state['y']:>5.2f}, {current_state['z']:>5.2f})"
                # 1.5. 姿态 (将弧度转为角度打印，保留 1 位小数，极其直观)
                r_deg = np.degrees(current_state['roll'])
                p_deg = np.degrees(current_state['pitch'])
                y_deg = np.degrees(current_state['yaw'])
                att_str = f"({r_deg:>5.1f}, {p_deg:>5.1f}, {y_deg:>5.1f})"
                # 2. 力和力矩
                force_str = f"({tau[0]:>5.1f}, {tau[1]:>5.1f}, {tau[2]:>5.1f})"
                moment_str = f"({tau[3]:>5.1f}, {tau[4]:>5.1f}, {tau[5]:>5.1f})"
                # 3 & 4. 转速 (保持你现在的样子)
                target_str = f"[{target_rpm[0]:>5.0f} {target_rpm[1]:>5.0f} {target_rpm[2]:>5.0f} {target_rpm[3]:>5.0f} {target_rpm[4]:>5.0f} {target_rpm[5]:>5.0f}]"
                actual_str = f"[{actual_rpm[0]:>5.0f} {actual_rpm[1]:>5.0f} {actual_rpm[2]:>5.0f} {actual_rpm[3]:>5.0f} {actual_rpm[4]:>5.0f} {actual_rpm[5]:>5.0f}]"
                # 5. 组合打印 (把 att_str 塞进去)
                print(f" {sim_time:>7.3f}s | {pos_str:^20} | {att_str:^20} | {force_str:^22} | {moment_str:^22} | {target_str:^32} | {actual_str:^32}")
                last_display_time = sim_time
                
    except KeyboardInterrupt:
        print("\n\n Stoping...")
        
    finally:
        # 确保安全关闭内存映射文件，释放 Linux 系统资源
        cosim_interface.close()
        print("Memory mapping released, program exits safely.")

if __name__ == "__main__":
    main()
