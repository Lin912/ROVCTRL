"""
水下机器人PID位姿控制系统
基于论文: "A Novel Couple Method.pdf" 中的ROV参数
作者: T. Zhang
日期: 2026-02-01
"""

import numpy as np
import pandas as pd
import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
import control as ct
import control.optimal as opt

class ROVConfig:
    """ROV配置参数 - 基于论文Table 4和Table 5"""
    
    def __init__(self):
        # 机器人基本参数 (Table 5)
        self.mass_air = 50.0  # kg (空气中质量)
        self.mass_submerged = 1.642  # kg (水下质量)
        self.length = 0.660  # m
        self.width = 0.524  # m
        self.height = 0.410  # m
        self.design_speed = 0.40  # m/s
        
        # 重心位置 (Table 5)
        self.CG = np.array([-0.129, 0.000, -0.005])  # m (Xr, Yr, Zr坐标系)
        
        # 拖曳点位置 (Table 5)
        self.towing_point = np.array([0.000, 0.000, 0.000])  # m
        
        # 螺旋桨配置 - KA 4-70系列 (Table 4)
        self.n_propellers = 6
        self.propeller_radius = 0.050  # m
        self.propeller_deflection_angle = np.deg2rad(30)  # 30度偏角
        
        # 螺旋桨布局 (Figure 8)
        # 后推进器横向偏移
        self.Bbp = 0.150  # m (Aft thrusters lateral offset)
        # 前推进器横向偏移
        self.Bfp = 0.150  # m (Fore thrusters lateral offset)
        # 垂向推进器横向偏移
        self.Btp = 0.165  # m (Vertical thrusters lateral offset)
        # 后推进器纵向距离 (到CG)
        self.Lbp = 0.250  # m
        # 前推进器纵向距离 (到CG)
        self.Lfp = 0.250  # m
        # 垂向推进器垂向距离
        self.Htp = 0.037  # m
        # CG到中心线的垂向距离
        self.H0 = 0.100  # m
        
        # 螺旋桨位置计算 (基于图8的坐标系)
        self._calculate_propeller_positions()
        
        # 推力配置矩阵 (6自由度 x 6推进器)
        self._calculate_thrust_allocation_matrix()
        
        # 螺旋桨敞水曲线参数 (KA 4-70系列)
        self._init_propeller_curves()
        
    def _calculate_propeller_positions(self):
        """计算每个螺旋桨的位置坐标"""
        # 螺旋桨位置数组 [x, y, z] 在船体坐标系中
        # x: 垂向 (向下为正), y: 横向 (左舷为正), z: 纵向 (向前为正)
        self.prop_positions = np.zeros((6, 3))
        
        # 后推进器 (Aft) - Pbr, Pbl
        # Pbr (右后)
        self.prop_positions[0] = [0, -self.Bbp, -self.Lbp]
        # Pbl (左后)
        self.prop_positions[1] = [0, self.Bbp, -self.Lbp]
        
        # 前推进器 (Fore) - Pfr, Pfl
        # Pfr (右前)
        self.prop_positions[2] = [0, -self.Bfp, self.Lfp]
        # Pfl (左前)
        self.prop_positions[3] = [0, self.Bfp, self.Lfp]
        
        # 垂向推进器 (Vertical) - Ptr, Ptl
        # Ptr (右上)
        self.prop_positions[4] = [-self.Htp, -self.Btp, 0]
        # Ptl (左上)
        self.prop_positions[5] = [-self.Htp, self.Btp, 0]
        
        # 螺旋桨方向向量 (单位向量，表示推力方向)
        self.prop_directions = np.zeros((6, 3))
        
        # 水平推进器 (0-3) - 与纵轴成30度角
        angle = self.propeller_deflection_angle
        for i in range(4):
            self.prop_directions[i] = [0, np.sin(angle), np.cos(angle)]
            # 根据左右调整横向分量符号
            if i in [0, 2]:  # 右侧推进器
                self.prop_directions[i, 1] = -np.sin(angle)
        
        # 垂向推进器 (4-5) - 垂直方向
        self.prop_directions[4] = [1, 0, 0]  # 向下为正
        self.prop_directions[5] = [1, 0, 0]
        
    def _calculate_thrust_allocation_matrix(self):
        """计算推力配置矩阵 A (6x6)
        将6个推进器的推力映射到6自由度力和力矩
        """
        self.A_matrix = np.zeros((6, 6))
        
        for i in range(6):
            # 力分量 (X, Y, Z)
            self.A_matrix[0:3, i] = self.prop_directions[i]
            
            # 力矩分量 (K, M, N) - r × F
            r = self.prop_positions[i]
            F = self.prop_directions[i]
            moment = np.cross(r, F)
            self.A_matrix[3:6, i] = moment
            
    def _init_propeller_curves(self):
        """初始化KA 4-70螺旋桨敞水曲线
        基于螺旋桨理论: T = ρ D^4 KT(J) n|n|
        """
        # 水的密度
        self.rho_water = 1000  # kg/m³
        
        # 螺旋桨直径
        self.D_prop = 2 * self.propeller_radius
        
        # 推力系数KT随进速比J的变化 (KA4-70典型曲线)
        # J = Va/(nD), Va是进速, n是转速(rps)
        self.J_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        self.KT_points = np.array([0.38, 0.35, 0.32, 0.28, 0.24, 0.19, 0.14, 0.08, 0.02])
        
        # 创建插值函数
        self.KT_interp = interp1d(self.J_points, self.KT_points, 
                                  kind='cubic', fill_value='extrapolate')
        
    def thrust_to_rpm(self, T_desired, Va):
        """
        根据期望推力和进速计算所需转速
        逆螺旋桨模型: T = ρ D^4 KT(J) n|n|
        
        Parameters:
        -----------
        T_desired : float
            期望推力 (N)
        Va : float
            进速 (m/s) - 流向螺旋桨的水流速度
            
        Returns:
        --------
        n_rpm : float
            所需转速 (RPM)
        """
        if abs(T_desired) < 1e-6:
            return 0.0
            
        sign = np.sign(T_desired)
        T_abs = abs(T_desired)
        
        # 迭代求解转速
        # 初始猜测
        n_rps = np.sqrt(T_abs / (self.rho_water * self.D_prop**4 * self.KT_points[0]))
        if n_rps < 0.1:
            n_rps = 5.0  # 最小转速
            
        # 简单迭代
        for _ in range(5):
            J = Va / (n_rps * self.D_prop) if n_rps > 0.1 else 0
            J = np.clip(J, 0, 0.8)
            KT = self.KT_interp(J)
            n_rps_new = np.sqrt(T_abs / (self.rho_water * self.D_prop**4 * KT))
            n_rps = 0.5 * n_rps + 0.5 * n_rps_new
            
        return sign * n_rps * 60  # 转换为RPM


class SensorMonitor:
    """实时传感器数据监控器
    从txt文件读取ROV状态
    """
    
    def __init__(self, sensor_file='rov_sensor_data.txt'):
        self.sensor_file = sensor_file
        self.lock = threading.Lock()
        self.current_data = None
        self.last_update = 0
        self.running = True
        
        # 传感器数据格式
        self.sensor_columns = [
            'timestamp',      # 时间戳
            'x', 'y', 'z',    # 位置 (m) - 惯性坐标系
            'roll', 'pitch', 'yaw',  # 姿态角 (rad)
            'u', 'v', 'w',    # 线速度 (m/s) - 船体坐标系
            'p', 'q', 'r',    # 角速度 (rad/s)
            'depth',          # 深度 (m)
            'altitude'        # 对地高度 (m)
        ]
        
    def start_monitoring(self):
        """启动监控线程"""
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _monitor_loop(self):
        """监控循环"""
        last_pos = 0
        while self.running:
            try:
                # 读取文件最后一行
                with open(self.sensor_file, 'r') as f:
                    # 跳到文件末尾
                    f.seek(0, 2)
                    # 找到最后一行
                    pos = f.tell()
                    while pos > 0:
                        pos -= 1
                        f.seek(pos)
                        if f.read(1) == '\n':
                            break
                    last_line = f.readline().strip()
                    
                if last_line:
                    values = list(map(float, last_line.split(',')))
                    if len(values) == len(self.sensor_columns):
                        with self.lock:
                            self.current_data = dict(zip(self.sensor_columns, values))
                            self.last_update = time.time()
                            
            except Exception as e:
                print(f"传感器读取错误: {e}")
                
            time.sleep(0.01)  # 100Hz采样
            
    def get_state(self):
        """获取当前状态"""
        with self.lock:
            if self.current_data is None:
                return None
            return self.current_data.copy()
            
    def stop(self):
        """停止监控"""
        self.running = False


class ROVPIDController:
    """ROV PID位姿控制器"""
    
    def __init__(self, rov_config):
        self.config = rov_config
        
        # PID参数 [P, I, D]
        # 位置控制 (x, y, z)
        self.Kp_pos = np.array([50.0, 50.0, 80.0])   # 比例增益
        self.Ki_pos = np.array([5.0, 5.0, 10.0])     # 积分增益
        self.Kd_pos = np.array([30.0, 30.0, 40.0])   # 微分增益
        
        # 姿态控制 (roll, pitch, yaw)
        self.Kp_att = np.array([20.0, 20.0, 40.0])   # 比例增益
        self.Ki_att = np.array([2.0, 2.0, 5.0])      # 积分增益
        self.Kd_att = np.array([15.0, 15.0, 25.0])   # 微分增益
        
        # 积分项限制
        self.max_integral = np.array([100.0, 100.0, 200.0,  # 位置积分限制
                                      50.0, 50.0, 100.0])    # 姿态积分限制
        
        # 误差历史
        self.error_history = []
        self.max_history = 1000
        
        # 积分项
        self.integral_pos = np.zeros(3)
        self.integral_att = np.zeros(3)
        
        # 上一时刻误差
        self.last_error_pos = None
        self.last_error_att = None
        self.last_time = None
        
        # 控制输出限制
        self.max_force = 200.0  # 最大力 (N)
        self.max_moment = 50.0   # 最大力矩 (Nm)
        self.max_rpm = 2000      # 最大转速 (RPM)
        
    def compute_control(self, current_state, desired_state):
        """
        计算控制输出
        
        Parameters:
        -----------
        current_state : dict
            当前状态
        desired_state : dict
            期望状态
            
        Returns:
        --------
        tau : ndarray
            期望的广义控制力 [X, Y, Z, K, M, N]
        """
        current_time = time.time()
        
        # 提取当前位置和姿态
        current_pos = np.array([current_state['x'], 
                                current_state['y'], 
                                current_state['z']])
        current_att = np.array([current_state['roll'],
                                current_state['pitch'],
                                current_state['yaw']])
        
        # 提取当前速度
        current_vel = np.array([current_state['u'],
                                current_state['v'],
                                current_state['w']])
        current_omega = np.array([current_state['p'],
                                  current_state['q'],
                                  current_state['r']])
        
        # 期望位置和姿态
        desired_pos = np.array([desired_state['x'],
                                desired_state['y'],
                                desired_state['z']])
        desired_att = np.array([desired_state['roll'],
                                desired_state['pitch'],
                                desired_state['yaw']])
        
        # 计算误差
        error_pos = desired_pos - current_pos
        error_att = self._normalize_angle(desired_att - current_att)
        
        # 计算微分项 (使用速度反馈，更稳定)
        if self.last_error_pos is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                # 使用速度作为微分项
                deriv_pos = -current_vel  # 负的速度
                deriv_att = -current_omega
            else:
                deriv_pos = np.zeros(3)
                deriv_att = np.zeros(3)
        else:
            deriv_pos = np.zeros(3)
            deriv_att = np.zeros(3)
            
        # 更新积分项
        if self.last_time is not None:
            dt = current_time - self.last_time
            self.integral_pos += error_pos * dt
            self.integral_att += error_att * dt
            
            # 积分限幅
            self.integral_pos = np.clip(self.integral_pos, 
                                       -self.max_integral[:3], 
                                       self.max_integral[:3])
            self.integral_att = np.clip(self.integral_att,
                                       -self.max_integral[3:],
                                       self.max_integral[3:])
        
        # PID控制律
        force = (self.Kp_pos * error_pos + 
                 self.Ki_pos * self.integral_pos + 
                 self.Kd_pos * deriv_pos)
        
        moment = (self.Kp_att * error_att + 
                  self.Ki_att * self.integral_att + 
                  self.Kd_att * deriv_att)
        
        # 限制输出
        force = np.clip(force, -self.max_force, self.max_force)
        moment = np.clip(moment, -self.max_moment, self.max_moment)
        
        # 组合广义力
        tau = np.concatenate([force, moment])
        
        # 保存状态
        self.last_error_pos = error_pos
        self.last_error_att = error_att
        self.last_time = current_time
        
        # 记录误差
        self.error_history.append({
            'time': current_time,
            'error_pos': error_pos.copy(),
            'error_att': error_att.copy()
        })
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
            
        return tau
    
    def _normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi


class ThrustAllocator:
    """推力分配器
    将广义力分配为各螺旋桨推力
    """
    
    def __init__(self, rov_config):
        self.config = rov_config
        self.A_matrix = rov_config.A_matrix
        
        # 计算伪逆用于推力分配
        self.A_pinv = np.linalg.pinv(self.A_matrix)
        
        # 推力限制
        self.max_thrust_per_prop = 50.0  # N
        self.min_thrust_per_prop = -30.0  # N (反向推力)
        
    def allocate(self, tau):
        """
        分配推力
        
        Parameters:
        -----------
        tau : ndarray
            期望的广义力 [X, Y, Z, K, M, N]
            
        Returns:
        --------
        thrusts : ndarray
            各螺旋桨推力 [T0, T1, T2, T3, T4, T5]
        """
        # 使用伪逆求解推力
        thrusts = self.A_pinv @ tau
        
        # 推力限制
        thrusts = np.clip(thrusts, 
                         self.min_thrust_per_prop, 
                         self.max_thrust_per_prop)
        
        return thrusts


class MotorController:
    """电机控制器
    将推力指令转换为转速指令，考虑来流影响
    """
    
    def __init__(self, rov_config):
        self.config = rov_config
        
        # 电机响应模型
        self.motor_time_constant = 0.1  # s
        self.last_rpm = np.zeros(6)
        self.last_time = None
        
        # 命令输出文件
        self.cmd_file = 'motor_commands.txt'
        
    def thrust_to_rpm(self, thrusts, Va_array):
        """
        将推力转换为转速
        
        Parameters:
        -----------
        thrusts : ndarray
            期望推力 [T0, T1, T2, T3, T4, T5]
        Va_array : ndarray
            各螺旋桨的进速
            
        Returns:
        --------
        rpm : ndarray
            期望转速
        """
        rpm = np.zeros(6)
        
        for i in range(6):
            rpm[i] = self.config.thrust_to_rpm(thrusts[i], Va_array[i])
            
        # 转速限制
        rpm = np.clip(rpm, -self.config.max_rpm, self.config.max_rpm)
        
        return rpm
    
    def apply_motor_dynamics(self, desired_rpm):
        """应用电机动态响应"""
        current_time = time.time()
        
        if self.last_time is None:
            self.last_time = current_time
            self.last_rpm = desired_rpm
            return desired_rpm
        
        dt = current_time - self.last_time
        
        # 一阶低通滤波模拟电机响应
        alpha = dt / (self.motor_time_constant + dt)
        actual_rpm = self.last_rpm + alpha * (desired_rpm - self.last_rpm)
        
        self.last_rpm = actual_rpm
        self.last_time = current_time
        
        return actual_rpm
    
    def send_commands(self, rpm):
        """发送转速指令到执行器"""
        try:
            timestamp = datetime.now().isoformat()
            cmd_str = f"{timestamp}," + ",".join([f"{r:.1f}" for r in rpm])
            
            with open(self.cmd_file, 'a') as f:
                f.write(cmd_str + '\n')
                
        except Exception as e:
            print(f"命令发送错误: {e}")


class ROVControlSystem:
    """ROV控制系统主类
    集成所有组件，实现实时控制
    """
    
    def __init__(self, config_file=None):
        # 初始化配置
        self.config = ROVConfig()
        
        # 初始化各组件
        self.sensor = SensorMonitor()
        self.pid = ROVPIDController(self.config)
        self.allocator = ThrustAllocator(self.config)
        self.motor = MotorController(self.config)
        
        # 控制循环参数
        self.control_rate = 100  # Hz
        self.running = False
        
        # 期望状态 (默认悬停)
        self.desired_state = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        
        # 数据记录
        self.log_data = []
        
    def set_desired_position(self, x, y, z):
        """设置期望位置"""
        self.desired_state['x'] = x
        self.desired_state['y'] = y
        self.desired_state['z'] = z
        
    def set_desired_attitude(self, roll, pitch, yaw):
        """设置期望姿态"""
        self.desired_state['roll'] = roll
        self.desired_state['pitch'] = pitch
        self.desired_state['yaw'] = yaw
        
    def start(self):
        """启动控制系统"""
        print("启动ROV控制系统...")
        
        # 启动传感器监控
        self.sensor.start_monitoring()
        time.sleep(1)  # 等待传感器数据
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        print("控制系统已启动")
        
    def _control_loop(self):
        """主控制循环"""
        cycle_time = 1.0 / self.control_rate
        next_time = time.time()
        
        while self.running:
            try:
                # 获取当前状态
                current_state = self.sensor.get_state()
                
                if current_state is not None:
                    # 计算控制输出
                    tau = self.pid.compute_control(current_state, self.desired_state)
                    
                    # 分配推力
                    thrusts = self.allocator.allocate(tau)
                    
                    # 计算各螺旋桨进速
                    Va_array = self._compute_advance_velocities(current_state)
                    
                    # 转换为转速
                    desired_rpm = self.motor.thrust_to_rpm(thrusts, Va_array)
                    
                    # 应用电机动态
                    actual_rpm = self.motor.apply_motor_dynamics(desired_rpm)
                    
                    # 发送命令
                    self.motor.send_commands(actual_rpm)
                    
                    # 记录数据
                    self._log_data(current_state, tau, thrusts, actual_rpm)
                    
                # 精确时序控制
                next_time += cycle_time
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"控制循环错误: {e}")
                time.sleep(0.01)
                
    def _compute_advance_velocities(self, state):
        """计算各螺旋桨的进速"""
        # 从状态中提取速度
        u, v, w = state['u'], state['v'], state['w']
        p, q, r = state['p'], state['q'], state['r']
        
        Va_array = np.zeros(6)
        
        # 计算每个螺旋桨位置的水流速度
        for i in range(6):
            # 螺旋桨位置
            pos = self.config.prop_positions[i]
            
            # 该位置由于旋转引起的速度
            rot_vel = np.cross([p, q, r], pos)
            
            # 总速度 = 本体速度 + 旋转速度
            total_vel = np.array([u, v, w]) + rot_vel
            
            # 沿螺旋桨轴向的速度分量
            Va_array[i] = np.dot(total_vel, self.config.prop_directions[i])
            
        return Va_array
    
    def _log_data(self, state, tau, thrusts, rpm):
        """记录控制数据"""
        log_entry = {
            'time': time.time(),
            'position': [state['x'], state['y'], state['z']],
            'attitude': [state['roll'], state['pitch'], state['yaw']],
            'velocity': [state['u'], state['v'], state['w']],
            'tau': tau.copy(),
            'thrusts': thrusts.copy(),
            'rpm': rpm.copy()
        }
        self.log_data.append(log_entry)
        
        # 限制日志大小
        if len(self.log_data) > 10000:
            self.log_data.pop(0)
            
    def stop(self):
        """停止控制系统"""
        print("停止控制系统...")
        self.running = False
        self.sensor.stop()
        
        # 发送零转速指令
        self.motor.send_commands(np.zeros(6))
        
        # 保存日志
        self.save_log()
        
        print("控制系统已停止")
        
    def save_log(self, filename='control_log.csv'):
        """保存控制日志"""
        if not self.log_data:
            return
            
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
                           'u', 'v', 'w', 'tau_x', 'tau_y', 'tau_z',
                           'tau_k', 'tau_m', 'tau_n'] + 
                           [f'thrust_{i}' for i in range(6)] +
                           [f'rpm_{i}' for i in range(6)])
            
            for entry in self.log_data:
                row = [entry['time']]
                row.extend(entry['position'])
                row.extend(entry['attitude'])
                row.extend(entry['velocity'])
                row.extend(entry['tau'])
                row.extend(entry['thrusts'])
                row.extend(entry['rpm'])
                writer.writerow(row)
                
        print(f"日志已保存到 {filename}")


class ControlMonitor:
    """控制监控与可视化"""
    
    def __init__(self, control_system):
        self.cs = control_system
        
        # 创建图形
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 初始化线条
        self.lines = {}
        self._init_plots()
        
    def _init_plots(self):
        """初始化绘图"""
        # 位置误差
        self.axes[0, 0].set_title('位置误差')
        self.axes[0, 0].set_xlabel('时间 (s)')
        self.axes[0, 0].set_ylabel('误差 (m)')
        self.lines['pos_error'] = [
            self.axes[0, 0].plot([], [], 'r-', label='X')[0],
            self.axes[0, 0].plot([], [], 'g-', label='Y')[0],
            self.axes[0, 0].plot([], [], 'b-', label='Z')[0]
        ]
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True)
        
        # 姿态误差
        self.axes[0, 1].set_title('姿态误差')
        self.axes[0, 1].set_xlabel('时间 (s)')
        self.axes[0, 1].set_ylabel('误差 (rad)')
        self.lines['att_error'] = [
            self.axes[0, 1].plot([], [], 'r-', label='Roll')[0],
            self.axes[0, 1].plot([], [], 'g-', label='Pitch')[0],
            self.axes[0, 1].plot([], [], 'b-', label='Yaw')[0]
        ]
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True)
        
        # 控制力
        self.axes[0, 2].set_title('控制力')
        self.axes[0, 2].set_xlabel('时间 (s)')
        self.axes[0, 2].set_ylabel('力 (N)')
        self.lines['force'] = [
            self.axes[0, 2].plot([], [], 'r-', label='X')[0],
            self.axes[0, 2].plot([], [], 'g-', label='Y')[0],
            self.axes[0, 2].plot([], [], 'b-', label='Z')[0]
        ]
        self.axes[0, 2].legend()
        self.axes[0, 2].grid(True)
        
        # 控制力矩
        self.axes[1, 0].set_title('控制力矩')
        self.axes[1, 0].set_xlabel('时间 (s)')
        self.axes[1, 0].set_ylabel('力矩 (Nm)')
        self.lines['moment'] = [
            self.axes[1, 0].plot([], [], 'r-', label='K')[0],
            self.axes[1, 0].plot([], [], 'g-', label='M')[0],
            self.axes[1, 0].plot([], [], 'b-', label='N')[0]
        ]
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True)
        
        # 螺旋桨推力
        self.axes[1, 1].set_title('螺旋桨推力')
        self.axes[1, 1].set_xlabel('时间 (s)')
        self.axes[1, 1].set_ylabel('推力 (N)')
        self.lines['thrust'] = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i in range(6):
            line, = self.axes[1, 1].plot([], [], colors[i], 
                                         label=f'Prop{i}', alpha=0.7)
            self.lines['thrust'].append(line)
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True)
        
        # 转速
        self.axes[1, 2].set_title('电机转速')
        self.axes[1, 2].set_xlabel('时间 (s)')
        self.axes[1, 2].set_ylabel('转速 (RPM)')
        self.lines['rpm'] = []
        for i in range(6):
            line, = self.axes[1, 2].plot([], [], colors[i], 
                                         label=f'Motor{i}', alpha=0.7)
            self.lines['rpm'].append(line)
        self.axes[1, 2].legend()
        self.axes[1, 2].grid(True)
        
        plt.tight_layout()
        
    def update(self, frame):
        """更新绘图"""
        if len(self.cs.pid.error_history) < 2:
            return
            
        # 提取数据
        times = [e['time'] - self.cs.pid.error_history[0]['time'] 
                 for e in self.cs.pid.error_history]
        
        # 更新位置误差
        pos_errors = np.array([e['error_pos'] for e in self.cs.pid.error_history])
        for i, line in enumerate(self.lines['pos_error']):
            line.set_data(times, pos_errors[:, i])
            
        # 更新姿态误差
        att_errors = np.array([e['error_att'] for e in self.cs.pid.error_history])
        for i, line in enumerate(self.lines['att_error']):
            line.set_data(times, att_errors[:, i])
            
        # 更新控制力和力矩
        if len(self.cs.log_data) > 1:
            log_times = [entry['time'] - self.cs.log_data[0]['time'] 
                        for entry in self.cs.log_data]
            taus = np.array([entry['tau'] for entry in self.cs.log_data])
            
            for i in range(3):
                self.lines['force'][i].set_data(log_times, taus[:, i])
            for i in range(3):
                self.lines['moment'][i].set_data(log_times, taus[:, i+3])
                
            # 更新推力和转速
            thrusts = np.array([entry['thrusts'] for entry in self.cs.log_data])
            rpms = np.array([entry['rpm'] for entry in self.cs.log_data])
            
            for i in range(6):
                self.lines['thrust'][i].set_data(log_times, thrusts[:, i])
                self.lines['rpm'][i].set_data(log_times, rpms[:, i])
                
        # 调整坐标轴
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()
            
        return [line for line_list in self.lines.values() 
                for line in (line_list if isinstance(line_list, list) else [line_list])]
        
    def start_monitor(self):
        """启动监控"""
        ani = FuncAnimation(self.fig, self.update, interval=100, blit=True)
        plt.show()
        return ani


def main():
    """主函数"""
    print("=" * 60)
    print("水下机器人PID位姿控制系统")
    print("基于论文参数: KA 4-70螺旋桨, 6推进器配置")
    print("=" * 60)
    
    # 创建控制系统
    cs = ROVControlSystem()
    
    try:
        # 启动系统
        cs.start()
        
        # 设置期望状态 (悬停在原点)
        cs.set_desired_position(0.0, 0.0, -5.0)  # 深度5米
        cs.set_desired_attitude(0.0, 0.0, 0.0)
        
        print("\n期望状态: 深度5米, 零姿态")
        print("等待传感器数据...")
        
        # 等待稳定
        time.sleep(5)
        
        # 创建监控
        monitor = ControlMonitor(cs)
        
        print("\n监控窗口已打开，按Ctrl+C停止")
        monitor.start_monitor()
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        cs.stop()
        
    print("\n控制系统测试完成")


if __name__ == "__main__":
    main()