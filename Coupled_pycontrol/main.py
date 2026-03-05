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
import yaml
import struct

class ConfigLoader:
    """加载并解析 YAML 配置文件"""
    @staticmethod
    def load(filepath='rov_config.yaml'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"警告: 找不到配置文件 {filepath}，将使用默认参数。")
            return None

class CoSimInterface:
    """与 STAR-CCM+ 耦合的共享内存接口"""
    def __init__(self, mem_name="CFD_Control_Shared_Mem"):
        # 1. 连接到 CFD 创建的共享内存块
        # self.shm = shared_memory.SharedMemory(name=mem_name)
        
        # 握手标志位偏移量 (假设内存前几个字节用于状态同步)
        self.FLAG_CFD_READY = 1     # CFD写完了，等待Python读
        self.FLAG_PYTHON_READY = 2  # Python写完了，等待CFD读
        
    def wait_for_cfd_data(self):
        """阻塞等待 CFD 计算完成当前步，并读取状态"""
        while True:
            # 读取内存中的标志位
            # flag = struct.unpack('i', self.shm.buf[0:4])[0]
            flag = self.FLAG_CFD_READY # 伪代码：假设读到CFD已准备好
            
            if flag == self.FLAG_CFD_READY:
                # 解析传感器数据 (从共享内存中按照约定好的 C++ struct 格式解包)
                # 例如: 1个double时间戳 + 12个double位姿与速度 = 13*8 = 104字节
                # data = struct.unpack('13d', self.shm.buf[4:108])
                
                state = {
                    'timestamp': 0.001, # data[0]
                    'x': 0.0, 'y': 0.0, 'z': -5.0, # data[1:4]...
                    # ... 填充完整的字典
                }
                return state
                
            # 极短的休眠，防止死循环占满 100% CPU
            time.sleep(0.0001) 

    def send_motor_commands(self, rpm_array):
        """将算好的转速写入共享内存，并通知 CFD"""
        # 1. 将 6 个 RPM 转换为字节 (6个double)
        # rpm_bytes = struct.pack('6d', *rpm_array)
        
        # 2. 写入特定的内存地址
        # self.shm.buf[108:156] = rpm_bytes
        
        # 3. 修改标志位，通知 STAR-CCM+ 可以继续算了
        # self.shm.buf[0:4] = struct.pack('i', self.FLAG_PYTHON_READY)
        pass

class ROVConfig:
    """ROV配置参数"""
    def __init__(self, config_dict=None):
        # 如果没有传入配置，给一个空字典防报错（或者在这里写一套备用默认值）
        cfg = config_dict.get('robot', {}) if config_dict else {}
        
        # 从字典中读取，支持回退到默认值
        self.mass_air = cfg.get('mass_air', 50.0)
        self.mass_submerged = cfg.get('mass_submerged', 1.642)
        
        dims = cfg.get('dimensions', [0.660, 0.524, 0.410])
        self.length, self.width, self.height = dims
        
        self.CG = np.array(cfg.get('cg', [-0.129, 0.000, -0.005]))
        
        thruster_cfg = cfg.get('thrusters', {})
        self.max_rpm = thruster_cfg.get('max_rpm', 2000)
        self.propeller_deflection_angle = np.deg2rad(thruster_cfg.get('deflection_angle_deg', 30))
        
        # ... 保留你原有的螺旋桨布局计算 _calculate_propeller_positions 等代码 ...
        self.n_propellers = 6
        self.propeller_radius = 0.050
        # 螺旋桨布局
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
        
        self._calculate_propeller_positions()
        self._calculate_thrust_allocation_matrix()
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
        将6个推进器的推力映射 -> (6自由度)力和力矩
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
        基于螺旋桨理论: T = rho * D^4 * K_T * (J) * n|n|
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
        self.KT_interp = interp1d(self.J_points, self.KT_points, kind='cubic', fill_value='extrapolate') # "cubic extrapolate -> 三次曲线外推"
        
    def thrust_to_rpm(self, T_desired, Va):
        """
        根据期望推力和进速计算所需转速
        逆螺旋桨模型: T = rho * D^4 * K_T * (J) * n|n|
        
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
            J = np.clip(J, 0, 0.8) # 限制进速系数在(0.0, 0.8)
            KT = self.KT_interp(J) # 计算得到推力系数
            n_rps_new = np.sqrt(T_abs / (self.rho_water * self.D_prop**4 * KT)) # 计算新的转速
            n_rps = 0.5 * n_rps + 0.5 * n_rps_new # 松弛迭代,逐渐找到自洽的转速值
            
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
                
            time.sleep(0.01)  # 从传感器采样频率 -> 100Hz --------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
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
    
    def __init__(self, rov_config, config_dict=None):
        self.config = rov_config
        
        pid_cfg = config_dict.get('pid', {}) if config_dict else {}
        pos_cfg = pid_cfg.get('position', {})
        att_cfg = pid_cfg.get('attitude', {})
        limits_cfg = pid_cfg.get('limits', {})
        
        # 动态加载 PID 参数
        self.Kp_pos = np.array(pos_cfg.get('kp', [50.0, 50.0, 80.0]))
        self.Ki_pos = np.array(pos_cfg.get('ki', [5.0, 5.0, 10.0]))
        self.Kd_pos = np.array(pos_cfg.get('kd', [30.0, 30.0, 40.0]))
        
        self.Kp_att = np.array(att_cfg.get('kp', [20.0, 20.0, 40.0]))
        self.Ki_att = np.array(att_cfg.get('ki', [2.0, 2.0, 5.0]))
        self.Kd_att = np.array(att_cfg.get('kd', [15.0, 15.0, 25.0]))
        
        # 拼接积分限制
        pos_int = pos_cfg.get('max_integral', [100.0, 100.0, 200.0])
        att_int = att_cfg.get('max_integral', [50.0, 50.0, 100.0])
        self.max_integral = np.array(pos_int + att_int)
        
        # 控制输出限制
        self.max_force = limits_cfg.get('max_force', 200.0)
        self.max_moment = limits_cfg.get('max_moment', 50.0)
        self.max_rpm = 2000 # 最大转速 (RPM)
        
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
        # 删除：current_time = time.time()
        # 替换为从状态字典中获取仿真时间
        # current_time = time.time()
        current_time = current_state['timestamp']
        
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
            # 这里的 dt 就是真实的 CFD 时间步长 (例如 0.001s)，不再有系统的跳动
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
    
    def apply_motor_dynamics(self, desired_rpm, current_sim_time):
        """需要传入当前的仿真时间"""
        #"""应用电机动态响应"""
        #current_time = time.time()
        
        if self.last_time is None:
            self.last_time = current_sim_time
            self.last_rpm = desired_rpm
            return desired_rpm
        
        dt = current_sim_time - self.last_time
        
        # 一阶低通滤波模拟电机响应
        alpha = dt / (self.motor_time_constant + dt)
        actual_rpm = self.last_rpm + alpha * (desired_rpm - self.last_rpm)
        
        self.last_rpm = actual_rpm
        self.last_time = current_sim_time
        
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
    
    def __init__(self, config_file='rov_config.yaml'):
        # 1. 加载外部配置
        self.raw_config = ConfigLoader.load(config_file)
        
        # 初始化各组件
        self.config = ROVConfig(self.raw_config)
        self.pid = ROVPIDController(self.config, self.raw_config)
        
        # 系统级配置
        sys_cfg = self.raw_config.get('system', {}) if self.raw_config else {}
        self.control_rate = sys_cfg.get('control_rate', 100)
        sensor_file = sys_cfg.get('sensor_file', 'rov_sensor_data.txt')
        self.sensor = SensorMonitor(sensor_file=sensor_file)
        self.allocator = ThrustAllocator(self.config)
        self.motor = MotorController(self.config)
        
        # 替换传感器和电机模块
        self.cosim_interface = CoSimInterface()
        self.allocator = ThrustAllocator(self.config)
        
        # 控制循环参数
        # self.control_rate = 100  # Hz 控制器频率 (Controller Frequency) ->写入 motor_commands.txt -> 100Hz（每10ms执行一次控制）-> 执行PID计算，发送控制指令
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
        print("启动联合仿真ROV控制系统...")
        
        # 启动传感器监控
        self.sensor.start_monitoring()
        time.sleep(1)  # 等待传感器数据
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        self._sync_control_loop() # 直接在主线程运行，不再开新线程
        print("控制系统已启动")
    
    def _sync_control_loop(self):
        """基于 CFD 驱动的同步控制循环"""
        while self.running:
            try:
                # 1. 阻塞等待：直到 STAR-CCM+ 算完这一步并更新了内存
                current_state = self.cosim_interface.wait_for_cfd_data()
                current_sim_time = current_state['timestamp']
                
                # 2. 控制计算 (PID内部已经改为使用 current_sim_time)
                tau = self.pid.compute_control(current_state, self.desired_state)
                thrusts = self.allocator.allocate(tau)
                Va_array = self._compute_advance_velocities(current_state)
                
                # 计算转速 (注意这里传入了当前仿真时间)
                desired_rpm = self.motor.thrust_to_rpm(thrusts, Va_array)
                actual_rpm = self.motor.apply_motor_dynamics(desired_rpm, current_sim_time)
                
                # 3. 立即将指令写入共享内存，唤醒 STAR-CCM+ 算下一步
                self.cosim_interface.send_motor_commands(actual_rpm)
                
                # 记录数据 (可选)
                self._log_data(...)
                
                # ❌ 注意：这里绝对不能有 time.sleep()！
                # 循环速度完全取决于 CFD 算的有多快
                
            except KeyboardInterrupt:
                self.stop()
                break
    
    def _control_loop(self):
        """主控制循环"""
        cycle_time = 1.0 / self.control_rate #控制循环频率 -> 100Hz -> self.control_rate(100) -> cycle_time(0.01s) -> 100Hz
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

def main():
    """主函数 - 带详细输出的版本"""
    print("=" * 70)
    print("水下机器人PID位姿控制系统")
    print("控制机构-KA 4-70, 6推配置")
    print("=" * 70)
    
    # 检查传感器数据文件是否存在
    import os
    sensor_file = 'rov_sensor_data.txt'
    
    if not os.path.exists(sensor_file) or os.path.getsize(sensor_file) == 0:
        print(f"\n警告: 传感器数据文件 {sensor_file} 不存在或为空")
        print("请确保已运行 PID_DATA.py 生成数据")
        print("继续运行，但可能无法获取数据...\n")
    
    # 创建控制系统
    cs = ROVControlSystem()
    
    try:
        # 启动系统
        cs.start()
        
        # 设置期望状态 (悬停在原点)
        cs.set_desired_position(0.0, 0.0, -5.0)  # 深度5米
        cs.set_desired_attitude(0.0, 0.0, 0.0)
        
        # 显示PID参数
        print("\n" + "=" * 70)
        print("PID控制器参数:")
        print("-" * 70)
        print(f"位置控制 - P: {cs.pid.Kp_pos}  I: {cs.pid.Ki_pos}  D: {cs.pid.Kd_pos}")
        print(f"姿态控制 - P: {cs.pid.Kp_att}  I: {cs.pid.Ki_att}  D: {cs.pid.Kd_att}")
        print(f"积分限制: {cs.pid.max_integral}")
        print(f"输出限制 - 最大力: {cs.pid.max_force} N, 最大力矩: {cs.pid.max_moment} Nm")
        print("=" * 70)
        
        print("\n期望状态:")
        print(f"  位置: x={cs.desired_state['x']:.2f}m, y={cs.desired_state['y']:.2f}m, z={cs.desired_state['z']:.2f}m")
        print(f"  姿态: roll={np.rad2deg(cs.desired_state['roll']):.1f}°, "
              f"pitch={np.rad2deg(cs.desired_state['pitch']):.1f}°, "
              f"yaw={np.rad2deg(cs.desired_state['yaw']):.1f}°")
        
        print("\n等待传感器数据...")
        
        # 等待第一个传感器数据
        start_time = time.time()
        while cs.sensor.get_state() is None:
            time.sleep(0.1)
            if time.time() - start_time > 10:
                print("错误: 未收到传感器数据，请确保PID_DATA.py正在运行")
                return
        
        print("传感器数据已连接")
        
        # 记录开始时间
        program_start = time.time()
        last_display = 0
        display_interval = 2.0  # 信息显示频率：每2秒显示一次
        
        print("\n" + "=" * 90)
        print(f"{'运行时间':>8} | {'位置误差(m)':^25} | {'姿态误差(°)':^25} | {'当前状态':^30}")
        print(f"{'':>8} | {'x':^8}{'y':^8}{'z':^8} | {'roll':^8}{'pitch':^8}{'yaw':^8} | {'信息':^30}")
        print("=" * 90)
        
        while True:
            current_time = time.time()
            runtime = current_time - program_start
            
            # 定期显示状态
            if current_time - last_display >= display_interval:
                current = cs.sensor.get_state()
                
                if current:
                    # 计算误差
                    error_pos = np.array([
                        cs.desired_state['x'] - current['x'],
                        cs.desired_state['y'] - current['y'],
                        cs.desired_state['z'] - current['z']
                    ])
                    
                    error_att = np.array([
                        cs.desired_state['roll'] - current['roll'],
                        cs.desired_state['pitch'] - current['pitch'],
                        cs.desired_state['yaw'] - current['yaw']
                    ])
                    error_att_deg = np.rad2deg(error_att)
                    
                    # 获取当前PID输出
                    if cs.pid.last_time:
                        control_dt = current_time - cs.pid.last_time
                    else:
                        control_dt = 0
                    
                    # 获取控制力
                    if cs.log_data:
                        last_tau = cs.log_data[-1]['tau']
                        force_magnitude = np.linalg.norm(last_tau[:3])
                        moment_magnitude = np.linalg.norm(last_tau[3:])
                    else:
                        force_magnitude = 0
                        moment_magnitude = 0
                    
                    # 格式化输出 - 修复字符串拼接错误
                    time_str = f"{runtime:>8.1f}s"
                    
                    pos_error_str = f"{error_pos[0]:>8.3f}{error_pos[1]:>8.3f}{error_pos[2]:>8.3f}"
                    
                    att_error_str = f"{error_att_deg[0]:>8.2f}{error_att_deg[1]:>8.2f}{error_att_deg[2]:>8.2f}"
                    
                    # 当前状态信息
                    current_pos_str = f"p:({current['x']:.2f},{current['y']:.2f},{current['z']:.2f})"
                    current_att_str = f"a:({np.rad2deg(current['roll']):.1f},{np.rad2deg(current['pitch']):.1f},{np.rad2deg(current['yaw']):.1f})"
                    control_info = f"F:{force_magnitude:>5.1f}N M:{moment_magnitude:>5.1f}Nm"
                    
                    # 修复这里的字符串拼接问题 - 使用逗号分隔或者格式化字符串
                    print(f"{time_str} | {pos_error_str} | {att_error_str} | {current_pos_str}")
                    print(f"{'':>8} | {'':25} | {'':25} | {current_att_str}")
                    print(f"{'':>8} | {'':25} | {'':25} | {control_info}")
                    
                    # 修复采样周期显示 - 使用正确的格式化
                    if control_dt > 0:
                        print(f"{'':>8} | 采样周期: {control_dt*1000:>5.1f}ms")
                    else:
                        print(f"{'':>8} | 采样周期: 等待中...")
                    
                    print("-" * 90)
                    
                    last_display = current_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("用户中断 - 正在停止控制系统...")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cs.stop()
        
        # 显示统计信息
        if cs.log_data:
            print("\n" + "=" * 70)
            print("运行统计:")
            print(f"总运行时间: {time.time() - program_start:.1f}秒")
            print(f"记录数据点: {len(cs.log_data)}")
            
            # 计算平均误差
            if len(cs.pid.error_history) > 0:
                avg_pos_error = np.mean([e['error_pos'] for e in cs.pid.error_history[-100:]], axis=0)
                avg_att_error = np.mean([e['error_att'] for e in cs.pid.error_history[-100:]], axis=0)
                print(f"最近100步平均位置误差: x={avg_pos_error[0]:.3f}m, y={avg_pos_error[1]:.3f}m, z={avg_pos_error[2]:.3f}m")
                print(f"最近100步平均姿态误差: roll={np.rad2deg(avg_att_error[0]):.2f}°, pitch={np.rad2deg(avg_att_error[1]):.2f}°, yaw={np.rad2deg(avg_att_error[2]):.2f}°")
        
        print("=" * 70)
        print("控制系统测试完成")


if __name__ == "__main__":
    main()