# controller.py
import numpy as np
import yaml
from scipy.interpolate import interp1d

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

class ROVConfig:
    """Tether Vehicle 物理与几何配置参数"""
    def __init__(self, config_dict=None):
        cfg = config_dict.get('robot', {}) if config_dict else {}
        
        # 质量与尺寸参数
        self.mass_air = cfg.get('mass_air', 50.0)
        self.mass_submerged = cfg.get('mass_submerged', 1.642)
        dims = cfg.get('dimensions', [0.660, 0.524, 0.410])
        self.length, self.width, self.height = dims
        self.CG = np.array(cfg.get('cg', [-0.129, 0.000, -0.005]))
        
        # 推进器基础配置
        thruster_cfg = cfg.get('thrusters', {})
        self.max_rpm = thruster_cfg.get('max_rpm', 2000)
        self.thruster_deflection_angle = np.deg2rad(thruster_cfg.get('deflection_angle_deg', 30))
        self.n_thrusters = 6
        self.thruster_radius = 0.050
        
        # 推进器安装位置偏移量 (相对于 CG)
        self.Bbp = 0.150  # 后推进器横向偏移
        self.Bfp = 0.150  # 前推进器横向偏移
        self.Btp = 0.165  # 垂向推进器横向偏移
        self.Lbp = 0.250  # 后推进器纵向距离
        self.Lfp = 0.250  # 前推进器纵向距离
        self.Htp = 0.037  # 垂向推进器垂向距离
        self.H0 = 0.100   # CG到中心线的垂向距离
        
        # 初始化计算
        self._calculate_thruster_positions()
        self._calculate_thrust_allocation_matrix()
        self._init_thruster_curves()
        
        # 【新增】硬件符号映射表 (Actuator Sign Map)
        # 索引对应: 0:Pbr, 1:Pbl, 2:Pfr, 3:Pfl, 4:Ptr, 5:Ptl
        # 系数含义: 期望产生正向推力时，RPM 需要乘以的符号
        self.rpm_sign_map = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0])   

    def _calculate_thruster_positions(self):
        """计算每个 thruster 的空间位置和推力方向向量"""
        self.thruster_positions = np.zeros((6, 3))
        # 0: 右后, 1: 左后, 2: 右前, 3: 左前, 4: 右上(垂), 5: 左上(垂)
        self.thruster_positions[0] = [0, -self.Bbp, -self.Lbp]
        self.thruster_positions[1] = [0, self.Bbp, -self.Lbp]
        self.thruster_positions[2] = [0, -self.Bfp, self.Lfp]
        self.thruster_positions[3] = [0, self.Bfp, self.Lfp]
        self.thruster_positions[4] = [-self.Htp, -self.Btp, 0]
        self.thruster_positions[5] = [-self.Htp, self.Btp, 0]
        
        self.thruster_directions = np.zeros((6, 3))
        angle = self.thruster_deflection_angle
        # 水平推进器 (0-3) - 呈 X 型布置
        for i in range(4):
            self.thruster_directions[i] = [0, np.sin(angle), np.cos(angle)]
            if i in [0, 2]:  # 右侧推进器横向分量反号
                self.thruster_directions[i, 1] = -np.sin(angle)
        
        # 垂向推进器 (4-5) - 向下为正
        self.thruster_directions[4] = [1, 0, 0]  
        self.thruster_directions[5] = [1, 0, 0]
        
    def _calculate_thrust_allocation_matrix(self):
        """生成推力分配矩阵 A_matrix (6x6)"""
        self.A_matrix = np.zeros((6, 6))
        for i in range(6):
            # 前三行: 力分量 (X, Y, Z)
            self.A_matrix[0:3, i] = self.thruster_directions[i]
            # 后三行: 力矩分量 (K, M, N) = r × F
            r = self.thruster_positions[i]
            F = self.thruster_directions[i]
            moment = np.cross(r, F)
            self.A_matrix[3:6, i] = moment
            
    def _init_thruster_curves(self):
        """初始化 KA 4-70 敞水曲线的插值模型"""
        self.rho_water = 1000  # kg/m³
        self.D_prop = 2 * self.thruster_radius
        # 进速比 J 与 推力系数 KT 的关系
        self.J_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        self.KT_points = np.array([0.38, 0.35, 0.32, 0.28, 0.24, 0.19, 0.14, 0.08, 0.02])
        self.KT_interp = interp1d(self.J_points, self.KT_points, kind='cubic', fill_value='extrapolate')
        
    def thrust_to_rpm(self, T_desired, Va, thruster_index):
        """逆螺旋桨模型：结合硬件安装方向解算最终指令转速"""
        if abs(T_desired) < 1e-6:
            return 0.0
        
        # 获取推力方向
        sign = np.sign(T_desired)
        T_abs = abs(T_desired)
        
        # 根据 J=0 时的静态推力系数做初始转速猜测
        n_rps = np.sqrt(T_abs / (self.rho_water * self.D_prop**4 * self.KT_points[0]))
        if n_rps < 0.1:
            n_rps = 5.0  
            
        # 松弛迭代法逼近真实转速
        for _ in range(5):
            J = Va / (n_rps * self.D_prop) if n_rps > 0.1 else 0
            J = np.clip(J, 0, 0.8) 
            KT = self.KT_interp(J) 
            n_rps_new = np.sqrt(T_abs / (self.rho_water * self.D_prop**4 * KT)) 
            n_rps = 0.5 * n_rps + 0.5 * n_rps_new 
            
        # 【核心修改】最终转速 = 绝对转速 * 期望推力方向 * 硬件安装映射符号
        final_rpm = n_rps * 60 * sign * self.rpm_sign_map[thruster_index]
        return final_rpm

class ROVPIDController:
    """6自由度位姿 PID 控制器"""
    def __init__(self, rov_config, config_dict=None):
        self.config = rov_config
        
        pid_cfg = config_dict.get('pid', {}) if config_dict else {}
        pos_cfg = pid_cfg.get('position', {})
        att_cfg = pid_cfg.get('attitude', {})
        limits_cfg = pid_cfg.get('limits', {})
        
        # 提取增益参数
        self.Kp_pos = np.array(pos_cfg.get('kp', [50.0, 50.0, 80.0]))
        self.Ki_pos = np.array(pos_cfg.get('ki', [5.0, 5.0, 10.0]))
        self.Kd_pos = np.array(pos_cfg.get('kd', [30.0, 30.0, 40.0]))
        
        self.Kp_att = np.array(att_cfg.get('kp', [20.0, 20.0, 40.0]))
        self.Ki_att = np.array(att_cfg.get('ki', [2.0, 2.0, 5.0]))
        self.Kd_att = np.array(att_cfg.get('kd', [15.0, 15.0, 25.0]))
        
        # 积分限幅器
        pos_int = pos_cfg.get('max_integral', [100.0, 100.0, 200.0])
        att_int = att_cfg.get('max_integral', [50.0, 50.0, 100.0])
        self.max_integral = np.array(pos_int + att_int)
        
        # 广义力输出限制
        self.max_force = limits_cfg.get('max_force', 200.0)
        self.max_moment = limits_cfg.get('max_moment', 50.0)
        
        self.integral_pos = np.zeros(3)
        self.integral_att = np.zeros(3)
        self.last_error_pos = None
        self.last_time = None
        
    def compute_control(self, current_state, desired_state):
        """计算期望广义力 tau = [X, Y, Z, K, M, N]"""
        current_time = current_state['timestamp']
        
        current_pos = np.array([current_state['x'], current_state['y'], current_state['z']])
        current_att = np.array([current_state['roll'], current_state['pitch'], current_state['yaw']])
        current_vel = np.array([current_state['u'], current_state['v'], current_state['w']])
        current_omega = np.array([current_state['p'], current_state['q'], current_state['r']])
        
        desired_pos = np.array([desired_state['x'], desired_state['y'], desired_state['z']])
        desired_att = np.array([desired_state['roll'], desired_state['pitch'], desired_state['yaw']])
        
        # 位置与姿态误差计算
        error_pos = desired_pos - current_pos
        error_att = self._normalize_angle(desired_att - current_att)
        
        # 采用微分先行(速度反馈)策略避免阶跃冲击
        if self.last_error_pos is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                deriv_pos = -current_vel  
                deriv_att = -current_omega
            else:
                deriv_pos = np.zeros(3)
                deriv_att = np.zeros(3)
        else:
            deriv_pos = np.zeros(3)
            deriv_att = np.zeros(3)
            
        # 积分累加与抗饱和 (Anti-windup)
        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                self.integral_pos += error_pos * dt
                self.integral_att += error_att * dt
                self.integral_pos = np.clip(self.integral_pos, -self.max_integral[:3], self.max_integral[:3])
                self.integral_att = np.clip(self.integral_att, -self.max_integral[3:], self.max_integral[3:])
        
        # 核心 PID 计算
        force = (self.Kp_pos * error_pos + self.Ki_pos * self.integral_pos + self.Kd_pos * deriv_pos)
        moment = (self.Kp_att * error_att + self.Ki_att * self.integral_att + self.Kd_att * deriv_att)
        
        # 输出限幅
        force = np.clip(force, -self.max_force, self.max_force)
        moment = np.clip(moment, -self.max_moment, self.max_moment)
        tau = np.concatenate([force, moment])
        
        self.last_error_pos = error_pos
        self.last_time = current_time
        
        return tau
    
    def _normalize_angle(self, angle):
        """将角度规范到 [-pi, pi] 范围内"""
        return (angle + np.pi) % (2 * np.pi) - np.pi


class ThrustAllocator:
    """推力分配器"""
    def __init__(self, rov_config):
        self.config = rov_config
        self.A_matrix = rov_config.A_matrix
        # 计算广义逆矩阵 (Moore-Penrose pseudo-inverse)
        self.A_pinv = np.linalg.pinv(self.A_matrix)
        # 单个推进器物理推力极值
        self.max_thrust_per_prop = 50.0  
        self.min_thrust_per_prop = -30.0  
        
    def allocate(self, tau):
        """通过伪逆法将六自由度广义力分配至六个推进器"""
        thrusts = self.A_pinv @ tau
        thrusts = np.clip(thrusts, self.min_thrust_per_prop, self.max_thrust_per_prop)
        return thrusts


class MotorController:
    """电机动态响应控制器 (线性恒定速率爬坡版)"""
    def __init__(self, rov_config):
        self.config = rov_config
        # 设定的转速最大变化率: 15000 rpm/s (即 0.1秒内提速 1500 rpm)
        self.max_rpm_acceleration = 15000.0  
        self.last_rpm = np.zeros(6)
        self.last_time = None
        
    def thrust_to_rpm(self, thrusts, Va_array):
        """包装调用配置中的推力转 RPM 功能"""
        rpm = np.zeros(6)
        for i in range(6):
            # 【修改】将当前的索引 i 传给配置类，以便提取对应的符号映射
            rpm[i] = self.config.thrust_to_rpm(thrusts[i], Va_array[i], thruster_index=i)
        
        rpm = np.clip(rpm, -self.config.max_rpm, self.config.max_rpm)
        return rpm
    
    def apply_motor_dynamics(self, desired_rpm, current_sim_time):
        """引入线性速率限制器 (Rate Limiter) 以模拟恒定爬坡"""
        if self.last_time is None:
            self.last_time = current_sim_time
            self.last_rpm = desired_rpm
            return desired_rpm
        
        dt = current_sim_time - self.last_time
        if dt <= 0: return self.last_rpm

        # 1. 计算当前时间步长 (dt) 内允许的最大转速变化量绝对值
        max_delta = self.max_rpm_acceleration * dt
        
        # 2. 计算期望转速与当前真实转速的差值
        diff = desired_rpm - self.last_rpm
        
        # 3. 将差值强制截断在允许的最大变化量范围内
        # 如果差值很大，本步只增加 max_delta；如果差值很小，就直接补齐差值达到目标
        actual_delta = np.clip(diff, -max_delta, max_delta)
        
        # 4. 获得真实的瞬间转速
        actual_rpm = self.last_rpm + actual_delta
        
        self.last_rpm = actual_rpm
        self.last_time = current_sim_time
        return actual_rpm


class ROVControlSystem:
    """
    Tether Vehicle 控制系统核心枢纽
    实现多速率联合仿真(Multi-rate Co-simulation)架构
    """
    def __init__(self, cosim_interface, config_file='rov_config.yaml'):
        # 依赖注入：挂载外部提供的内存/通信接口
        self.cosim = cosim_interface
        
        # 初始化控制组件
        self.raw_config = ConfigLoader.load(config_file)
        self.config = ROVConfig(self.raw_config)
        self.pid = ROVPIDController(self.config, self.raw_config)
        self.allocator = ThrustAllocator(self.config)
        self.motor = MotorController(self.config)
        
        # 期望保持的默认位姿
        self.desired_state = {
            'x': 0.0, 'y': 0.0, 'z': -5.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        
        # --- 多速率架构配置 ---
        self.control_rate = 50.0  # 真实物理控制器的运算频率 (50Hz)
        self.control_dt = 1.0 / self.control_rate # 0.02秒算一次 PID
        
        # 初始化为负数，保证在仿真启动瞬间(t=0)必须执行首次 PID 计算
        self.last_control_time = -self.control_dt 
        
        # 内部缓存信号，供高频电机环使用
        self.current_target_rpm = np.zeros(6) 
        self.current_tau = np.zeros(6)

    def set_desired_position(self, x, y, z):
        self.desired_state['x'] = x
        self.desired_state['y'] = y
        self.desired_state['z'] = z
        
    def set_desired_attitude(self, roll, pitch, yaw):
        self.desired_state['roll'] = roll
        self.desired_state['pitch'] = pitch
        self.desired_state['yaw'] = yaw

    def step(self):
        """
        执行联合仿真的核心单步步进逻辑。
        由外界的主循环(main.py)无脑高频调用即可。
        """
        # 1. 阻塞等待：直到获取到最新的高频物理流场状态
        raw_state = self.cosim.wait_for_cfd_data()
        # current_state = self.cosim.wait_for_cfd_data()

        # 杆臂效应修正：将局部坐标系原点速度平移至真实的重心 (CG)
        v_Ob = np.array([raw_state['u'], raw_state['v'], raw_state['w']])
        omega = np.array([raw_state['p'], raw_state['q'], raw_state['r']])
        
        # 从随体坐标系原点 (Ob) 指向重心 (CG) 的向量
        r_Ob_to_CG = np.array([-0.1, 0.0, -0.005])
        
        # 刚体运动学速度平移公式: V_cg = V_ob + omega × r
        v_CG = v_Ob + np.cross(omega, r_Ob_to_CG)

        # 坐标系拦截器 (CFD Frame -> Standard NED Frame)
        # CFD Frame: X=下, Y=右, Z=前
        # NED Frame: X=前, Y=右, Z=下
        current_state = {}
        current_state['timestamp'] = raw_state['timestamp']
        
        # 位置与线速度映射 (X 与 Z 对调)
        current_state['x'] = raw_state['z']  # 将 CFD 的前(Z) 映射给 标准的前(X)
        current_state['y'] = raw_state['y']  # 右(Y) 不变
        current_state['z'] = raw_state['x']  # 将 CFD 的下(X) 映射给 标准的下(Z)
        
        current_state['u'] = v_CG[2]         # 标准前进速度 u 取自 CFD 的 w
        current_state['v'] = v_CG[1]         # 标准横移速度 v 取自 CFD 的 v
        current_state['w'] = v_CG[0]         # 标准下潜速度 w 取自 CFD 的 u
        
        # 姿态与角速度映射 (修正 Java 宏中的名称错位)
        # 记住：Java里 roll=rx(下), pitch=ry(右), yaw=rz(前)
        current_state['roll']  = raw_state['yaw']   # 绕前方轴旋转才是真 Roll
        current_state['pitch'] = raw_state['pitch'] # 绕右方轴旋转是真 Pitch
        current_state['yaw']   = raw_state['roll']  # 绕下方轴旋转才是真 Yaw
        
        current_state['p'] = raw_state['r']  # 真 Roll_rate
        current_state['q'] = raw_state['q']  # 真 Pitch_rate
        current_state['r'] = raw_state['p']  # 真 Yaw_rate
        
        current_sim_time = current_state['timestamp']
        
        # ==========================================
        # 离散低频环 (50Hz): 真实控制算法域
        # ==========================================
        if current_sim_time - self.last_control_time >= (self.control_dt - 1e-6):
            # A. 运算 PID 误差，获得期望广义力
            self.current_tau = self.pid.compute_control(current_state, self.desired_state)
            # B. 推力分配至 6 个执行器
            thrusts = self.allocator.allocate(self.current_tau)
            # C. 结合局部进速计算，解算期望转速
            Va_array = self._compute_advance_velocities(current_state)
            self.current_target_rpm = self.motor.thrust_to_rpm(thrusts, Va_array)
            
            self.last_control_time = current_sim_time

        # ==========================================
        # 连续高频环 (与 CFD 同频, 如1000Hz): 物理延迟域
        # ==========================================
        # 无论当前是否执行了 PID，电机都在随着时间常数拼命向 target_rpm 逼近
        actual_rpm = self.motor.apply_motor_dynamics(self.current_target_rpm, current_sim_time)
        
        # 3. 将真实产生的瞬态转速送回共享内存，触发 CFD 步进
        self.cosim.send_motor_commands(actual_rpm)
        
        # 4. 向上层返回核心数据用于日志和终端打印
        return current_sim_time, current_state, self.current_tau, self.current_target_rpm, actual_rpm

    def _compute_advance_velocities(self, state):
        """计算各推进器盘面处的局部进速（Advance Velocity）"""
        u, v, w = state['u'], state['v'], state['w']
        p, q, r = state['p'], state['q'], state['r']
        Va_array = np.zeros(6)
        
        for i in range(6):
            # 提取安装位置和偏转方向
            pos = self.config.thruster_positions[i]
            direction = self.config.thruster_directions[i]
            
            # 计算因本体旋转而引起的盘面线速度增量 v = ω × r
            rot_vel = np.cross([p, q, r], pos)
            
            # 盘面处相对于周围水流的绝对总合速度
            total_vel = np.array([u, v, w]) + rot_vel
            
            # 将总合速度投影到推进器的主推力轴线上，得到有效进速
            Va_array[i] = np.dot(total_vel, direction)
            
        return Va_array
