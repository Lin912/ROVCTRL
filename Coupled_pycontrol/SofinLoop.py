# SofinLoop.py
import mmap
import os
import struct
import time

class CoSimInterface:
    """
    与 CFD 软件耦合的共享内存接口 (基于 Linux mmap)
    根据 C++ 结构体:
    struct ControlDirect {
        int OFFSET_PROGRAM_STARCCM; // Offset: 0 (4 bytes)
        int OFFSET_PROGRAM_ROVCTRL; // Offset: 4 (4 bytes)
        char data[1024];            // Offset: 8 (1024 bytes)
    };
    """
    def __init__(self, use_dummy=False, filename="ControlDirect_SharedMemory"):
        self.use_dummy = use_dummy
        self.filename = filename
        
        # 字节偏移量定义
        self.OFFSET_STARCCM = 0
        self.OFFSET_CITRINE = 4
        self.OFFSET_DATA_START = 8
        
        # 假设 CFD 传来的状态是 13 个 double (104字节)，写在 data 的开头
        # 假设控制模块算出的转速是 6 个 double (48字节)，写在状态数据的后面
        self.READ_OFFSET = self.OFFSET_DATA_START
        self.WRITE_OFFSET = self.OFFSET_DATA_START + 104 
        
        self.mm = None
        
        if not self.use_dummy:
            print(f"正在连接共享内存文件: {self.filename}")
            # 以读写模式打开文件 (不需要 O_CREAT, 因为 C++ Creator 已经创建)
            try:
                fd = os.open(self.filename, os.O_RDWR)
                # 总大小为 4 + 4 + 1024 = 1032 字节
                self.mm = mmap.mmap(fd, 1032, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            except FileNotFoundError:
                raise FileNotFoundError(f"找不到内存映射文件 {self.filename}，请确保 MMF_Creator.cpp 已运行。")
        else:
            # 测试模式变量
            self.sim_time = 0.0
            self.dt = 0.001
            self.dummy_state = {
                'timestamp': 0.0, 'x': 0.0, 'y': 0.0, 'z': -5.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'u': 0.35, 'v': 0.0, 'w': 0.0,
                'p': 0.0, 'q': 0.0, 'r': 0.0
            }

    def wait_for_cfd_data(self):
        """阻塞等待 CITRINE 标志位置 1, 读取状态数据"""
        if self.use_dummy:
            self.sim_time += self.dt
            self.dummy_state['timestamp'] = self.sim_time
            self.dummy_state['x'] += self.dummy_state['u'] * self.dt
            time.sleep(0.0001)
            return self.dummy_state.copy()
            
        while True:
            # 读取 CITRINE 的轮询标志位
            self.mm.seek(self.OFFSET_CITRINE)
            citrine_turn = struct.unpack('i', self.mm.read(4))[0]
            
            if citrine_turn == 1:
                # 轮到控制模块计算，读取 CFD 写入的 13 个 double 数据
                self.mm.seek(self.READ_OFFSET)
                raw_data = self.mm.read(104) # 13 * 8 字节
                unpacked = struct.unpack('13d', raw_data)
                
                state = {
                    'timestamp': unpacked[0],
                    'x': unpacked[1], 'y': unpacked[2], 'z': unpacked[3],
                    'roll': unpacked[4], 'pitch': unpacked[5], 'yaw': unpacked[6],
                    'u': unpacked[7], 'v': unpacked[8], 'w': unpacked[9],
                    'p': unpacked[10], 'q': unpacked[11], 'r': unpacked[12]
                }
                return state
                
            # 自旋锁，休眠极短时间防止吃满 CPU
            time.sleep(0.0001)

    def send_motor_commands(self, rpm_array):
        """将转速写入内存，并将控制权交还给 STAR-CCM+"""
        if self.use_dummy:
            return
            
        # 1. 打包 6 个转速数据 (6 * 8 = 48 字节)
        rpm_bytes = struct.pack('6d', *rpm_array)
        
        # 2. 写入指定的 data 偏移位置
        self.mm.seek(self.WRITE_OFFSET)
        self.mm.write(rpm_bytes)
        
        # 3. 翻转标志位：CITRINE = 0, STARCCM = 1 (三方握手)
        self.mm.seek(self.OFFSET_CITRINE)
        self.mm.write(struct.pack('i', 0))
        
        self.mm.seek(self.OFFSET_STARCCM)
        self.mm.write(struct.pack('i', 1))

    def close(self):
        if self.mm is not None:
            self.mm.close()