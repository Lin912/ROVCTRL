import star.common.*;
import java.io.*;
import java.util.*;
import star.base.neo.*;
import star.sixdof.*;
import star.base.report.*;

import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteOrder;

import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.Condition;

public class FileConditioned extends StarMacro {
    private static final String FILE_SHARED = "../HydroSimulation/ControlDirect_SharedMemory";
    private static final int OFFSET_PROGRAM_STARCCM = 0;
    private static final int OFFSET_PROGRAM_ROVCTRL = 4;
    private static final int BUFFER_SIZE = 1024 + 8; //缓冲区
    private static final int OFFSET_DATA_START = 8;  //配置数据起始点

    @Override
    public void execute() {
        try{
            Simulation simulation = getActiveSimulation();
            
            RandomAccessFile file = new RandomAccessFile(FILE_SHARED, "rw");
            FileChannel channel = file.getChannel();
            MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_WRITE,0 ,BUFFER_SIZE);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            while(true){
                
                    while(buffer.getInt(OFFSET_PROGRAM_STARCCM) == 1){
                        simulation.println("[STAR CCM+] Resuming Simulation");
                        
                        // =========================================================
                        // 1. 极速读取外部数据 (直接从物理内存中按二进制读取，0 I/O 延迟)
                        // 0 ~ 3 STARCCM Flag (int)
                        // 4 ~ 7 CITRINE Flag (int)
                        // =========================================================
                        int OFFSET_RPM_IN = OFFSET_DATA_START + (13 * 8); // 螺旋桨转速读取位置 (13个状态量-[1时间戳-3位置量-3姿态角-3线速度-3角速度]
                        int OFFSET_FORCE_IN = OFFSET_RPM_IN + (6 * 8);    // 缆绳力读取位置 (6个螺旋桨转速)

                        // 读取 6 个螺旋桨的转速 (RPM)
                        double[] rpms = new double[6];
                        for (int i = 0; i < 6; i++) {
                            rpms[i] = buffer.getDouble(OFFSET_RPM_IN + (i * 8));
                        }
                        
                        // 将 RPM 应用到螺旋桨 (使用全局参数 Global Parameter)
                        // STAR-CCM+ 界面中创建 6 个标量全局参数 (Global Parameter)，例如命名为 "RPM_0" 到 "RPM_5"
                        // 旋转坐标系的转速链接到这些参数上
                        try {
                            for (int i = 0; i < 6; i++) {
                                String paramName = "RPM_" + i;
                                ScalarGlobalParameter rpmParam = (ScalarGlobalParameter) simulation.get(GlobalParameterManager.class).getObject(paramName);
                                
                                // 将转速设置进去，注意 STAR-CCM+ 里默认角速度单位可能是 rad/s
                                double radPerSec = rpms[i] * (Math.PI / 30.0);
                                rpmParam.getQuantity().setValue(radPerSec); 
                            }
                        } catch (Exception e) {
                            simulation.println("警告: 未找到对应的 RPM 全局参数，请在软件中创建 RPM_0 ~ RPM_5");
                        }


                        // 读取 Tethra 计算出的绳缆力 (X, Y, Z)
                        // double forceX = buffer.getDouble(OFFSET_FORCE_IN);
                        // double forceY = buffer.getDouble(OFFSET_FORCE_IN + 8);
                        // double forceZ = buffer.getDouble(OFFSET_FORCE_IN + 16); 
                        // ContinuumBody continuumBody = ((ContinuumBody) simulation.get(star.sixdof.BodyManager.class).getObject("MainBody"));//Body's Name
                        // ExternalForce externalForce = ((ExternalForce) continuumBody.getExternalForceAndMomentManager().getObject("CableForce"));//Force's Name              
                        // Units units_N = ((Units) simulation.getUnitsManager().getObject("N"));
                        // externalForce.getForce().setComponentsAndUnits(-forceX, -forceY, -forceZ, units_N);//Force value
                        // Units units_1 = ((Units) simulation.getUnitsManager().getObject("m"));
                        // externalForce.getPositionAsCoordinate().setCoordinate(units_1, units_1, units_1, new DoubleVector(new double[] {-0.22, 0.0, -0.00497}));//Force acting point  

                        //Running simulation
                        int t0 = simulation.getSimulationIterator().getCurrentIteration();

                        simulation.getSimulationIterator().step(1);
                        
                        while(simulation.getSimulationIterator().getCurrentIteration() <= t0){
                            try{
                                Thread.sleep(50);
                            }catch(InterruptedException e){
                                simulation.println("Interrupted while waiting for step: " + e.getMessage());
                            }
                        }
                        
                        // 提取位姿数据并直接写入 MappedByteBuffer
                        try {
                            // 1. 获取当前物理时间 (Timestamp)
                            double timestamp = simulation.getSolution().getPhysicalTime();

                            // 2. 获取位置 x, y, z
                            // 如果你没有创建，请在 STAR-CCM+ 的 Reports 节点下创建它们，监控 MainBody 的平移
                            double x = ((Report) simulation.getReportManager().getReport("DisX")).getReportMonitorValue();
                            double y = ((Report) simulation.getReportManager().getReport("DisY")).getReportMonitorValue();
                            double z = ((Report) simulation.getReportManager().getReport("DisZ")).getReportMonitorValue();

                            // 3. 获取姿态角 (EulerAngle)
                            double roll  = ((Report) simulation.getReportManager().getReport("rx")).getReportMonitorValue();
                            double pitch = ((Report) simulation.getReportManager().getReport("ry")).getReportMonitorValue();
                            double yaw   = ((Report) simulation.getReportManager().getReport("rz")).getReportMonitorValue();

                            // 4. 获取相对线速度 (VelocityRelative)
                            double u = ((Report) simulation.getReportManager().getReport("Vrx")).getReportMonitorValue();
                            double v = ((Report) simulation.getReportManager().getReport("Vry")).getReportMonitorValue();
                            double w = ((Report) simulation.getReportManager().getReport("Vrz")).getReportMonitorValue();

                            // 5. 获取相对角速度 (omegaRelative)
                            double p = ((Report) simulation.getReportManager().getReport("omegarx")).getReportMonitorValue();
                            double q = ((Report) simulation.getReportManager().getReport("omegary")).getReportMonitorValue();
                            double r = ((Report) simulation.getReportManager().getReport("omegarz")).getReportMonitorValue();

                            // =========================================================
                            // 严格按照 Python 端 unpack('13d') 的顺序，写入连续的 104 字节
                            // =========================================================
                            buffer.putDouble(OFFSET_DATA_START, timestamp);
                            buffer.putDouble(OFFSET_DATA_START + 8, x);
                            buffer.putDouble(OFFSET_DATA_START + 16, y);
                            buffer.putDouble(OFFSET_DATA_START + 24, z);
    
                            buffer.putDouble(OFFSET_DATA_START + 32, roll);
                            buffer.putDouble(OFFSET_DATA_START + 40, pitch);
                            buffer.putDouble(OFFSET_DATA_START + 48, yaw);
    
                            buffer.putDouble(OFFSET_DATA_START + 56, u);
                            buffer.putDouble(OFFSET_DATA_START + 64, v);
                            buffer.putDouble(OFFSET_DATA_START + 72, w);
    
                            buffer.putDouble(OFFSET_DATA_START + 80, p);
                            buffer.putDouble(OFFSET_DATA_START + 88, q);
                            buffer.putDouble(OFFSET_DATA_START + 96, r);

                        } catch (Exception e) {
                            simulation.println("写入共享内存失败，请检查 Report 名称是否正确: " + e.getMessage());
                        }

			// simulation.saveState("star.sim");
                        simulation.println("Step Completed");

                        buffer.putInt(OFFSET_PROGRAM_ROVCTRL, 1);
                        buffer.putInt(OFFSET_PROGRAM_STARCCM, 0);
                        buffer.force();                            
                    }

                    try{
                        Thread.sleep(50); // 轮询延迟
                    } catch(InterruptedException e){
                        simulation.println("Thread interrupted: " + e.getMessage());
                    }
 			                       
                    simulation.println("Printing Mapped File Content");
                    MappedByteBuffer readBuffer = buffer.duplicate();
                    readBuffer.position(0);
                    int offsetStarCCM = readBuffer.getInt(0);
                    int offsetCitrine = readBuffer.getInt(4);
                        
                    simulation.println("OFFSET_PROGRAM_STARCCM" + offsetStarCCM);
                    simulation.println("OFFSET_PROGRAM_ROVCTRL" + offsetCitrine);
                    simulation.println("[STAR CCM+] Pausing Simulation");
                    // simulation.getSimulationIterator().stop();
            }
        }catch(IOException e){
            e.printStackTrace();
        }
    }
}

