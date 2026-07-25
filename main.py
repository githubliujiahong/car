# rpi5_detect/main.py
import time
from multiprocessing import Process, Queue, Event
from nodes.camera import camera_worker
from nodes.inference import inference_worker
from nodes.logic_test import logic_test_worker
from nodes.recorder import recorder_worker
from config import ENABLE_RECORDER

def main():
    # ==========================================
    # 1. 初始化跨进程队列 (IPC Queues)
    # ==========================================
    # 限制 maxsize 以防堆积导致延迟，始终获取最新数据
    frame_q = Queue(maxsize=2)      # 摄像头 -> 推理
    result_q = Queue(maxsize=2)     # 推理 -> 逻辑
    recorder_q = Queue(maxsize=2) if ENABLE_RECORDER else None # 推理 -> 录制
    state_q = Queue(maxsize=1)      # 逻辑 -> 录制 (状态名)

    # 系统停止事件
    stop_event = Event()

    # ==========================================
    # 2. 进程拓扑定义 (Process Topology)
    # ==========================================
    processes = [
        # 摄像头节点：负责视频流采集
        Process(target=camera_worker, args=(frame_q, stop_event), name="CameraNode"),
        # 推理节点：负责 AI 模型检测 (同时直连录制队列，绕过逻辑的1秒瓶颈)
        Process(target=inference_worker, args=(frame_q, result_q, recorder_q, stop_event), name="InferenceNode"),
        # 逻辑节点：执行状态机、PID控制、串口通信
        Process(target=logic_test_worker, args=(result_q, recorder_q, stop_event, state_q), name="LogicNode")
    ]
    
    # 如果开启录制，则添加录制节点
    if ENABLE_RECORDER:
        processes.append(Process(target=recorder_worker, args=(recorder_q, stop_event, state_q), name="RecorderNode"))

    # ==========================================
    # 3. 启动所有进程 (System Start)
    # ==========================================
    print("--- 正在启动 RPi5 目标检测系统 (多进程竞赛模式) ---")
    for p in processes:
        p.daemon = True # 设置为守护进程，主进程退出时子进程也退出
        p.start()

    try:
        # 主循环：监控退出信号
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n检测到用户中断，正在关闭系统...")
        stop_event.set()

    # ==========================================
    # 4. 资源清理 (Cleanup)
    # ==========================================
    for p in processes:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate() # 强制关闭未及时退出的进程
            
    print("--- 系统已安全关闭 ---")

if __name__ == "__main__":
    main()
