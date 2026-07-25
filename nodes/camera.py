# rpi5_detect/nodes/camera.py
import cv2
import time
from picamera2 import Picamera2
from config import FRAME_WIDTH, FRAME_HEIGHT

def camera_worker(frame_queue, stop_event):
    """
    摄像头采集节点 (Camera Node) — 直连 Picamera2，不再走 UDP 桥接
    前提：树莓派的系统 Python 3.13 已安装 picamera2 + ncnn + opencv 等全部依赖，
          无需再分两个 Python 环境通信。
    """
    print("--- 摄像头节点已启动 (直连 Picamera2) ---")

    # 1. 初始化 Picamera2
    picam2 = Picamera2(camera_num=0)
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    try:
        while not stop_event.is_set():
            # 2. 直接获取 RGB 帧并转为 BGR（省去了 JPEG 编解码的 CPU 开销）
            frame_rgb = picam2.capture_array()
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)  # 旋转 180 度，因为摄像头装反了
            #frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 3. 队列维护：保持旧有的丢帧逻辑，确保推理节点永远消费最新帧
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except:
                    pass

            #frame_queue.put(frame_bgr)#改前发现画面红蓝颠倒
            frame_queue.put(frame_rgb)

    except Exception as e:
        print(f"摄像头节点异常: {e}")
    finally:
        picam2.stop()
        print("--- 摄像头节点已安全停止 ---")
