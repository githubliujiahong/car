# rpi5_detect/nodes/logic_v3.py
"""
逻辑节点 v3 — 基于像素偏移的比例旋转控制。

流程:
  1. 等待画面中出现己色球 (取最新一帧)
  2. 计算球中心 x 与画面中线的像素偏移
  3. 偏移量 × 比例系数 → 旋转时长
  4. 旋转 → 停止 → 退出

旋转速度依赖硬件: 3 秒 / 180 度
"""

import serial
import struct
import time
from config import *


# ==========================================
# 目标筛选 (从 v2 移植)
# ==========================================

def classify_detections(det_results):
    """将检测结果分为 球 (cls 0-3) 和 安全区 (cls 4-5)"""
    balls = []
    for res in det_results:
        cls_id = res["class_id"]
        if cls_id in (0, 1, 2, 3):
            balls.append(res)
    return balls


def select_target(balls, own_color):
    """
    优先己色球，选最近的 (y2 最大)。
    返回选中的球 dict 或 None。
    """
    if not balls:
        return None
    own_balls = [b for b in balls if b["class_id"] == own_color]
    candidates = own_balls if own_balls else balls
    return max(candidates, key=lambda b: b["box"][3])


# ==========================================
# 串口通信 (从 v2 移植)
# ==========================================

def send_command(ser, cmd):
    """发送单字节控制指令"""
    if ser is not None:
        packet = struct.pack('<BBBB', 0xAA, 0x02, cmd, 0x55)
        ser.write(packet)


# ==========================================
# v3 旋转控制参数 (调参改这里)
# ==========================================

# 像素死区: 球中心偏离中线小于此值视为已对准，不旋转
PIXEL_DEADZONE = 20

# 像素 → 秒 比例系数
# 旋转速度 3s/180° ≈ 0.0167 s/度
# 画面约 820px 覆盖 ~95° FOV → 约 8.6 px/度 → 约 0.00194 s/px
# 实际取 0.003，留有裕量，可根据实测调整
PIXEL_TO_SECOND = 0.0028


# ==========================================
# 主循环
# ==========================================

def logic_v3_worker(result_queue, recorder_queue, stop_event):
    """
    逻辑节点 v3 — 一次定位 + 像素偏移旋转。
    """
    # ---- 队色 ----
    own_color = RED if if_red else BLUE
    color_name = "红色" if if_red else "蓝色"

    # ---- 串口 ----
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        print(f"[v3 串口] 成功打开 {SERIAL_PORT}")
    except Exception as e:
        print(f"[v3 串口] 无法打开串口 → {e}")
        ser = None

    frame_cx = FRAME_WIDTH / 2.0

    print(f"\n--- 逻辑节点 v3 启动 | 队色: {color_name} | "
          f"死区: ±{PIXEL_DEADZONE}px | 系数: {PIXEL_TO_SECOND}s/px ---")

    try:
        # ==========================================
        # 阶段 1: 等待发现己色球
        # ==========================================
        target = None

        while not stop_event.is_set():
            if result_queue.empty():
                time.sleep(0.01)
                continue

            # 取最新帧
            data = result_queue.get()
            while not result_queue.empty():
                data = result_queue.get()
            frame, det_results = data

            balls = classify_detections(det_results)
            target = select_target(balls, own_color)

            if target is not None:
                break

        if stop_event.is_set():
            send_command(ser, CMD_STILL)
            return

        if target is None:
            print("[v3] 未发现目标, 退出")
            send_command(ser, CMD_STILL)
            return

        # ==========================================
        # 阶段 2: 计算像素偏移 → 旋转时长
        # ==========================================
        cx = (target["box"][0] + target["box"][2]) / 2.0
        offset = cx - frame_cx  # 正 = 偏右, 负 = 偏左

        print(f"[v3] 发现目标 cls={target['class_id']} conf={target['confidence']:.2f} "
              f"cx={cx:.0f} 偏移={offset:+.0f}px")

        if abs(offset) <= PIXEL_DEADZONE:
            print(f"[v3] 已对准 (偏移={offset:+.0f}px ≤ ±{PIXEL_DEADZONE}px), 无需旋转")
            send_command(ser, CMD_STILL)
            return

        duration = abs(offset) * PIXEL_TO_SECOND
        direction = "右" if offset > 0 else "左"

        print(f"[v3] 偏移={offset:+.0f}px → {direction}转 {duration:.2f}s")

        # ==========================================
        # 阶段 3: 执行旋转
        # ==========================================
        cmd = CMD_TURN_RIGHT if offset > 0 else CMD_TURN_LEFT
        send_command(ser, cmd)

        rotate_start = time.time()
        while time.time() - rotate_start < duration:
            if stop_event.is_set():
                break
            time.sleep(0.02)

        # ==========================================
        # 阶段 4: 停止
        # ==========================================
        send_command(ser, CMD_STILL)
        print("[v3] 旋转完成, 已停止")

    except Exception as e:
        print(f"[v3 崩溃] 主循环异常: {e}")
        import traceback
        traceback.print_exc()

    # ---- 安全退出 ----
    if ser is not None:
        send_command(ser, CMD_STILL)
        ser.close()
    print("--- 逻辑节点 v3 已停止 ---")
