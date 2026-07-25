# rpi5_detect/nodes/logic_v2.py
"""
比赛级逻辑节点 — 状态机驱动，纯视觉伺服。

状态流转:
  INIT → FIND_TARGET(一直左转) → CALIBRATE → APPROACH_TARGET(连续前进)
       → [每+30y2: 增量校准] → GRAB
       → FIND_SAFE_ZONE(一直左转) → ALIGN_SAFE_ZONE → APPROACH_SAFE_ZONE → RELEASE
       → FIND_TARGET (循环)

首球规则: 必须且仅运1个己色球。之后优先级: 黄 > 黑 > 己色。
执行节奏: 动作 LOGIC_ACTION_DURATION 秒 → 停止 LOGIC_STOP_DURATION 秒交替。
"""

import serial
import struct
import time
import math
import cv2
import numpy as np
from config import *


# ==========================================
# 几何工具函数
# ==========================================

def point_in_quad(px, py, quad):
    """
    射线法判断点 (px, py) 是否在四边形 quad 内。
    quad: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]  4个角点 (仅使用 conf>0 的点)
    """
    pts = [(p[0], p[1]) for p in quad if p[2] > 0.1]
    if len(pts) < 3:
        return False
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
        j = i
    return inside


def ball_bottom_center(box):
    """球的底部中心坐标 (cx, y2) — 球触地点"""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def ball_in_quad(ball, quad):
    """球的底部中心是否在四边形内"""
    bcx, bcy = ball_bottom_center(ball["box"])
    return point_in_quad(bcx, bcy, quad)


def ball_in_any_safe_zone(ball, safe_zones):
    """球是否在任何已检测到的安全区内"""
    for sz in safe_zones:
        kpts = sz.get("keypoints", [])
        if len(kpts) >= 4 and ball_in_quad(ball, kpts):
            return True
    return False


def safe_zone_front_midpoint_x(sz, front_indices):
    """安全区入口正面两点的中点 x 坐标"""
    kpts = sz.get("keypoints", [])
    if len(kpts) < 4:
        return None
    i1, i2 = front_indices
    p1, p2 = kpts[i1], kpts[i2]
    if p1[2] < 0.1 or p2[2] < 0.1:
        return None
    return (p1[0] + p2[0]) / 2.0


def safe_zone_front_mean_y(sz, front_indices):
    """安全区入口正面两点的平均 y 坐标 (值越大越近)"""
    kpts = sz.get("keypoints", [])
    if len(kpts) < 4:
        return None
    i1, i2 = front_indices
    p1, p2 = kpts[i1], kpts[i2]
    if p1[2] < 0.1 or p2[2] < 0.1:
        return None
    return (p1[1] + p2[1]) / 2.0


# ==========================================
# 目标筛选
# ==========================================

def classify_detections(det_results):
    """将检测结果分为 球 (cls 0-3) 和 安全区 (cls 4-5)"""
    balls = []
    safe_zones = []
    for res in det_results:
        cls_id = res["class_id"]
        if cls_id in (0, 1, 2, 3):
            balls.append(res)
        elif cls_id in (4, 5):
            safe_zones.append(res)
    return balls, safe_zones


def filter_active_balls(balls, safe_zones):
    """过滤掉已在安全区内的球"""
    active = []
    for b in balls:
        if not ball_in_any_safe_zone(b, safe_zones):
            active.append(b)
    return active


def is_ball_in_grab_zone(box):
    """球的底部中心 Y 是否进入抓取区域 (仅判断 Y，不判断 X)"""
    cx, y2 = ball_bottom_center(box)
    return GRAB_ZONE_Y1 <= y2 <= GRAB_ZONE_Y2


def select_target(active_balls, own_color, first_done):
    """
    按优先级选择目标球。
    首球未运: 只选己色 (own_color)。
    首球已运: 黄 (3) > 黑 (0) > 己色 (own_color)。
    返回选中的球 dict, 或 None。
    """
    if not active_balls:
        return None

    if not first_done:
        # 首球: 只运己色普通球, 且只能选1个
        own_balls = [b for b in active_balls if b["class_id"] == own_color]
        if not own_balls:
            return None
        # 选最近的 (y2 最大)
        return max(own_balls, key=lambda b: b["box"][3])

    # 首球已运: 黄 > 黑 > 己色
    for priority_cls in (3, 0, own_color):
        candidates = [b for b in active_balls if b["class_id"] == priority_cls]
        if candidates:
            return max(candidates, key=lambda b: b["box"][3])

    return None


def find_own_safe_zone(safe_zones, own_sz_cls):
    """找到本方安全区 (cls 匹配), 返回置信度最高的"""
    candidates = [sz for sz in safe_zones if sz["class_id"] == own_sz_cls]
    if not candidates:
        return None
    return max(candidates, key=lambda sz: sz["confidence"])


# ==========================================
# 串口通信
# ==========================================

def send_command(ser, cmd):
    """发送单字节控制指令"""
    if ser is not None:
        packet = struct.pack('<BBBB', 0xAA, 0x02, cmd, 0x55)
        ser.write(packet)


# ==========================================
# 坐标校准 (基于世界坐标，一次精确旋转)
# ==========================================

# 旋转速度: 3 秒 / 180 度 → 60 度/秒
ROTATION_DEG_PER_SEC = 180.0 / 3.3  # ~54.5°/s

# 角度死区: 目标偏离正前方小于此角度则不旋转
ANGLE_DEADZONE_DEG = 2.0


def compute_target_angle(world_bottom):
    """
    根据目标的世界坐标计算相对于小车正前方的角度。
    X: 水平 (右为正), Y: 深度 (前为正)
    正角度 = 目标在右侧 → 需右转；负角度 = 需左转。
    """
    if world_bottom is None:
        return None
    wx, wy = world_bottom
    if wy <= 0.001:
        return None
    return math.degrees(math.atan2(wx, wy))


def compute_rotation_duration(angle_deg):
    """角度 → 旋转时长 (秒)，旋转速度 60°/s"""
    return abs(angle_deg) / ROTATION_DEG_PER_SEC


# ==========================================
# 状态机主循环
# ==========================================

def logic_v2_worker(result_queue, recorder_queue, stop_event, state_queue=None):
    """
    比赛决策主循环 — 纯视觉伺服装状态机。
    执行节奏: 动作 0.2s → 停止 0.2s 交替。
    """

    # ---- 队色与目标类别 ----
    own_color = RED if if_red else BLUE
    own_sz_cls = RED_SAFE_ZONE if if_red else BLUE_SAFE_ZONE
    color_name = "红色" if if_red else "蓝色"

    # ---- 串口 ----
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        print(f"[串口] 成功打开 {SERIAL_PORT}")
    except Exception as e:
        print(f"[串口] 无法打开串口 → {e}")
        ser = None

    print(f"\n--- 比赛逻辑 v2 启动 | 队色: {color_name} ---")

    # ==========================================
    # 持久状态 (跨循环保持)
    # ==========================================
    first_done = False        # 首球是否已运抵安全区
    release_step = 0          # 0=舵机上抬, 1..N=后退计数
    search_dir = 1            # 1=左转搜索, -1=右转搜索
    search_cycles = 0         # 当前方向已搜索周期数
    stuck_counter = 0         # 同状态卡住计数
    calibrated = False        # 是否已完成首次坐标校准
    last_cal_y2 = -1.0        # 上次增量校准时的 y2 值 (-1 表示未初始化)

    state_name = "INIT"
    prev_state = None

    # ---- 本轮选中的目标/安全区 (每帧重新选取) ----
    current_target = None
    current_safe_zone = None

    try:
        while not stop_event.is_set():

            # ======================================
            # 1. 获取帧 + 清空积压 (逐帧检查, 不放过任何命中帧)
            # ======================================
            if result_queue.empty():
                time.sleep(0.01)
                continue

            while not result_queue.empty():
                data = result_queue.get()
                frame, det_results = data
                balls, safe_zones = classify_detections(det_results)
                active_balls = filter_active_balls(balls, safe_zones)
                own_sz = find_own_safe_zone(safe_zones, own_sz_cls)
                if own_sz is not None:
                    current_safe_zone = own_sz
                target = select_target(active_balls, own_color, first_done)
                if target is not None:
                    current_target = target

            # ======================================
            # 4. 状态机 — 根据 state_name 调度
            # ======================================
            cmd = CMD_STILL
            log_line = ""

            # ---------- INIT: 离开出发区 ----------
            if state_name == "INIT":
                if ENABLE_INIT_FORWARD:
                    print(f"[INIT] 出发区直行 {INIT_FORWARD_SECONDS:.0f} 秒...")
                    send_command(ser, CMD_FORWARD)
                    init_start = time.time()
                    while time.time() - init_start < INIT_FORWARD_SECONDS:
                        if stop_event.is_set():
                            break
                        time.sleep(0.02)
                    send_command(ser, CMD_STILL)
                    log_line = "离开出发区 → 开始寻球"
                else:
                    log_line = "跳过出发区直行 → 开始寻球"
                state_name = "FIND_TARGET"
                continue

            # ---------- FIND_TARGET: 搜索目标 ----------
            elif state_name == "FIND_TARGET":
                if current_target is not None:
                    if not calibrated:
                        state_name = "CALIBRATE"
                        log_line = f"发现目标 cls={current_target['class_id']} → 坐标校准"
                    else:
                        state_name = "APPROACH_TARGET"
                        log_line = f"发现目标 cls={current_target['class_id']} conf={current_target['confidence']:.2f}"
                    continue

                # 没找到: 一直左转搜索
                cmd = CMD_TURN_LEFT
                log_line = "搜索中 左转"

            # ---------- CALIBRATE: 坐标校准旋转 (仅一次) ----------
            elif state_name == "CALIBRATE":
                if current_target is None:
                    state_name = "FIND_TARGET"
                    log_line = "校准前目标丢失 → 重新搜索"
                    continue

                world = current_target.get("world_bottom")
                if world is None:
                    # 无世界坐标, 跳过校准直接接近
                    calibrated = True
                    state_name = "APPROACH_TARGET"
                    log_line = "无世界坐标, 跳过校准 → 直行接近"
                    continue

                angle = compute_target_angle(world)
                if angle is None or abs(angle) < ANGLE_DEADZONE_DEG:
                    calibrated = True
                    state_name = "APPROACH_TARGET"
                    log_line = f"已校准 (偏角={'—' if angle is None else f'{angle:+.1f}°'}) → 直行接近"
                    continue

                duration = compute_rotation_duration(angle)
                direction = "右" if angle > 0 else "左"
                log_line = f"校准: {direction}转 {duration:.2f}s (偏角={angle:+.1f}°)"

                # 计算完立即执行旋转 (打破动作/停止节奏，直接发长指令)
                cmd = CMD_TURN_RIGHT if angle > 0 else CMD_TURN_LEFT
                send_command(ser, cmd)
                # 跳过本轮的动作/停止节奏，直接按校准时长旋转
                rotate_start = time.time()
                while time.time() - rotate_start < duration:
                    if stop_event.is_set():
                        break
                    time.sleep(0.02)
                send_command(ser, CMD_STILL)

                calibrated = True
                state_name = "APPROACH_TARGET"
                print(f"[CALIBRATE] {log_line}")
                continue

            # ---------- APPROACH_TARGET: 直行接近 (连续前进 + 增量校准) ----------
            elif state_name == "APPROACH_TARGET":
                if current_target is None:
                    state_name = "FIND_TARGET"
                    last_cal_y2 = -1.0
                    log_line = "目标丢失 → 重新搜索"
                    continue

                cx, y2 = ball_bottom_center(current_target["box"])

                # 初始化校准基线
                if last_cal_y2 < 0:
                    last_cal_y2 = y2

                # 增量校准: y2 每增大 30 就校准一次
                if y2 - last_cal_y2 >= 30:
                    send_command(ser, CMD_STILL)  # 先停下前进
                    world = current_target.get("world_bottom")
                    if world is not None:
                        angle = compute_target_angle(world)
                        if angle is not None and abs(angle) >= ANGLE_DEADZONE_DEG:
                            duration = compute_rotation_duration(angle)
                            direction = "右" if angle > 0 else "左"
                            print(f"[APPROACH] 增量校准 y2={y2:.0f}: {direction}转 {duration:.2f}s (偏角={angle:+.1f}°)")

                            cmd = CMD_TURN_RIGHT if angle > 0 else CMD_TURN_LEFT
                            send_command(ser, cmd)
                            rotate_start = time.time()
                            while time.time() - rotate_start < duration:
                                if stop_event.is_set():
                                    break
                                time.sleep(0.02)
                            send_command(ser, CMD_STILL)
                    last_cal_y2 = y2
                    log_line = f"增量校准完成 y2={y2:.0f}"
                    continue

                # 球进入抓取区域 → 补 0.5s 前进偏差 → 抓!
                if is_ball_in_grab_zone(current_target["box"]):
                    print(f"[APPROACH] 球进入抓取区 y2={y2:.0f}, 补前进 0.5s...")
                    send_command(ser, CMD_FORWARD)
                    fwd_start = time.time()
                    while time.time() - fwd_start < 0.5:
                        if stop_event.is_set():
                            break
                        time.sleep(0.02)
                    send_command(ser, CMD_STILL)
                    state_name = "GRAB"
                    log_line = f"偏差补偿完成 → 执行抓取"
                    continue

                # 连续前进 (绕过全局启停节奏)
                send_command(ser, CMD_FORWARD)
                time.sleep(0.05)
                log_line = f"接近 y2={y2:.0f} cx={cx:.0f} (下次校准@{last_cal_y2+30:.0f})"
                continue

            # ---------- GRAB: 执行抓取 ----------
            elif state_name == "GRAB":
                cmd = CMD_SERVO_DOWN
                state_name = "FIND_SAFE_ZONE"
                # 首球标记
                if not first_done:
                    first_done = True
                    log_line = "舵机下放-抓球 (首球) → 寻找安全区"
                else:
                    log_line = "舵机下放-抓球 → 寻找安全区"
                # 清空当前目标 (准备下次重新选)
                current_target = None
                search_cycles = 0
                search_dir = 1

            # ---------- FIND_SAFE_ZONE: 搜索本方安全区 (一直左转) ----------
            elif state_name == "FIND_SAFE_ZONE":
                if current_safe_zone is not None:
                    state_name = "ALIGN_SAFE_ZONE"
                    log_line = "发现安全区 → 对齐入口"
                    continue

                # 一直左转: 转 0.3s / 停 0.5s (绕过全局节奏)
                send_command(ser, CMD_TURN_LEFT)
                turn_start = time.time()
                while time.time() - turn_start < 0.3:
                    if stop_event.is_set():
                        break
                    time.sleep(0.02)
                send_command(ser, CMD_STILL)
                if stop_event.is_set():
                    break
                stop_start = time.time()
                while time.time() - stop_start < 0.5:
                    if stop_event.is_set():
                        break
                    time.sleep(0.05)
                log_line = "找安全区 左转"
                continue

            # ---------- ALIGN_SAFE_ZONE: 对准安全区入口 ----------
            elif state_name == "ALIGN_SAFE_ZONE":
                if current_safe_zone is None:
                    state_name = "FIND_SAFE_ZONE"
                    log_line = "安全区丢失 → 重新搜索"
                    continue

                mid_x = safe_zone_front_midpoint_x(current_safe_zone, SAFE_ZONE_FRONT_KPT_INDICES)
                if mid_x is None:
                    # 关键点不可见，尝试接近
                    state_name = "APPROACH_SAFE_ZONE"
                    log_line = "安全区入口点不可见 → 尝试接近"
                    continue

                offset = mid_x - FRAME_WIDTH / 2
                if abs(offset) <= SAFE_ZONE_ALIGN_TOLERANCE_X:
                    state_name = "APPROACH_SAFE_ZONE"
                    log_line = f"入口已对准 mid_x={mid_x:.0f} → 直行接近"
                    continue
                elif offset < 0:
                    cmd = CMD_TURN_LEFT
                    log_line = f"左转对准安全区 (偏移={offset:.0f}px)"
                else:
                    cmd = CMD_TURN_RIGHT
                    log_line = f"右转对准安全区 (偏移={offset:.0f}px)"

            # ---------- APPROACH_SAFE_ZONE: 接近安全区 ----------
            elif state_name == "APPROACH_SAFE_ZONE":
                if current_safe_zone is None:
                    state_name = "FIND_SAFE_ZONE"
                    log_line = "安全区丢失 → 重新搜索"
                    continue

                mid_x = safe_zone_front_midpoint_x(current_safe_zone, SAFE_ZONE_FRONT_KPT_INDICES)
                mean_y = safe_zone_front_mean_y(current_safe_zone, SAFE_ZONE_FRONT_KPT_INDICES)

                # 够近了 → 释放
                if mean_y is not None and mean_y >= SAFE_ZONE_RELEASE_Y_THRESHOLD:
                    state_name = "RELEASE"
                    release_step = 0
                    log_line = f"已到释放距离 (入口 y={mean_y:.0f}) → 释放"
                    continue

                # 检查是否偏离
                if mid_x is not None and abs(mid_x - FRAME_WIDTH / 2) > SAFE_ZONE_ALIGN_TOLERANCE_X * 1.5:
                    state_name = "ALIGN_SAFE_ZONE"
                    log_line = f"偏离入口 → 重新对齐"
                    continue

                cmd = CMD_FORWARD
                y_str = f"入口 y={mean_y:.0f}" if mean_y is not None else "入口距离未知"
                log_line = f"接近安全区 {y_str}"

            # ---------- RELEASE: 释放并后退 ----------
            elif state_name == "RELEASE":
                if release_step == 0:
                    cmd = CMD_SERVO_UP
                    release_step = 1
                    log_line = "舵机上抬-释放球"
                elif release_step <= RELEASE_BACK_CYCLES:
                    cmd = CMD_BACKWARD
                    release_step += 1
                    log_line = f"后退 {release_step-1}/{RELEASE_BACK_CYCLES}"
                else:
                    # 释放完成 → 开始新一轮
                    state_name = "FIND_TARGET"
                    current_safe_zone = None
                    calibrated = False       # 新一轮目标需要重新校准
                    last_cal_y2 = -1.0       # 重置增量校准基线
                    search_cycles = 0
                    search_dir = 1
                    log_line = "释放完成 → 开始新一轮寻球"
                    continue

            # ======================================
            # 5. 卡住检测: 同状态过久则降级
            # ======================================
            if state_name == prev_state:
                stuck_counter += 1
            else:
                stuck_counter = 0
                prev_state = state_name
                # 通知录制节点当前状态
                if state_queue is not None:
                    if state_queue.full():
                        try: state_queue.get_nowait()
                        except: pass
                    state_queue.put(state_name)

            if stuck_counter >= STUCK_MAX_CYCLES:
                print(f"[卡住] 状态 '{state_name}' 持续 {stuck_counter} 周期, 强制重置搜索")
                stuck_counter = 0
                current_target = None
                current_safe_zone = None
                if state_name.startswith("FIND_"):
                    search_dir = -search_dir  # 换方向
                    search_cycles = 0
                else:
                    state_name = "FIND_TARGET" if "SAFE" not in state_name else "FIND_SAFE_ZONE"
                    search_cycles = 0

            # ======================================
            # 6. 执行: 发指令 → 等待动作时长
            # ======================================
            send_command(ser, cmd)
            action_start = time.time()
            while time.time() - action_start < LOGIC_ACTION_DURATION:
                if stop_event.is_set():
                    break
                time.sleep(0.02)

            if stop_event.is_set():
                break

            # ---- 停止阶段 ----
            send_command(ser, CMD_STILL)
            stop_start = time.time()
            while time.time() - stop_start < LOGIC_STOP_DURATION:
                if stop_event.is_set():
                    break
                time.sleep(0.02)

            # ---- 日志 ----
            print(f"[{state_name}] {log_line}")

    except Exception as e:
        print(f"[逻辑崩溃] 主循环异常: {e}")
        import traceback
        traceback.print_exc()

    # ---- 安全退出 ----
    if ser is not None:
        send_command(ser, CMD_STILL)
        ser.close()
    print("--- 比赛逻辑 v2 已停止 ---")
