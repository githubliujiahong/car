# rpi5_detect/nodes/logic_test.py
"""
测试用逻辑节点 — 帧驱动架构，找球 → 接近 → 抓取，一轮即停。

状态流转:
  OBSERVE(5帧) → SEARCH(持续左转) → CALIBRATE(像素迭代校准) → APPROACH(帧驱动前进)
       → [每+30y2: 像素增量校准] → GRAB
       → ZONE_SEARCH → ZONE_CALIBRATE(水平对准kp1,2中点)
       → ZONE_APPROACH(前进 → 距目标<70px停车 → kp置信度自动对准)
       → ZONE_RELEASE(舵机上抬 → 前冲1s → 后退) → 下一球
"""

import serial
import struct
import time
import math
import cv2
import numpy as np
from config import *

CENTER_Y = FRAME_HEIGHT // 2


# ==========================================
# ⚙️ 测试参数
# ==========================================

OBSERVE_FRAMES = 5            # 启动后观察帧数
LOSS_TOLERANCE = 10           # 连续丢目标超过此帧数才回退
GRAB_VERIFY_FRAMES = 10      # 抓取后验证帧数
PUSH_VERIFY_FRAMES = 10      # 首次推球后验证帧数



# ==========================================
# 几何工具
# ==========================================

def ball_bottom_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2


# ==========================================
# 目标筛选
# ==========================================

def classify_detections(det_results):
    return [res for res in det_results if res["class_id"] in (0, 1, 2, 3)]


def select_target(balls, target_cls):
    """选 y2 最大的球。target_cls 可以是单个 class_id 或 (cls1, cls2, ...) 元组"""
    if isinstance(target_cls, (list, tuple)):
        candidates = [b for b in balls if b["class_id"] in target_cls]
    else:
        candidates = [b for b in balls if b["class_id"] == target_cls]
    if not candidates:
        return None
    return max(candidates, key=lambda b: b["box"][3])


def is_ball_in_grab_zone(box):
    # Y 方向下界触发：球底部越过 Y1 即算到位，抗过冲（快球越过 Y2 也能停）
    # X 方向仍要求球大致居中
    cx, y2 = ball_bottom_center(box)
    return (GRAB_ZONE_X1 <= cx <= GRAB_ZONE_X2) and (y2 >= GRAB_ZONE_Y1)


# ==========================================
# 区域检测工具 (Zone Detection)
# ==========================================

def get_zone_keypoints_midpoint(keypoints):
    """
    计算安全区 4 个关键点的中点 (像素坐标)。
    keypoints: [[kx, ky, kconf], ...]  — 4 个关键点
    返回 (cx, cy) 或 None (关键点无效时)
    """
    if not keypoints or len(keypoints) < 4:
        return None
    xs = [kp[0] for kp in keypoints[:4]]
    ys = [kp[1] for kp in keypoints[:4]]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def get_zone_top_edge(keypoints):
    """
    获取区域上边 (画面中 y 最小的两个关键点)。
    keypoints: [[kx, ky, kconf], ...] — 4 个关键点
    返回 (idx_l, idx_r, kp_l, kp_r, mid_x, mid_y) 或 None (关键点无效时)
    其中 idx_l/kp_l 是左侧点 (x较小), idx_r/kp_r 是右侧点 (x较大)。
    """
    if not keypoints or len(keypoints) < 4:
        return None
    # 取前4个关键点，按 y 排序，选 y 最小的两个 (画面上方 = 远处 = 区域的"前"面)
    indexed = list(enumerate(keypoints[:4]))
    indexed.sort(key=lambda x: x[1][1])  # 按 y 升序
    top_two = indexed[:2]                # y 最小的两个
    # 按 x 排序，分出左右
    top_two.sort(key=lambda x: x[1][0])  # 按 x 升序
    (idx_l, kp_l), (idx_r, kp_r) = top_two
    mid_x = (kp_l[0] + kp_r[0]) / 2.0
    mid_y = (kp_l[1] + kp_r[1]) / 2.0
    return idx_l, idx_r, kp_l, kp_r, mid_x, mid_y


def classify_zones(det_results):
    """筛选安全区检测结果 (cls 4=red_safe_zone, 5=blue_safe_zone)"""
    return [res for res in det_results if res["class_id"] in (4, 5)]


def is_ball_near_zone(ball_box, zone_keypoints, margin=30):
    """球包围盒与区域四关键点包围盒(+margin)是否重叠"""
    if len(zone_keypoints) < 4:
        return False
    xs = [kp[0] for kp in zone_keypoints[:4]]
    ys = [kp[1] for kp in zone_keypoints[:4]]
    zx1, zx2 = min(xs) - margin, max(xs) + margin
    zy1, zy2 = min(ys) - margin, max(ys) + margin
    bx1, by1, bx2, by2 = ball_box
    return not (bx2 < zx1 or bx1 > zx2 or by2 < zy1 or by1 > zy2)


def filter_balls_near_zones(balls, zones, margin=30):
    """排除与任意区域重叠的球"""
    if not zones or not balls:
        return balls
    return [b for b in balls
            if not any(is_ball_near_zone(b["box"], z.get("keypoints", []), margin)
                       for z in zones)]


# ==========================================
# 颜色触发检测 (Color Trigger for ZONE_APPROACH)
# ==========================================

def check_zone_color_trigger(frame):
    """
    检查抓取区内缩 20px 区域中目标颜色像素占比是否 >= 80%。
    """
    h, w = frame.shape[:2]
    M = 20  # 内缩量

    x1 = max(0, GRAB_ZONE_X1 + M)
    y1 = max(0, GRAB_ZONE_Y1 + M)
    x2 = min(w, GRAB_ZONE_X2 - M)
    y2 = min(h, GRAB_ZONE_Y2 - M)

    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels < 50:
        return False

    roi = cv2.convertScaleAbs(roi, alpha=1.8, beta=10)

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int32)
    h = roi_hsv[:, :, 0]
    s = np.clip(roi_hsv[:, :, 1] * 1.4, 0, 255)
    v = roi_hsv[:, :, 2]

    if if_red:
        color_mask = ((h <= 5) | (h >= 175)) & (s > 170) & (v > 120)
    else:
        color_mask = (h >= 108) & (h <= 122) & (s > 170) & (v > 120)

    color_pixels = np.count_nonzero(color_mask)
    ratio = color_pixels / total_pixels

    if ratio >= 0.90:
        print(f"  [颜色触发] {'红' if if_red else '蓝'}色占比 {ratio:.1%} >= 80% "
              f"(内缩区{total_pixels}px, 命中{color_pixels}px)")
        return True
    return False


def count_balls_in_ring(frame, det_results):
    """
    统计抓取区内缩 20px 区域内有多少个球的中心点。
    """
    h, w = frame.shape[:2]
    M = 20
    x1 = max(0, GRAB_ZONE_X1 + M)
    y1 = max(0, GRAB_ZONE_Y1 + M)
    x2 = min(w, GRAB_ZONE_X2 - M)
    y2 = min(h, GRAB_ZONE_Y2 - M)

    balls = classify_detections(det_results)
    count = 0
    for ball in balls:
        bx1, by1, bx2, by2 = ball["box"]
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        if (x1 <= bcx <= x2) and (y1 <= bcy <= y2):
            count += 1
    return count


def select_zone(zones, target_zone_cls):
    """
    选目标安全区: 同类中选 y 最大 (画面最低、最近) 的那个。
    返回选中的 zone dict 或 None。
    """
    candidates = [z for z in zones if z["class_id"] == target_zone_cls]
    if not candidates:
        return None
    # 用关键点中点 y 来判断远近
    def zone_y(z):
        mid = get_zone_keypoints_midpoint(z.get("keypoints", []))
        return mid[1] if mid else 0
    return max(candidates, key=zone_y)


# ==========================================
# 串口
# ==========================================

def send_command(ser, cmd):
    """发送串口指令。"""
    global _last_cmd_time, _last_cmd

    if ser is None:
        return

    now = time.time()

    cmd_names = {
        0x10: "FORWARD", 0x11: "TURN_L", 0x12: "TURN_R",
        0x20: "STILL", 0x15: "SERVO_UP", 0x16: "SERVO_DN",
        0x14: "SERVO_MID", 0x13: "BACK", 0x19: "RESET",
    }

    interval = now - _last_cmd_time if _last_cmd_time > 0 else 0
    _last_cmd_time = now
    _last_cmd = cmd

    name = cmd_names.get(cmd, f"0x{cmd:02X}")
    print(f"  [CMD] {name}  (距上次 {interval:.3f}s)")

    packet = struct.pack('<BBBB', 0xAA, 0x02, cmd, 0x55)
    ser.write(packet)

_last_cmd_time = 0.0
_last_cmd = None


# ==========================================
# 像素坐标校准 (替代不准确的世界坐标校准)
# ==========================================

def do_calibration_pixel(ser, cx, label=""):
    """
    固定时长旋转校准: 偏左 → 右转 CALIB_DURATION 秒 → 停, 偏右 → 左转 CALIB_DURATION 秒 → 停。
    调用方须自行判断死区 (abs(cx - CENTER_X) >= PIXEL_DEADZONE)。
    """
    if cx < CENTER_X:
        direction = "右"
        cmd = CMD_TURN_RIGHT
    else:
        direction = "左"
        cmd = CMD_TURN_LEFT

    print(f"  [{label}校准] {direction}转 {CALIB_DURATION:.2f}s "
          f"(偏移={cx - CENTER_X:+.0f}px)")

    send_command(ser, cmd)
    time.sleep(CALIB_DURATION)
    send_command(ser, CMD_STILL)
    return True


# ==========================================
# 主循环
# ==========================================

def logic_test_worker(result_queue, recorder_queue, stop_event, state_queue=None):

    # ---- 区域目标颜色 (全程不变) ----
    zone_class_name = f"{OWN_SAFE_ZONE_COLOR.lower()}_safe_zone"
    target_zone_cls = CLASS_ID.get(zone_class_name)
    if target_zone_cls is None:
        print(f"[TEST] 无效的己方安全区颜色: {OWN_SAFE_ZONE_COLOR} (解析为 {zone_class_name})")
        return

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        print(f"[TEST] 串口 {SERIAL_PORT} 已打开")
        send_command(ser, 0x15)
        #ime.sleep(3.5)
    except Exception as e:
        print(f"[TEST] 串口失败: {e}")
        ser = None
    # 阵营直接从 config.py 读取 (if_red / FIRST_BALL / SECONDARY_BALLS / OWN_SAFE_ZONE_COLOR)
    print(f"[TEST] 队伍设定 (config.py): {'红' if if_red else '蓝'}队\n")

    count = 10
    while(count>0):#开机后十次重置
        count -= 1
        send_command(ser,0x19)

    # 启动直行
    print(f"[TEST] 启动直行 {INIT_FORWARD_SECONDS}s\n")
    send_command(ser, 0x17)
    t0 = time.time()
    while time.time() - t0 < INIT_FORWARD_SECONDS:
        if stop_event.is_set():
            break
        time.sleep(0.05)
    send_command(ser, CMD_STILL)

    total_cycles = 20  # 首球 + 19轮黄/黑
    cycle_idx = 0

    print(f"[TEST] 启动 | 阵营: {'红' if if_red else '蓝'}队 | "
          f"首球: {FIRST_BALL} → 后续: {SECONDARY_BALLS} | 共{total_cycles}个周期\n")

    # ---- 阶段 1 (找球) 状态变量 ----
    current_target = None
    loss_count = 0
    last_cal_y2 = -1.0
    observe_count = 0
    frame_count = 0

    # SEARCH 间歇参数
    search_phase = 'rotate'       # 'rotate' / 'observe'
    search_phase_start = 0.0      # 当前 phase 开始时间

    state_name = "OBSERVE"
    prev_state = None
    approach_entered = False      # APPROACH 入口标记，避免重复发 CMD_FORWARD

    # ---- 阶段 2 (搬区域) 状态变量 ----
    current_zone = None
    zone_loss_count = 0
    zone_last_cal_my = -1.0       # 上次增量校准时中点 y
    zone_search_phase = 'rotate'
    zone_search_phase_start = 0.0
    zone_approach_entered = False
    zone_top_kpt_indices = None   # (idx_l, idx_r) 区域上边两个关键点的原始序号
    zone_align_done = False       # ZONE_APPROACH 中自动对准是否已完成
    zone_color_triggered = False  # ZONE_APPROACH 中颜色触发标志
    state_frame_count = 0         # 当前状态持续帧数 (防卡死)
    first_ball_fail_count = 0   # 首球验证连续失败次数 (>=2 直接进 cycle 1)

    # ---- 周期目标设置: cycle0=首球单色, cycle1+=黄/黑 ----
    target_cls = CLASS_ID.get(FIRST_BALL.lower())
    if target_cls is None:
        print(f"[TEST] 无效的首球颜色: {FIRST_BALL}")
        return

    print(f"\n{'='*50}")
    print(f"[周期 1/{total_cycles}] 目标球: {FIRST_BALL}")
    print(f"{'='*50}\n")

    try:
        while not stop_event.is_set():
            # ======================================
            # 取帧 (逐帧检查，不放过命中帧)
            # ======================================
            if result_queue.empty():
                time.sleep(0.01)
                continue

            found_this_frame = False
            batch_detections = 0  # 这批帧里推理节点检测到的总数
            balls = []            # classify_detections 后的候选球
            grab_hit = False      # 本批任意一帧小球进入抓取区 → 锁存停车
            grab_hit_target = None

            # zone 状态下每批开始时清空，只有本批检测到才赋值 (确保丢失时正确累加 loss)
            if state_name in ("ZONE_SEARCH", "ZONE_CALIBRATE", "ZONE_APPROACH"):
                current_zone = None
            if state_name == "ZONE_APPROACH":
                zone_color_triggered = False  # 每批重置

            while not result_queue.empty():
                data = result_queue.get()
                frame, det_results = data
                batch_detections += len(det_results)

                # 始终分类区域: 球状态用于过滤, 区域状态用于追踪
                zones_all = classify_zones(det_results)
                zone_candidate = select_zone(zones_all, target_zone_cls)
                if state_name in ("ZONE_SEARCH", "ZONE_CALIBRATE", "ZONE_APPROACH"):
                    if zone_candidate is not None:
                        current_zone = zone_candidate

                balls = classify_detections(det_results)
                # 球状态下排除区域周围的球
                if state_name not in ("ZONE_SEARCH", "ZONE_CALIBRATE", "ZONE_APPROACH", "ZONE_RELEASE"):
                    balls = filter_balls_near_zones(balls, zones_all, margin=10)
                target = select_target(balls, target_cls)

                # 逐帧检查：任何一帧命中就更新 current_target
                if target is not None:
                    current_target = target
                    found_this_frame = True

                    # 逐帧锁存抓取命中：只要有一帧到位就记下，避免被后续帧覆盖漏判
                    if state_name == "APPROACH" and is_ball_in_grab_zone(target["box"]):
                        grab_hit = True
                        grab_hit_target = target

                # ZONE_APPROACH 中逐帧颜色触发检测
                if state_name == "ZONE_APPROACH" and not zone_color_triggered:
                    if check_zone_color_trigger(frame):
                        zone_color_triggered = True

            # ---- 诊断日志 (每 5 帧打印，排查"有画面无状态切换") ----
            if frame_count % 5 == 0:
                t_info = ""
                if state_name in ("ZONE_SEARCH", "ZONE_CALIBRATE", "ZONE_APPROACH"):
                    if current_zone is not None:
                        kpts = current_zone.get("keypoints", [])
                        if zone_top_kpt_indices is not None and len(kpts) >= 4:
                            idx_l, idx_r = zone_top_kpt_indices
                            mx = (kpts[idx_l][0] + kpts[idx_r][0]) / 2.0
                            my = (kpts[idx_l][1] + kpts[idx_r][1]) / 2.0
                            t_info = f" 锁定区域: cls={current_zone['class_id']} 上边mx={mx:.0f} my={my:.0f}"
                        else:
                            mid = get_zone_keypoints_midpoint(kpts)
                            if mid:
                                t_info = f" 锁定区域: cls={current_zone['class_id']} mx={mid[0]:.0f} my={mid[1]:.0f}"
                            else:
                                t_info = f" 锁定区域: cls={current_zone['class_id']} (kpt无效)"
                    elif found_this_frame:
                        t_info = " 锁定区域: (已命中但被覆盖)"
                elif state_name not in ("ZONE_SEARCH", "ZONE_CALIBRATE", "ZONE_APPROACH", "ZONE_RELEASE"):
                    if current_target is not None:
                        cx, y2 = ball_bottom_center(current_target["box"])
                        t_info = f" 锁定: cls={current_target['class_id']} cx={cx:.0f} y2={y2:.0f}"
                    elif found_this_frame:
                        t_info = " 锁定: (已命中但被覆盖)"

                zone_info = ""
                if state_name in ("ZONE_SEARCH", "ZONE_CALIBRATE", "ZONE_APPROACH"):
                    zone_info = f" zone_phase={zone_search_phase if state_name=='ZONE_SEARCH' else '-'} zone_loss={zone_loss_count}"
                else:
                    zone_info = f" phase={search_phase if state_name=='SEARCH' else '-'}"

                print(f"  [诊断#{frame_count}] state={state_name}{zone_info} "
                      f"推理检测={batch_detections} 候选球={len(balls)} "
                      f"命中={found_this_frame} loss={loss_count}{t_info}")

            # ---- 丢失帧计数 (LOSS_TOLERANCE 机制) ----
            # 仅球追踪阶段生效；zone 阶段用独立的 zone_loss_count，
            # 否则此计数器在 zone 阶段找不到球会一路累加，误触发"回退 SEARCH"
            if state_name in ("ZONE_SEARCH", "ZONE_CALIBRATE", "ZONE_APPROACH", "ZONE_RELEASE"):
                loss_count = 0
            else:
                if found_this_frame:
                    loss_count = 0
                else:
                    loss_count += 1

                # 连续丢失超过容错阈值 → 清空目标，触发回退
                if loss_count >= LOSS_TOLERANCE and current_target is not None:
                    print(f"  连续丢失{loss_count}帧 → 目标丢失，回退 SEARCH\n")
                    current_target = None
                    loss_count = 0
                    if state_name in ("CALIBRATE", "APPROACH"):
                        state_name = "SEARCH"
                        search_phase = 'rotate'
                        search_phase_start = time.time()

            frame_count += 1

            # ---- 卡死检测: 同一状态超过 50 帧 → 回退 SEARCH ----
            if state_name != prev_state:
                state_frame_count = 0
            state_frame_count += 1
            if state_frame_count > 50 and state_name not in ("SEARCH", "OBSERVE", "ZONE_RELEASE"):
                print(f"  [卡死] {state_name} 持续{state_frame_count}帧 → 回退 SEARCH\n")
                send_command(ser, CMD_STILL)
                current_target = None
                loss_count = 0
                search_phase = 'rotate'
                search_phase_start = time.time()
                state_name = "SEARCH"
                prev_state = None
                approach_entered = False
                current_zone = None
                zone_loss_count = 0
                zone_last_cal_my = -1.0
                zone_search_phase = 'rotate'
                zone_search_phase_start = 0.0
                zone_approach_entered = False
                zone_top_kpt_indices = None
                zone_align_done = False
                zone_color_triggered = False
                state_frame_count = 0
                if state_queue is not None:
                    if state_queue.full():
                        try: state_queue.get_nowait()
                        except: pass
                    state_queue.put(state_name)
                continue

            # ======================================
            # 状态切换时打印 (不每帧刷屏)
            # ======================================
            if state_name != prev_state:
                print(f"[{state_name}] → 进入状态")
                prev_state = state_name
                # 通知录制节点当前状态
                if state_queue is not None:
                    if state_queue.full():
                        try: state_queue.get_nowait()
                        except: pass
                    state_queue.put(state_name)

            # ======================================
            # 状态机
            # ======================================

            # ---- OBSERVE: 观察 ----
            if state_name == "OBSERVE":

                observe_count += 1

                if current_target is not None:
                    print(f"  第{observe_count}帧 命中 → CALIBRATE\n")
                    state_name = "CALIBRATE"
                    continue

                if observe_count >= OBSERVE_FRAMES:
                    print(f"  未发现 → SEARCH\n")
                    search_phase = 'rotate'
                    search_phase_start = time.time()
                    state_name = "SEARCH"
                    continue

            # ---- SEARCH: 间歇旋转搜索 (转停交替，避免卷帘快门模糊) ----
            elif state_name == "SEARCH":
                if current_target is not None:
                    cx, y2 = ball_bottom_center(current_target["box"])
                    send_command(ser, CMD_STILL)
                    time.sleep(0.05)
                    print(f"  命中 (cx={cx:.0f} phase={search_phase}) → CALIBRATE\n")
                    state_name = "CALIBRATE"
                    continue

                # 间歇搜索: 转 → 停 → 等帧确认 → 循环
                now = time.time()
                elapsed = now - search_phase_start

                if search_phase == 'rotate':
                    if elapsed >= BALL_SEARCH_ROTATE_DURATION:
                        send_command(ser, CMD_STILL)
                        search_phase = 'observe'
                        search_phase_start = now
                elif search_phase == 'observe':
                    if elapsed >= BALL_SEARCH_OBSERVE_DURATION:
                        # 到期不等同于可以转，先等当前帧处理完
                        search_phase = 'observe_wait'
                        search_phase_start = now
                elif search_phase == 'observe_wait':
                    # 上轮刚处理完一批帧，未命中 → 确认可以旋转
                    turn_cmd = CMD_TURN_LEFT if BALL_SEARCH_ROTATE_DIR == 'left' else CMD_TURN_RIGHT
                    send_command(ser, turn_cmd)
                    search_phase = 'rotate'
                    search_phase_start = now

            # ---- CALIBRATE: 像素校准 (迭代至球在画面中心) ----
            elif state_name == "CALIBRATE":
                if current_target is None:
                    # 校准后目标丢失：保持前进，等待新帧重新捕获
                    if loss_count >= LOSS_TOLERANCE:
                        send_command(ser, CMD_STILL)
                        print(f"  校准后连续丢失{loss_count}帧 → 回退 SEARCH\n")
                        state_name = "SEARCH"
                        search_phase = 'rotate'
                        search_phase_start = time.time()
                        loss_count = 0
                    elif loss_count == 1:
                        # 刚开始丢帧，发一次前进指令继续接近
                        send_command(ser, CMD_FORWARD)
                    continue

                cx, y2 = ball_bottom_center(current_target["box"])

                if abs(cx - CENTER_X) < PIXEL_DEADZONE:
                    state_name = "APPROACH"
                    approach_entered = False
                    wb = current_target.get("world_bottom")
                    off_str = f"横偏={wb[0]*100:+.1f}cm" if wb is not None else f"偏移={cx-CENTER_X:+.0f}px"
                    print(f"  已对准 ({off_str}) → APPROACH\n")
                    continue

                do_calibration_pixel(ser, cx, label="首次")
                last_cal_y2 = -1.0
                current_target = None  # 校准后清空，下轮必须用新帧判断
                continue

            # ---- APPROACH: 帧驱动前进 + 增量校准 ----
            elif state_name == "APPROACH":
                # 最高优先：本批任意一帧已进入抓取区 → 立即停车抓取
                # (逐帧锁存，避免命中帧被同批后续帧覆盖而漏判)
                if grab_hit:
                    gcx, gy2 = ball_bottom_center(grab_hit_target["box"])
                    print(f"  进入抓取区 (cx={gcx:.0f} y2={gy2:.0f}) → GRAB\n")
                    send_command(ser, CMD_STILL)
                    current_target = grab_hit_target
                    state_name = "GRAB"
                    continue

                if current_target is None:
                    # 模型暂时丢帧，不回溯搜索，继续前进
                    if not approach_entered:
                        send_command(ser, CMD_FORWARD)
                        approach_entered = True
                    continue

                # 仅状态首次进入时发一次前进指令
                if not approach_entered:
                    send_command(ser, CMD_FORWARD)
                    approach_entered = True

                cx, y2 = ball_bottom_center(current_target["box"])

                if is_ball_in_grab_zone(current_target["box"]):
                    cx, y2 = ball_bottom_center(current_target["box"])
                    print(f"  进入抓取区 (cx={cx:.0f} y2={y2:.0f}) → GRAB\n")
                    send_command(ser, CMD_STILL)
                    state_name = "GRAB"
                    continue

                # 增量校准: y2 每增大 30px 且球偏离中线时校准一次
                if abs(cx - CENTER_X) >= PIXEL_DEADZONE \
                        and (last_cal_y2 < 0 or y2 - last_cal_y2 >= 30):
                    do_calibration_pixel(ser, cx, label=f"y2={y2:.0f}")
                    last_cal_y2 = y2
                    approach_entered = False  # 校准后需重新发前进指令

            # ---- GRAB: 抓取 + 验证 → 进入区域搬运阶段 ----
            elif state_name == "GRAB":
                # 补距 + 抓取
                send_command(ser, CMD_FORWARD)
                time.sleep(0.25)

                send_command(ser, CMD_SERVO_DOWN)
                time.sleep(0.2)
                send_command(ser, CMD_STILL)

                # 清空累积的旧帧，确保验证用的是抓取后的新帧
                while not result_queue.empty():
                    result_queue.get()

                # 验证抓取: 10帧内检查grab_zone是否有目标
                print(f"[GRAB] 抓取动作完成 → 验证抓取结果 (最多{GRAB_VERIFY_FRAMES}帧)\n")
                grab_ok = False
                for i in range(GRAB_VERIFY_FRAMES):
                    # 等待新帧到达
                    t0 = time.time()
                    while result_queue.empty() and (time.time() - t0) < 0.2:
                        time.sleep(0.02)
                    if not result_queue.empty():
                        data = result_queue.get()
                        while not result_queue.empty():
                            data = result_queue.get()
                        frame, det_results = data
                        balls = classify_detections(det_results)
                        target = select_target(balls, target_cls)
                        if target is not None and is_ball_in_grab_zone(target["box"]):
                            grab_ok = True
                            cx, y2 = ball_bottom_center(target["box"])
                            print(f"  第{i+1}帧 检测到球在抓取区 (cx={cx:.0f} y2={y2:.0f}) → 抓取成功\n")
                            break
                    else:
                        pass  # 无新帧，继续等待

                if grab_ok:
                    print(f"[GRAB] 抓取成功 → 进入区域搜索阶段\n")
                    state_name = "ZONE_SEARCH"
                else:
                    print(f"[GRAB] 抓取失败 ({GRAB_VERIFY_FRAMES}帧内抓取区无目标) → 回退 SEARCH\n")
                    send_command(ser, CMD_SERVO_UP)
                    state_name = "SEARCH"
                    search_phase = 'rotate'
                    search_phase_start = time.time()

                current_target = None  # 球阶段结束，清空残留目标
                loss_count = 0
                zone_search_phase = 'rotate'
                zone_search_phase_start = time.time()
                continue

            # ==========================================
            # 阶段 2: 区域搬运 (Zone Delivery)
            # ==========================================

            # ---- ZONE_SEARCH: 间歇旋转搜索安全区 ----
            elif state_name == "ZONE_SEARCH":
                # current_zone 已在帧排空循环中逐帧更新

                if current_zone is not None:
                    kpts = current_zone.get("keypoints", [])
                    send_command(ser, CMD_STILL)
                    time.sleep(0.05)
                    if kpts and len(kpts) >= 4:
                        zone_top_kpt_indices = (1, 2)  # 固定序号1,2
                        zone_align_done = False
                        zone_color_triggered = False
                        mx = (kpts[1][0] + kpts[2][0]) / 2.0
                        my = (kpts[1][1] + kpts[2][1]) / 2.0
                        print(f"  发现目标区域 cls={current_zone['class_id']} "
                              f"kpt1,2中点=({mx:.0f}, {my:.0f}) → ZONE_CALIBRATE\n")
                    else:
                        print(f"  发现目标区域 cls={current_zone['class_id']} (关键点无效) → ZONE_CALIBRATE\n")
                    state_name = "ZONE_CALIBRATE"
                    zone_loss_count = 0
                    continue

                # 间歇搜索: 转 → 停 → 等帧确认 → 循环
                now = time.time()
                elapsed = now - zone_search_phase_start

                if zone_search_phase == 'rotate':
                    if elapsed >= ZONE_SEARCH_ROTATE_DURATION:
                        send_command(ser, CMD_STILL)
                        zone_search_phase = 'observe'
                        zone_search_phase_start = now
                elif zone_search_phase == 'observe':
                    if elapsed >= ZONE_SEARCH_OBSERVE_DURATION:
                        # 到期不等同于可以转，先等当前帧处理完
                        zone_search_phase = 'observe_wait'
                        zone_search_phase_start = now
                elif zone_search_phase == 'observe_wait':
                    # 上轮刚处理完一批帧，未命中 → 确认可以旋转
                    turn_cmd = CMD_TURN_LEFT if ZONE_SEARCH_ROTATE_DIR == 'left' else CMD_TURN_RIGHT
                    send_command(ser, turn_cmd)
                    zone_search_phase = 'rotate'
                    zone_search_phase_start = now

            # ---- ZONE_CALIBRATE: 水平对准四关键点中点 ----
            elif state_name == "ZONE_CALIBRATE":
                # current_zone 已在帧排空循环中逐帧更新

                if current_zone is None:
                    zone_loss_count += 1
                    if zone_loss_count >= ZONE_LOSS_TOLERANCE:
                        send_command(ser, CMD_STILL)
                        print(f"  ZONE_CALIBRATE 连续丢失{zone_loss_count}帧 → 回退 ZONE_SEARCH\n")
                        state_name = "ZONE_SEARCH"
                        zone_search_phase = 'rotate'
                        zone_search_phase_start = time.time()
                        zone_loss_count = 0
                        current_zone = None
                    elif zone_loss_count == 1:
                        send_command(ser, CMD_FORWARD)
                    continue
                zone_loss_count = 0

                kpts = current_zone.get("keypoints", [])
                if len(kpts) < 4:
                    zone_loss_count += 1
                    continue
                # 四个关键点中点
                mid = get_zone_keypoints_midpoint(kpts)
                if mid is None:
                    zone_loss_count += 1
                    continue
                mx = mid[0]

                if abs(mx - CENTER_X) < PIXEL_DEADZONE:
                    state_name = "ZONE_APPROACH"
                    zone_approach_entered = False
                    print(f"  区域已水平对准 (偏移={mx-CENTER_X:+.0f}px) → ZONE_APPROACH\n")
                    continue

                do_calibration_pixel(ser, mx, label="区域首次")
                zone_last_cal_my = -1.0
                current_zone = None
                continue

            # ---- ZONE_APPROACH: 前进 → 颜色触发或avgY达标 → 后退 ----
            elif state_name == "ZONE_APPROACH":
                # 颜色触发: 立刻停车后退
                if zone_color_triggered:
                    send_command(ser, CMD_STILL)
                    print(f"  颜色触发! → ZONE_RELEASE\n")
                    state_name = "ZONE_RELEASE"
                    continue

                # current_zone 已在帧排空循环中逐帧更新
                if current_zone is None:
                    zone_loss_count += 1
                    if zone_loss_count >= ZONE_LOSS_TOLERANCE:
                        send_command(ser, CMD_STILL)
                        print(f"  ZONE_APPROACH 连续丢失{zone_loss_count}帧 → 回退 ZONE_SEARCH\n")
                        state_name = "ZONE_SEARCH"
                        zone_search_phase = 'rotate'
                        zone_search_phase_start = time.time()
                        zone_loss_count = 0
                        current_zone = None
                        zone_approach_entered = False
                    elif not zone_approach_entered:
                        send_command(ser, CMD_FORWARD)
                        zone_approach_entered = True
                    continue
                zone_loss_count = 0

                kpts = current_zone.get("keypoints", [])
                if len(kpts) < 4:
                    if not zone_approach_entered:
                        send_command(ser, CMD_FORWARD)
                        zone_approach_entered = True
                    continue

                # 目标点: 四关键点中点 (实时更新)
                mid = get_zone_keypoints_midpoint(kpts)
                if mid is None:
                    continue
                tx, ty = mid
                avg_y = ty

                # 仅状态首次进入时发一次前进指令
                if not zone_approach_entered:
                    send_command(ser, CMD_FORWARD)
                    zone_approach_entered = True

                # 停止条件: 四关键点平均 y >= ZONE_STOP_Y_THRESHOLD (越近 y 越大)
                if avg_y >= ZONE_STOP_Y_THRESHOLD:
                    send_command(ser, CMD_STILL)

                    # 自动对准: 根据 kp1/kp2 置信度判断朝向
                    if not zone_align_done:
                        conf1 = kpts[1][2] if len(kpts[1]) > 2 else 0
                        conf2 = kpts[2][2] if len(kpts[2]) > 2 else 0
                        kp1_ok = conf1 >= KP_CONF_THRESHOLD
                        kp2_ok = conf2 >= KP_CONF_THRESHOLD

                        if kp1_ok and not kp2_ok:
                            print(f"  自动对准: kp1可见(conf={conf1:.2f}) kp2不可见(conf={conf2:.2f}) "
                                  f"→ 右转{ZONE_ALIGN_TURN_DURATION:.2f}s\n")
                            send_command(ser, CMD_TURN_RIGHT)
                            time.sleep(ZONE_ALIGN_TURN_DURATION)
                        elif kp2_ok and not kp1_ok:
                            print(f"  自动对准: kp2可见(conf={conf2:.2f}) kp1不可见(conf={conf1:.2f}) "
                                  f"→ 左转{ZONE_ALIGN_TURN_DURATION:.2f}s\n")
                            send_command(ser, CMD_TURN_LEFT)
                            time.sleep(ZONE_ALIGN_TURN_DURATION)
                        else:
                            print(f"  自动对准: kp1(conf={conf1:.2f}) kp2(conf={conf2:.2f}) 均可见或均不可见 → 跳过\n")
                        send_command(ser, CMD_STILL)
                        zone_align_done = True

                    print(f"  到达区域前方 (四点中点({tx:.0f},{ty:.0f}) "
                          f"avg_y={avg_y:.0f} >= {ZONE_STOP_Y_THRESHOLD}) → ZONE_RELEASE\n")
                    state_name = "ZONE_RELEASE"
                    continue

                # 增量校准: ty 每增大 30px 且偏离中线时校准一次
                if abs(tx - CENTER_X) >= PIXEL_DEADZONE \
                        and (zone_last_cal_my < 0 or ty - zone_last_cal_my >= 30):
                    do_calibration_pixel(ser, tx, label=f"区域 ty={ty:.0f}")
                    zone_last_cal_my = ty
                    zone_approach_entered = False  # 校准后需重新发前进指令

            # ---- ZONE_RELEASE: 后退 → 抬舵机 → 首球验证 → 下一周期 ----
            elif state_name == "ZONE_RELEASE":
                # 1. 停车
                print(f"[ZONE_RELEASE] 停车\n")
                send_command(ser, CMD_STILL)
                time.sleep(0.2)

                send_command(ser, CMD_FORWARD)
                time.sleep(1.5)#补距离

                send_command(ser, CMD_SERVO_UP)

                # 2. 后退
                print(f"[ZONE_RELEASE] 后退 {ZONE_BACKWARD_SECONDS}s\n")
                send_command(ser,0x18)
                t0 = time.time()
                while time.time() - t0 < ZONE_BACKWARD_SECONDS:
                    if stop_event.is_set():
                        break
                    time.sleep(0.05)
                send_command(ser, CMD_STILL)

                # ---- 首球验证 (仅 cycle_idx == 0): 环形区域是否有球 ----
                retry_first = False
                if cycle_idx == 0:
                    while not result_queue.empty():
                        result_queue.get()

                    print(f"[ZONE_RELEASE] 首球验证: 检查环形区域是否有球 (最多{PUSH_VERIFY_FRAMES}帧)...\n")
                    push_ok = False
                    for i in range(PUSH_VERIFY_FRAMES):
                        t0 = time.time()
                        while result_queue.empty() and (time.time() - t0) < 0.3:
                            time.sleep(0.03)
                        if not result_queue.empty():
                            data = result_queue.get()
                            while not result_queue.empty():
                                data = result_queue.get()
                            frame, det_results = data
                            ball_count = count_balls_in_ring(frame, det_results)
                            if ball_count >= 1:
                                push_ok = True
                                print(f"  第{i+1}帧 环形区域检测到 {ball_count} 个球 → 推球成功\n")
                                break

                    if not push_ok:
                        first_ball_fail_count += 1
                        if first_ball_fail_count >= 2:
                            print(f"[ZONE_RELEASE] 首球验证连续失败{first_ball_fail_count}次 → 放弃首球，直接进入 cycle 1\n")
                            retry_first = False  # 不再重试，直接推进下一周期
                        else:
                            print(f"[ZONE_RELEASE] 首球验证失败 (第{first_ball_fail_count}次) → 重新搜索首球 ({FIRST_BALL})\n")
                            retry_first = True
                    else:
                        first_ball_fail_count = 0  # 验证成功，清零

                if retry_first:
                    target_cls = CLASS_ID.get(FIRST_BALL.lower())
                    current_target = None
                    loss_count = 0
                    last_cal_y2 = -1.0
                    observe_count = 0
                    search_phase = 'rotate'
                    search_phase_start = time.time()
                    state_name = "SEARCH"
                    prev_state = None
                    approach_entered = False
                    current_zone = None
                    zone_loss_count = 0
                    zone_last_cal_my = -1.0
                    zone_search_phase = 'rotate'
                    zone_search_phase_start = 0.0
                    zone_approach_entered = False
                    zone_top_kpt_indices = None
                    zone_align_done = False
                    zone_color_triggered = False
                    state_frame_count = 0
                    if state_queue is not None:
                        if state_queue.full():
                            try: state_queue.get_nowait()
                            except: pass
                        state_queue.put(state_name)
                    continue


                # ---- 正常推进下一周期 ----
                cycle_idx += 1
                if cycle_idx >= total_cycles:
                    print(f"[ZONE_RELEASE] 全部{total_cycles}个周期完成，退出\n")
                    break

                # ---- 切换到下一周期: 本方 + 黄 + 黑 ----
                target_cls = tuple(CLASS_ID[c] for c in SECONDARY_BALLS)
                ball_label = str(SECONDARY_BALLS)

                print(f"\n{'='*50}")
                print(f"[周期 {cycle_idx+1}/{total_cycles}] 目标球: {ball_label}")
                print(f"{'='*50}\n")

                # 重置所有状态变量
                current_target = None
                loss_count = 0
                last_cal_y2 = -1.0
                observe_count = 0
                search_phase = 'rotate'
                search_phase_start = time.time()
                state_name = "OBSERVE"
                prev_state = None
                approach_entered = False
                current_zone = None
                zone_loss_count = 0
                zone_last_cal_my = -1.0
                zone_search_phase = 'rotate'
                zone_search_phase_start = 0.0
                zone_approach_entered = False
                zone_top_kpt_indices = None
                zone_align_done = False
                zone_color_triggered = False

                # 通知录制节点
                if state_queue is not None:
                    if state_queue.full():
                        try: state_queue.get_nowait()
                        except: pass
                    state_queue.put(state_name)

                continue

            # ---- 每 10 帧简要日志 ----
            if frame_count % 10 == 0:
                if state_name in ("ZONE_SEARCH", "ZONE_CALIBRATE", "ZONE_APPROACH"):
                    if current_zone is not None:
                        kpts = current_zone.get("keypoints", [])
                        if len(kpts) >= 4:
                            tx = (kpts[1][0] + kpts[2][0]) / 2.0
                            ty = (kpts[1][1] + kpts[2][1]) / 2.0
                            dist = math.hypot(tx - CENTER_X, ty - CENTER_Y)
                            t_str = f"目标({tx:.0f},{ty:.0f}) dist={dist:.0f}"
                        else:
                            t_str = "kpt无效"
                    else:
                        t_str = "无"
                else:
                    t_str = f"y2={ball_bottom_center(current_target['box'])[1]:.0f}" if current_target else "无"
                print(f"  [{state_name}] 帧#{frame_count} target={t_str} loss={loss_count}")

    except Exception as e:
        print(f"[TEST 崩溃] {e}")
        import traceback
        traceback.print_exc()

    if ser is not None:
        send_command(ser, CMD_STILL)
        ser.close()
    print("[TEST] 已停止")