# rpi5_detect/nodes/recorder.py
import cv2
import time
import math
import os
import numpy as np
from config import *

def recorder_worker(recorder_queue, stop_event, state_queue=None):
    """
    负责视频录制与结果绘制的进程 (Recorder Node)
    核心功能：通过前 N 帧的实时时间差计算系统的“真实处理帧率”，并以此初始化 VideoWriter。
    """
    print(f"--- 录制节点已启动，正在监测系统真实 FPS... ---")
    
    # 颜色映射 (BGR 格式)
    COLORS = {
        RED: (0, 0, 255),      # 红色
        BLUE: (255, 0, 0),     # 蓝色
        YELLOW: (0, 255, 255), # 黄色
        BLACK: (0, 0, 0)       # 黑色
    }
    DEFAULT_COLOR = (0, 255, 0)

    # 动态 FPS 计算变量
    frame_buffer = []      # 暂存 warmup 期间的帧
    timestamps = []        # 记录每帧到达的时间点
    WARMUP_FRAMES = 15     # 收集前 N 帧用于计算真实 FPS
    FPS_MIN = 5.0          # 最低帧率限制
    FPS_MAX = 30.0         # 最高帧率限制 (防止视频过快)
    out = None
    encoder_failed = False  # 编码器初始化失败后不再重试
    real_fps = RECORD_FPS  # 默认使用配置值
    state_text = "NONE"    # 当前逻辑状态 (从 state_queue 读取)
    video_path = os.path.abspath(OUTPUT_VIDEO_PATH)  # 实际输出路径 (可能回退到 .avi)
    frame_idx = 0           # 调试：已处理帧计数
    write_count = 0         # 调试：已写入帧计数
    debug_interval = 30     # 每 N 帧打一次状态

    try:
        # 主循环：只要停止信号未发出，或者队列里还有剩余帧，就继续处理
        while True:
            # 退出条件：停止事件已发 且 队列已排空
            if stop_event.is_set() and recorder_queue.empty():
                print(f"[Recorder] 退出条件满足, 共处理{frame_idx}帧, 写入{write_count}帧")
                break

            # 队列为空时短休眠，避免占用 CPU
            if recorder_queue.empty():
                time.sleep(0.01)
                continue

            # 从队列获取数据 (frame, det_results, status_text)
            data = recorder_queue.get()
            frame_idx += 1
            current_time = time.time() # 记录当前帧到达录制节点的时间
            
            # 解析数据
            status_text = ""
            if len(data) == 3:
                frame, det_results, status_text = data
            else:
                frame, det_results = data

            # --- 0. 读取最新状态 (从逻辑节点) ---
            if state_queue is not None:
                while not state_queue.empty():
                    state_text = state_queue.get_nowait()
            
            # --- 1. 结果绘制 (OSD) ---
            for res in det_results:
                class_id = res["class_id"]
                conf = res["confidence"]
                x1, y1, x2, y2 = res["box"]
                keypoints = res.get("keypoints", [])

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                label = f"{CATEGORIES[int(class_id)]} {conf:.2f} ({cx},{cy})"
                color = COLORS.get(int(class_id), DEFAULT_COLOR)

                # 绘制边界框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 世界坐标系标注 (底部中心触地点)
                world = res.get("world_bottom")
                if world is not None:
                    world_label = f"W:({world[0]:.2f},{world[1]:.2f})m"
                    cv2.putText(frame, world_label, (int(x1), int(y2) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

                # --- 绘制关键点 (Pose / Zone Keypoints) ---
                is_zone = int(class_id) in (4, 5)  # red_safe_zone, blue_safe_zone
                for i, kp in enumerate(keypoints):
                    kx, ky, kconf = kp
                    if kconf > 0.4: # 关键点置信度阈值
                        cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
                        cv2.circle(frame, (int(kx), int(ky)), 5, (255, 255, 255), 1)
                        # 区域关键点：标序号（醒目）
                        if is_zone:
                            idx_label = str(i + 1)
                            # 黑色背景衬底，确保任何光照下都清晰
                            (tw, th), _ = cv2.getTextSize(idx_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                            cv2.rectangle(frame,
                                          (int(kx) + 6, int(ky) - th - 6),
                                          (int(kx) + 6 + tw + 2, int(ky) - 2),
                                          (0, 0, 0), -1)
                            cv2.putText(frame, idx_label, (int(kx) + 7, int(ky) - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

                # --- 区域四关键点中点 & 距离 / avgY 显示 ---
                if is_zone and len(keypoints) >= 4:
                    kpts = keypoints[:4]
                    # 计算四个关键点中点
                    mx = sum(kp[0] for kp in kpts) / 4.0
                    my = sum(kp[1] for kp in kpts) / 4.0
                    # 到画面中心的像素距离
                    dist_px = math.hypot(mx - CENTER_X, my - CENTER_Y)

                    # 绘制中点标记 (手绘十字，兼容所有OpenCV版本)
                    mix, miy = int(mx), int(my)
                    CROSS_SIZE = 8
                    cv2.line(frame, (mix - CROSS_SIZE, miy), (mix + CROSS_SIZE, miy),
                             (0, 255, 255), 2)
                    cv2.line(frame, (mix, miy - CROSS_SIZE), (mix, miy + CROSS_SIZE),
                             (0, 255, 255), 2)
                    # 中点到画面中心的连线
                    cv2.line(frame, (mix, miy), (CENTER_X, CENTER_Y),
                             (0, 255, 255), 1, cv2.LINE_AA)

                    # avgY 与阈值对比 (调 ZONE_STOP_Y_THRESHOLD 用)
                    y_gap = my - ZONE_STOP_Y_THRESHOLD
                    gap_color = (0, 255, 0) if y_gap >= 0 else (0, 165, 255)  # 绿=达标, 橙=还差
                    gap_label = f"avgY={my:.0f} th={ZONE_STOP_Y_THRESHOLD} gap={y_gap:+.0f}"
                    cv2.putText(frame, gap_label, (mix + 10, miy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, gap_color, 2)

            if status_text:
                cv2.putText(frame, status_text, (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # 显示分辨率 (右上角)
            cv2.putText(frame, f"{FRAME_WIDTH}x{FRAME_HEIGHT}", (FRAME_WIDTH - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 当前状态 (左上角)
            cv2.putText(frame, state_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_HEIGHT), (255, 255, 255), 1)
            cv2.rectangle(frame, (VERIFY_ZONE_X[0], VERIFY_ZONE_Y[0]),
                          (VERIFY_ZONE_X[1], VERIFY_ZONE_Y[1]), (0, 255, 255), 2)

            # 抓取区域 (球底部进入此区域触发抓取)
            cv2.rectangle(frame, (GRAB_ZONE_X1, GRAB_ZONE_Y1),
                          (GRAB_ZONE_X2, GRAB_ZONE_Y2), (0, 255, 0), 2)
            cv2.putText(frame, "GRAB", (GRAB_ZONE_X1, GRAB_ZONE_Y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 抓取区域扩展黑框: 外60px, 内(上30 其余40)
            cv2.rectangle(frame,
                          (GRAB_ZONE_X1 - 40, GRAB_ZONE_Y1 - 30),
                          (GRAB_ZONE_X2 + 40, GRAB_ZONE_Y2 + 40),
                          (0, 0, 0), 2)
            cv2.rectangle(frame,
                          (GRAB_ZONE_X1 - 60, GRAB_ZONE_Y1 - 60),
                          (GRAB_ZONE_X2 + 60, GRAB_ZONE_Y2 + 60),
                          (0, 0, 0), 3)

            # ZONE_STOP_Y_THRESHOLD 参考线 (虚线效果: 短线段)
            th_y = ZONE_STOP_Y_THRESHOLD
            for sx in range(0, FRAME_WIDTH, 16):
                cv2.line(frame, (sx, th_y), (sx + 8, th_y), (0, 255, 0), 1)
            cv2.putText(frame, f"STOP_Y={th_y}", (FRAME_WIDTH - 120, th_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # --- 2. 动态录制逻辑 ---
            if out is None and not encoder_failed:
                # 处于样本收集阶段
                frame_buffer.append(frame.copy())
                timestamps.append(current_time)

                # 调试: warmup 进度
                if len(timestamps) % 5 == 0 or len(timestamps) == 1:
                    print(f"[Recorder] warmup {len(timestamps)}/{WARMUP_FRAMES}  qsize={recorder_queue.qsize()}")

                if len(timestamps) >= WARMUP_FRAMES:
                    total_duration = timestamps[-1] - timestamps[0]
                    if total_duration > 0:
                        measured_fps = (len(timestamps) - 1) / total_duration
                        # 限制在合理范围内, 防止帧率过高或过低
                        real_fps = max(FPS_MIN, min(measured_fps, FPS_MAX))
                    else:
                        real_fps = RECORD_FPS

                    print(f"[Recorder] 实测帧率: {measured_fps:.1f} FPS → 限幅为: {real_fps:.1f} FPS")

                    # 编码器回退链: mp4v → XVID → MJPG (树莓派兼容性)
                    codec_chain = [
                        ('mp4v', '.mp4'),
                        ('XVID', '.avi'),
                        ('MJPG', '.avi'),
                    ]
                    base, _ = os.path.splitext(video_path)

                    for codec_name, ext in codec_chain:
                        try_path = video_path if ext == '.mp4' else base + ext
                        fourcc = cv2.VideoWriter_fourcc(*codec_name)
                        print(f"[Recorder] 尝试 VideoWriter: {try_path} fourcc={codec_name}")
                        out = cv2.VideoWriter(try_path, fourcc, real_fps, (FRAME_WIDTH, FRAME_HEIGHT))
                        if out.isOpened():
                            video_path = try_path
                            print(f"[Recorder] ✓ {codec_name} 编码器可用 → {video_path}")
                            break
                        else:
                            out.release()
                            out = None
                            print(f"[Recorder] ✗ {codec_name} 不可用")

                    if out is None:
                        print(f"[Recorder] 错误: 所有编码器均不可用, 放弃录制")
                        encoder_failed = True
                    else:
                        print(f"[Recorder] VideoWriter.isOpened() = True, 回写{len(frame_buffer)}帧缓存...")
                        # 写入 warmup 期间缓存的帧
                        for i, f in enumerate(frame_buffer):
                            out.write(f)
                            if i % 5 == 0:
                                print(f"[Recorder]   回写缓存帧 {i+1}/{len(frame_buffer)}")
                        frame_buffer = []
                        timestamps = []

                        print(f"[Recorder] VideoWriter 初始化完成, 开始持续录制...")
            elif out is not None:
                out.write(frame)
                write_count += 1
                if write_count % debug_interval == 0:
                    print(f"[Recorder] 已写入 {write_count} 帧 (共处理 {frame_idx} 帧)")
            # out 为 None 且已过 warmup → 编码器初始化失败，跳过写入
            elif encoder_failed:
                if frame_idx % debug_interval == 0:
                    print(f"[Recorder] 编码器失败，跳过写入 (frame #{frame_idx})")
        print(f"[Recorder] 录制进程已安全退出")
    except Exception as e:
        print(f"录制节点出现异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if out is not None:
            out.release()
            size_mb = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0
            print(f"--- 录制节点已安全关闭，视频已保存: {video_path} ({size_mb:.1f} MB) ---")
        else:
            print("--- 录制节点停止，未生成视频 ---")
