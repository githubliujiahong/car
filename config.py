# rpi5_detect/config.py
import math
import numpy as np

# ==========================================
# ⚙️ 阵营 & 启动配置 (改这里！)
# ==========================================
if_red = True                     # True = 红队, False = 蓝队
INIT_FORWARD_SECONDS = 2.0        # 启动后直行时长 (秒)

# 根据阵营自动推导 (无需手动改)
FIRST_BALL = 'red' if if_red else 'blue'
SECONDARY_BALLS = ['red', 'yellow', 'black'] if if_red else ['blue', 'yellow', 'black']

# ==========================================
# 状态机常量定义 (State Machine Constants)
# ==========================================
class State:
    START    = 0   # 等待开始
    FIND     = 1   # 寻找目标 (第一阶段：找本方颜色球)
    STOPA    = 2   # 停止准备抓取 (第一阶段)
    CHECK    = 3   # 检查抓取结果 (第一阶段)
    XYW      = 4   # 重新搜索 (第一阶段失败后)
    CENTER   = 5   # 回到中心区域
    DELIVER  = 6   # 搬运目标至安全区
    BACKWARD = 7   # 放置后后退
    REFIND   = 8   # 二次寻找目标 (优先级：黄 > 黑 > 颜色球)
    STOPB    = 9   # 停止准备抓取 (第二阶段)
    RECHECK  = 10  # 检查抓取结果 (第二阶段)
    REXYW      = 11  # 重新搜索 (第二阶段失败后)
    CALIBRATE  = 12  # 坐标校准旋转 (仅一次)

# ==========================================
# 硬件与通信配置 (Hardware & Communication)
# ==========================================
SERIAL_PORT = "/dev/ttyUSB0" # 树莓派5 默认串口或 /dev/serial0
BAUD_RATE = 115200           # 波特率需与下位机一致

# = :摄像机配置 (Camera Configuration)
# ==========================================
FRAME_WIDTH = 820
FRAME_HEIGHT = 616

# 图像预处理开关
ENABLE_GAMMA = True          # 是否开启伽马校正
CAMERA_GAMMA = 1.5           # 伽马校正值 (降低以减少暗部噪点)

ENABLE_CLAHE = True          # 是否开启 CLAHE
CLAHE_CLIP_LIMIT = 1.2       # 显著降低阈值 (原为 3.0)，抑制噪点放大
CLAHE_GRID_SIZE = (8, 8)

ENABLE_DENOISE = True        # 新增：开启平滑去噪

# --- 抗闪烁深度配置 ---
# 50: 对应 50Hz 电网 (如中国、欧洲)
# 60: 对应 60Hz 电网 (如美国、日本)
FLICKER_FREQ = 50

# ==========================================
# 相机标定参数 (Camera Calibration)
# 用于 pixel_to_world 像素坐标 → 世界坐标变换
# ==========================================
CAMERA_K = np.array([
    [372.30907323, 0.,          412.95512148],
    [0.,          372.58378073, 314.86756525],
    [0.,          0.,           1.          ]
], dtype=np.float64)

CAMERA_DIST = np.array(
    [-0.07987671, 0.09794424, -0.00130723, 0.00139241, -0.05037346],
    dtype=np.float64
)  # k1, k2, p1, p2, k3

CAMERA_HEIGHT = 0.1  # 相机距地面高度 (米)



# ==========================================
# AI 推理配置 (Inference Configuration)
# ==========================================
MODEL_SIZE = (640, 640)      # 模型输入尺寸
CONFIDENCE_THRESHOLD = 0.6   # 置信度阈值
NMS_THRESHOLD = 0.45         # 非极大值抑制阈值

# ==========================================
# 类别定义 (Categories)
# 注意：需与模型训练时的标签顺序完全一致
# YOLO11-Pose 4关键点模型 —— 6类 (与 dataset.yaml 对齐)
# ==========================================
CATEGORIES = [
    'black','blue','red','yellow','red_safe_zone','blue_safe_zone'
]

# ==========================================
# 语义常量与映射 (Semantic Mapping)
# ==========================================
# 自动生成类别ID映射 (忽略大小写)
CLASS_ID = {name.lower(): idx for idx, name in enumerate(CATEGORIES)}

# 常用类别别名，方便代码调用
BLACK  = CLASS_ID['black']
BLUE   = CLASS_ID['blue']
RED    = CLASS_ID['red']
YELLOW = CLASS_ID['yellow']

RED_SAFE_ZONE   = CLASS_ID['red_safe_zone']
BLUE_SAFE_ZONE  = CLASS_ID['blue_safe_zone']

# 注意: red_start_zone / blue_start_zone / center 在新 Pose 6类模型中不存在

# ==========================================
# 决策与控制参数 (Decision & Control)
# ==========================================
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
ALIGN_DEADZONE = 80          # 中轴线左右死区 (像素)
GRAB_THRESHOLD_Y = 550       # 触发抓取的 Y 坐标 (越近数值越大)

# 抓取成功判定区域 (Verification Zone)
VERIFY_ZONE_X = (270, 570)   # X 范围
VERIFY_ZONE_Y = (550, 616)   # Y 范围

# 指令编码定义
CMD_FORWARD    = 0x10
CMD_TURN_LEFT  = 0x11
CMD_TURN_RIGHT = 0x12
CMD_STILL      = 0x20
CMD_SERVO_UP    = 0x15
CMD_SERVO_DOWN  = 0x16
CMD_SERVO_MID   = 0x14  # 舵机中间角度
CMD_BACKWARD    = 0x13  # ⚠️ 需与下位机确认指令码
CMD_RESET      = 0x19

# ==========================================
# 逻辑节点控制参数 (Logic Control Parameters)
# ==========================================

# ---- 坐标校准 ----
ROTATION_DEG_PER_SEC = 180.0 / 5.0  # 旋转速度 36°/s (实测转两倍角度, 转速翻倍使校准时长减半)
ANGLE_DEADZONE_DEG = 10.0           # 角度校准死区 (±°)
GRAB_FORWARD_COMPENSATION = 1.0     # 进入抓取区后补偿前进时长 (s)

# 决策-执行节奏 (秒) — 执行/停止交替
LOGIC_ACTION_DURATION = 0.2
LOGIC_STOP_DURATION   = 0.2

# ---- 抓球区域 (Grab Zone) ----
# 当球底部中心 (cx, y2) 进入此矩形区域时触发抓取
GRAB_ZONE_X1 = 330
GRAB_ZONE_X2 = 450
GRAB_ZONE_Y1 = 500
GRAB_ZONE_Y2 = 610

# ---- 目标对齐 ----
ALIGN_TOLERANCE_X = 40             # 目标中心与画面中心的水平容差 (像素)
SAFE_ZONE_ALIGN_TOLERANCE_X = 15  # 安全区入口中点与画面中心的水平容差

# ---- 安全区参数 ----
# 安全区4个关键点中，入口正面两点的 0-indexed 索引
# ⚠️ 用 check_safe_zone_kpts.py 验证后填入正确值！
SAFE_ZONE_FRONT_KPT_INDICES = [2, 3]  # 待验证
# 当入口正面两点的平均 y 坐标 > 此值时，认为已到达释放距离
SAFE_ZONE_RELEASE_Y_THRESHOLD = 580

# ---- 释放后退 ----
RELEASE_BACK_CYCLES = 5  # 释放后后退周期数 (~2秒)

# ---- 搜索 ----
SEARCH_DIR_SWITCH_CYCLES = 15  # 单向搜索周期数后换方向

# ---- 卡住检测 ----
STUCK_MAX_CYCLES = 30  # 同一状态超过此周期数强制降级 (~12秒)

# ==========================================
# 区域搬运参数 (Zone Delivery)
# ==========================================
# 己方安全区颜色: 'red' 或 'blue'
# 红队 → red_safe_zone，蓝队 → blue_safe_zone
OWN_SAFE_ZONE_COLOR = 'red' if if_red else 'blue'

# 区域关键点中点 y 坐标 > 此值时停止前进
ZONE_STOP_Y_THRESHOLD = 380

# 放置后后退时长 (秒)
ZONE_BACKWARD_SECONDS = 2.0

# 区域搜索旋转方向: 'left' / 'right'
ZONE_SEARCH_ROTATE_DIR = 'left'

# 区域搜索参数
ZONE_SEARCH_ROTATE_DURATION = 0.15   # 每次旋转时长 (s)
ZONE_SEARCH_OBSERVE_DURATION = 1.0   # 停转观察时长 (s)

# 区域丢失容错帧数
ZONE_LOSS_TOLERANCE = 10

# ---- 区域方向校准 (正对近边) ----
# 释放前(抬舵机后、前进推球前)让车头垂直于"离车最近两关键点的连线"。
# 用几何(世界 Y 最小的两点)自动确定近边, 无需依赖 SAFE_ZONE_FRONT_KPT_INDICES。
ZONE_ORIENT_DEADZONE_DEG = 30   # 近边夹角小于此值认为已基本正对, 跳过 (偏大容忍, 只求大致)
ZONE_ORIENT_MAX_DURATION = 1.5    # 单次方向校准最大旋转时长 (s), 防一次转太多
ZONE_ORIENT_TURN_INVERT = False   # 实测转反了就置 True (倒装/接线方向兜底开关)

# ---- 像素校准参数 (Pixel Calibration) ----
PIXEL_DEADZONE = 50               # 水平像素死区: |cx - CENTER_X| < 此值认为已对准
CALIB_DURATION = 0.12             # 校准固定旋转时长 (s): 偏左→右转, 偏右→左转

# ---- 球搜索参数 (Ball Search) ----
BALL_SEARCH_ROTATE_DURATION = 0.25   # 每次旋转时长 (s)，短脉冲减少过冲
BALL_SEARCH_OBSERVE_DURATION = 0.8   # 每次停转观察时长 (s)
BALL_SEARCH_ROTATE_DIR = 'left'      # 搜索旋转方向: 'left' / 'right'

# ---- 区域前方停车 & 自动对准 ----
ZONE_TARGET_OFFSET_Y = 5            # 目标点向外偏移 (y+5, kp后方)
ZONE_STOP_DIST = 60                 # 目标点距画面中心 < 此值时停车 (像素)
ZONE_ALIGN_TURN_DURATION = 0.25     # 自动对准短转时长 (s)
KP_CONF_THRESHOLD = 0.4             # 关键点置信度阈值: 低于此值认为不可见

ENABLE_RECORDER=False
RECORD_FPS = 30.0
OUTPUT_VIDEO_PATH='out.mp4'



