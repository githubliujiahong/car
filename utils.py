# rpi5_detect/utils.py
import time
import math
import cv2
import numpy as np

# ==========================================
# PID 控制器 (Proportional-Integral-Derivative)
# ==========================================
class PID:
    """
    通用 PID 控制算法类，用于平滑控制小车运动
    """
    def __init__(self, p=0, i=0, d=0, imax=0, d_filter_hz=20):
        self._kp = float(p)        # 比例系数
        self._ki = float(i)        # 积分系数
        self._kd = float(d)        # 微分系数
        self._imax = abs(imax) if imax != 0 else float('inf') # 积分限幅 (Anti-windup)
        self._d_filter_hz = d_filter_hz # 微分低通滤波频率
        self.reset()

    def reset(self):
        """重置控制器内部状态"""
        self._integrator = 0.0
        self._last_error = 0.0
        self._last_derivative = float('nan')
        self._last_t = 0

    def get_pid(self, error, scaler=1.0):
        """
        根据当前误差计算 PID 输出
        :param error: 目标值与当前值的偏差
        :param scaler: 输出缩放系数
        :return: 控制输出量
        """
        tnow = int(time.time() * 1000) # 毫秒
        dt = max(tnow - self._last_t, 0)
        delta_time = dt / 1000.0

        # 如果间隔时间过长 (如5秒)，认为系统断连，重置控制器
        if dt > 5000:
            self.reset()
            dt = 0

        # P: 比例输出
        output = error * self._kp * scaler

        # D: 微分输出 (结合低通滤波减少噪声影响)
        if self._kd != 0 and dt > 0:
            if math.isnan(self._last_derivative):
                derivative = 0.0
            else:
                derivative = (error - self._last_error) / delta_time
                # 一阶低通滤波
                alpha = 1 - math.exp(-delta_time * 2 * math.pi * self._d_filter_hz)
                derivative = self._last_derivative + alpha * (derivative - self._last_derivative)
            output += self._kd * derivative * scaler
            self._last_derivative = derivative

        # I: 积分输出 (带限幅)
        if self._ki != 0 and dt > 0:
            self._integrator += error * self._ki * delta_time
            self._integrator = max(min(self._integrator, self._imax), -self._imax)
            output += self._integrator * scaler

        self._last_error = error
        self._last_t = tnow
        return output

# ==========================================
# 几何与图像算法 (Geometric & Imaging Utils)
# ==========================================

def calculate_iou(box_a, box_b):
    """
    计算两个矩形框的交并比 (Intersection over Union)
    box 格式 (字典): {"box": [x1, y1, x2, y2], ...}
    """
    ax1, ay1, ax2, ay2 = box_a["box"]
    bx1, by1, bx2, by2 = box_b["box"]

    # 计算重叠区域坐标
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # 如果没有交集
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    # 计算面积
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter_area / (area_a + area_b - inter_area + 1e-6)

def filter_det_boxes(det_results, iou_threshold=0.2):
    """
    根据 IOU 过滤重复的检测结果 (类似简易版 NMS)
    :param det_results: 检测结果列表 (字典列表)
    :param iou_threshold: 重叠阈值
    :return: 过滤后的结果
    """
    class_groups = {}
    for res in det_results:
        class_id = res["class_id"]
        class_groups.setdefault(class_id, []).append(res)

    filtered = []
    for class_id, group in class_groups.items():
        # 按得分降序排序
        sorted_group = sorted(group, key=lambda x: x["confidence"], reverse=True)
        keep = []
        while sorted_group:
            current = sorted_group.pop(0)
            keep.append(current)
            # 剔除与当前框重叠过大的框
            sorted_group = [
                res for res in sorted_group
                if calculate_iou(current, res) < iou_threshold
            ]
        filtered.extend(keep)
    return filtered

# ==========================================
# 图像预处理工具 (Image Preprocessing Utils)
# ==========================================

class GammaCorrection:
    """
    伽马校正类，用于调节图像亮度
    使用查表法 (LUT) 以提高处理速度
    """
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.lut = self._build_lut(gamma)

    def _build_lut(self, gamma):
        """构建查找表"""
        # 避免除以 0
        inv_gamma = 1.0 / max(gamma, 0.01)
        # 生成 [0, 255] 的映射表
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        return table

    def apply(self, image):
        """对图像应用伽马校正"""
        if self.gamma == 1.0:
            return image
        return cv2.LUT(image, self.lut)

class CLAHECorrection:
    """
    CLAHE (对比度受限自适应直方图均衡化)
    用于增强图像细节并平衡局部亮度
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, image):
        """对 BGR 图像应用 CLAHE"""
        # 转换到 LAB 空间以仅对亮度通道 (L) 应用 CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        # 转换回 BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
