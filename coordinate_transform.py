"""
坐标变换模块：像素坐标 → 世界坐标（水平地面）

原理：小孔成像 + 相似三角形
适用条件：
  - 目标点在水平地面（Z = 0）
  - 相机光轴水平（平行于地面），无俯仰、无横滚
  - 世界坐标系原点在相机正下方的地面投影点
  - 非鱼眼相机，畸变模型为 OpenCV 标准模型（径向 + 切向）

Geometry:
  相机在世界坐标系: (0, 0, H)
  光轴沿世界 +Y 方向（正前方）
  相机坐标系 → 世界坐标系:
    X_cam = X_world          (右)
    Y_cam = -Z_world + H     (下)
    Z_cam = Y_world          (前)

  对地面点 (X, Y, 0):
    P_cam = (X, H, Y)

  小孔投影:
    u = fx * X / Y + cx
    v = fy * H / Y + cy

  逆变换:
    Y = H / y_norm
    X = x_norm * Y

  其中 (x_norm, y_norm) 为去畸变后的归一化坐标。
"""

import numpy as np
import cv2


def pixel_to_world(points, K, dist_coeffs=None, H=0.1):
    """
    像素坐标 → 世界坐标（地面投影）

    Parameters
    ----------
    points : np.ndarray | tuple | list
        像素坐标，支持三种格式：
        - 单点: (u, v) 元组/列表 或 shape (2,) 的数组
        - 多点: shape (N, 2) 的数组
        - OpenCV 格式: shape (N, 1, 2) 的数组
    K : np.ndarray
        内参矩阵 (3, 3):
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    dist_coeffs : np.ndarray | None
        畸变系数。支持 OpenCV 标准格式：
        (k1, k2, p1, p2 [, k3 [, k4, k5, k6]])
        传入 None 或全零数组表示无畸变。
    H : float
        相机距水平面的高度，单位: 米。默认 0.1 (10 cm)。

    Returns
    -------
    world : np.ndarray
        世界坐标 (X, Y)，单位: 米。
        - 输入单点 → 返回 shape (2,)
        - 输入多点 → 返回 shape (N, 2)
        X: 水平方向（右为正），Y: 深度方向（前为正）

    Notes
    -----
    - 地平线位于 v = cy 处。在该行上 y_norm = 0，Y → ∞，返回 inf。
    - 地平线上方的点 (v < cy) 理论上对应地面后方，Y 为负值。调用方应自行过滤。
    """
    # ---- 1. 输入格式化 ----
    points = np.asarray(points, dtype=np.float64)
    single_point = (points.ndim == 1)

    if single_point:
        points = points.reshape(1, 1, 2)
    elif points.ndim == 2:
        points = points.reshape(-1, 1, 2)
    # else: ndim == 3, 已是 (N, 1, 2)，不改动

    # ---- 2. 畸变系数处理 ----
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4, dtype=np.float64)
    else:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel()

    # ---- 3. 去畸变 → 归一化相机坐标 ----
    # 无畸变时直接算，避免 cv2.undistortPoints 的开销
    if np.allclose(dist_coeffs, 0):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = points[..., 0, 0]
        v = points[..., 0, 1]
        xn = (u - cx) / fx
        yn = (v - cy) / fy
    else:
        # cv2.undistortPoints 默认输出归一化坐标 (不传 P 时)
        # 即 x' = (u_undist - cx)/fx, y' = (v_undist - cy)/fy
        undistorted = cv2.undistortPoints(points, K, dist_coeffs)
        xn = undistorted[..., 0, 0]
        yn = undistorted[..., 0, 1]

    # ---- 4. 逆投影到地平面 ----
    # xn = X / Y,  yn = H / Y
    # → Y = H / yn,  X = xn * Y
    # 地平线 yn=0 时 Y→∞，抑制除零警告，由调用方自行处理 inf
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = H / yn
        X = xn * Y

    world = np.stack([X, Y], axis=-1)

    if single_point:
        world = world[0]

    return world


def world_to_pixel(points_XY, K, dist_coeffs=None, H=0.1):
    """
    世界坐标（地面投影）→ 像素坐标（正向投影，用于验证）

    Parameters
    ----------
    points_XY : np.ndarray | tuple | list
        世界坐标 (X, Y)，单位: 米。
        - 单点: (X, Y) 元组/列表 或 shape (2,) 的数组
        - 多点: shape (N, 2) 的数组
    K : np.ndarray
        内参矩阵 (3, 3)。
    dist_coeffs : np.ndarray | None
        畸变系数，同 pixel_to_world。
    H : float
        相机高度，单位: 米。

    Returns
    -------
    pixels : np.ndarray
        像素坐标 (u, v)。
    """
    points_XY = np.asarray(points_XY, dtype=np.float64)
    single_point = (points_XY.ndim == 1)

    if single_point:
        points_XY = points_XY.reshape(1, 2)

    X = points_XY[..., 0]
    Y = points_XY[..., 1]

    # 归一化坐标（无畸变时）
    xn = X / Y
    yn = H / Y

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u_undist = fx * xn + cx
    v_undist = fy * yn + cy

    # 无畸变 → 直接返回
    if dist_coeffs is None or np.allclose(np.asarray(dist_coeffs), 0):
        pixels = np.stack([u_undist, v_undist], axis=-1)
    else:
        # 正向加畸变：用 cv2.projectPoints
        # 先把归一化坐标变成 3D 相机坐标（取 Z = Y）
        obj_points = np.stack([X, np.full_like(X, H), Y], axis=-1).reshape(-1, 1, 3)
        rvec = np.zeros((3,), dtype=np.float64)   # 无旋转
        tvec = np.zeros((3,), dtype=np.float64)   # 相机即世界原点
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K,
                                         np.asarray(dist_coeffs, dtype=np.float64).ravel())
        pixels = projected.reshape(-1, 2)

    if single_point:
        pixels = pixels[0]

    return pixels


# ============================================================
# 示例 & 自测
# ============================================================
if __name__ == "__main__":

    K = np.array([
        [372.30907323, 0.,          412.95512148],
        [0.,          372.58378073, 314.86756525],
        [0.,          0.,           1.          ],
    ], dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 畸变系数（演示用，实际中来自标定）
    dist = np.array([-0.07987671, 0.09794424, -0.00130723, 0.00139241, -0.05037346])  # k1, k2, p1, p2, k3

    H = 0.1  # 相机高 10 cm

    print("=" * 55)
    print("坐标变换自测")
    print("=" * 55)

    # ---- 测试 1: 图像中心正下方，前方 1m 处 ----
    u_test, v_test = cx, cy + 30  # 中心偏下
    Xw, Yw = pixel_to_world((u_test, v_test), K, dist, H)
    print(f"\n[测试1] 像素 ({u_test:.1f}, {v_test:.1f}) → 世界 ({Xw:.3f}, {Yw:.3f}) m")

    # 正向验证：把世界坐标投影回去
    u_back, v_back = world_to_pixel((Xw, Yw), K, dist, H)
    print(f"        正向验证 → 像素 ({u_back:.3f}, {v_back:.3f})")
    print(f"        误差: Δu={u_back - u_test:.4f}, Δv={v_back - v_test:.4f}")

    # ---- 测试 2: 多个点（一条水平线上的点） ----
    print("\n[测试2] 批量转换 — 画面最低行 (v=610) 上均匀采样 5 个点:")
    v_batch = 610
    u_batch = np.linspace(50, 770, 5)
    pts = np.stack([u_batch, np.full_like(u_batch, v_batch)], axis=-1)
    world_batch = pixel_to_world(pts, K, dist, H)
    for i, (u_i, v_i) in enumerate(zip(u_batch, [v_batch]*5)):
        print(f"  ({u_i:6.1f}, {v_i:.0f}) → (X={world_batch[i,0]:7.3f}, Y={world_batch[i,1]:7.3f}) m")

    # ---- 测试 3: 无畸变情况 ----
    print("\n[测试3] 无畸变 — 中心点前 0.5m:")
    Xw3, Yw3 = pixel_to_world((cx, cy + 60), K, None, H)
    print(f"  像素 (cx, cy+60) → 世界 ({Xw3:.3f}, {Yw3:.3f}) m")
    # 手动验算: Y = fy * H / (v - cy) = fy * 0.1 / 60
    Y_expected = fy * H / 60
    print(f"  手动验算 Y = {fy:.1f} * {H} / 60 = {Y_expected:.3f} m ✓")

    # ---- 测试 4: 地平线边界 ----
    print(f"\n[测试4] 地平线 (v=cy={cy:.0f}) 上的点:")
    X_horizon, Y_horizon = pixel_to_world((cx, cy), K, None, H)
    print(f"  Y = {Y_horizon} (应为 inf)")

    print("\n" + "=" * 55)
    print("自测完毕")
    print("=" * 55)
