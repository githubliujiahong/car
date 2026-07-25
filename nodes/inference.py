# rpi5_detect/nodes/inference.py
import numpy as np
import time
import cv2
import ncnn  # 替换原来的 onnxruntime
from config import CONFIDENCE_THRESHOLD, NMS_THRESHOLD, CATEGORIES
from config import CAMERA_K, CAMERA_DIST, CAMERA_HEIGHT
from coordinate_transform import pixel_to_world

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    自适应填充与缩放 (Letterbox) - 保持不变
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)

def inference_worker(frame_queue, result_queue, recorder_queue, stop_event):
    """
    推理节点 (Inference Node) - 升级为极致榨干树莓派性能的 YOLO11-Pose NCNN 版本
    """
    print("--- 推理节点已启动 (YOLO11-Pose NCNN 强力加速版) ---")
    
    # 配置文件路径 (请将编译出的这两个文件扔到树莓派对应的目录下)
    param_path = "/home/pi/Desktop/best_ncnn_model/model.ncnn.param"
    bin_path = "/home/pi/Desktop/best_ncnn_model/model.ncnn.bin"
    
    NUM_CLASSES = len(CATEGORIES)  # 自动与 config.py 同步，当前为6类
    
    # 1. 初始化 NCNN 推理网络
    try:
        net = ncnn.Net()
        # 针对树莓派 5 (ARM Cortex-A76 4核) 的极限优化开关
        net.opt.use_vulkan_compute = False  # 树莓派CPU足够强，且NCNN的CPU NEON优化极佳，开CPU最稳
        net.opt.num_threads = 4             # 榨干树莓派5全部4个核心

        # load_param / load_model 返回 0 表示成功，-1 表示失败
        # NCNN C++ 层失败时只打 stderr 警告，不抛 Python 异常！
        ret_p = net.load_param(param_path)
        if ret_p != 0:
            raise RuntimeError(f"无法加载 .param 文件: {param_path} (错误码={ret_p})")

        ret_m = net.load_model(bin_path)
        if ret_m != 0:
            raise RuntimeError(f"无法加载 .bin 文件: {bin_path} (错误码={ret_m})")

        print("🏆 NCNN Pose 模型在树莓派5上加载成功！")
    except Exception as e:
        print(f"NCNN 推理引擎初始化失败: {e}")
        stop_event.set()
        return
    
    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.002)
            continue

        frame = frame_queue.get()

        try:
            # 2. 图像自适应缩放预处理
            img, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640))

            # 3. 将 OpenCV 的 BGR 图像直接转换为 NCNN 内部专用的 Mat 结构 (底层自带高效色调转换和通道重排)
            ncnn_img = ncnn.Mat.from_pixels(
                img,
                ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                img.shape[1],
                img.shape[0]
            )

            # 减去均值并除以方差（等价于原先的 / 255.0，因为这里没有减去均值所以填0）
            mean_vals = [0.0, 0.0, 0.0]
            norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
            ncnn_img.substract_mean_normalize(mean_vals, norm_vals)

            # 4. 每帧创建新的 Extractor (NCNN 不支持复用 extractor!)
            ex = net.create_extractor()
            ex.input("in0", ncnn_img)
            ret, ncnn_output = ex.extract("out0")

            # ⚠️ 关键: 检查推理是否成功, 失败则跳过本帧
            if ret != 0:
                print(f"[推理错误] extract 返回码={ret}, 跳过本帧")
                continue

            # 5. 将 NCNN 的 Mat 转换为 NumPy 矩阵进行后处理
            output = np.array(ncnn_output)
            # YOLO11-Pose 输出: [4(bbox) + NUM_CLASSES + 3*num_keypoints, num_anchors]
            # 例: 6类4关键点 = [4+6+12, 8400] = [22, 8400]
            predictions = output.T

            # 6. 数据切片分离
            boxes = predictions[:, :4]                                 # [cx, cy, w, h]
            scores = predictions[:, 4 : 4 + NUM_CLASSES]               # 类别概率
            keypoints = predictions[:, 4 + NUM_CLASSES :]              # 关键点数据

            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)

            # 阈值初筛
            mask = confidences > CONFIDENCE_THRESHOLD
            boxes = boxes[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            keypoints = keypoints[mask]

            det_results = []
            if len(boxes) > 0:
                # 还原边界框到原图
                x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = (x_c - w / 2 - dw) / ratio[0]
                y1 = (y_c - h / 2 - dh) / ratio[1]
                x2 = (x_c + w / 2 - dw) / ratio[0]
                y2 = (y_c + h / 2 - dh) / ratio[1]

                # 转为 numpy float32 数组 (NMSBoxes 需要)
                nms_boxes = np.column_stack((x1, y1, x2 - x1, y2 - y1)).astype(np.float32)
                nms_scores = confidences.astype(np.float32)

                indices = cv2.dnn.NMSBoxes(
                    bboxes=nms_boxes,
                    scores=nms_scores,
                    score_threshold=CONFIDENCE_THRESHOLD,
                    nms_threshold=NMS_THRESHOLD
                )

                # 兼容 OpenCV 不同版本的 NMSBoxes 返回值：
                # - OpenCV < 4.5.5: 直接返回 (N, 1) 的 numpy 数组
                # - OpenCV >= 4.5.5: 返回 tuple (array,)，空结果为空的 tuple ()
                if isinstance(indices, tuple):
                    indices = indices[0] if len(indices) > 0 else np.array([])

                # 7. 过滤并封装最终字典，维持原来的接口
                if len(indices) > 0:
                    for idx in indices.flatten():
                        b_x1, b_y1, b_x2, b_y2 = x1[idx], y1[idx], x2[idx], y2[idx]
                        cls_id = int(class_ids[idx])

                        raw_kpts = keypoints[idx]
                        mapped_kpts = []

                        # 遍历关键点还原真实坐标 (每个关键点由 x, y, conf 三元组构成)
                        num_kpts = len(raw_kpts) // 3  # 安全计算实际关键点数量
                        for k in range(num_kpts):
                            kx = (raw_kpts[k * 3] - dw) / ratio[0]
                            ky = (raw_kpts[k * 3 + 1] - dh) / ratio[1]
                            kconf = raw_kpts[k * 3 + 2]
                            mapped_kpts.append([float(kx), float(ky), float(kconf)])

                        det_results.append({
                            "class_id": cls_id,
                            "confidence": float(confidences[idx]),
                            "box": [float(b_x1), float(b_y1), float(b_x2), float(b_y2)],
                            "keypoints": mapped_kpts
                        })

            # ---- 世界坐标变换: 批量计算所有检测目标的底部中心世界坐标 ----
            if det_results:
                bottom_pts = []
                for res in det_results:
                    x1, y1, x2, y2 = res["box"]
                    cx = (x1 + x2) / 2.0
                    by = y2  # 底部中心 (触地点)
                    bottom_pts.append([cx, by])

                world_xy = pixel_to_world(
                    np.array(bottom_pts, dtype=np.float64),
                    CAMERA_K, CAMERA_DIST, CAMERA_HEIGHT
                )

                # world_xy shape: (N, 2) for N >= 2, or (2,) for single point
                if world_xy.ndim == 1:
                    world_xy = world_xy.reshape(1, 2)

                for i, res in enumerate(det_results):
                    wx, wy = world_xy[i]
                    if np.isfinite(wx) and np.isfinite(wy):
                        res["world_bottom"] = [float(wx), float(wy)]
                    else:
                        res["world_bottom"] = None  # 地平线附近无法计算

            # 回传至逻辑队列
            if result_queue.full():
                try: result_queue.get_nowait()
                except: pass
            result_queue.put((frame, det_results))

            # 直连录制队列 (绕过逻辑的 1 秒阻塞，录制满帧率)
            if recorder_queue is not None:
                if recorder_queue.full():
                    try: recorder_queue.get_nowait()
                    except: pass
                recorder_queue.put((frame, det_results, ""))

        except Exception as e:
            print(f"[推理异常] 处理帧时出错: {e}, 跳过本帧继续运行")
            import traceback
            traceback.print_exc()
            continue
    
    print("--- 推理节点已停止 ---")
