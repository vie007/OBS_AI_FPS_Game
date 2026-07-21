"""
Watermelon Aimbot — ONNX + DirectML 推理引擎
用法: 将此文件替换 run.py，即可用 ONNX 模型 + DirectML GPU 加速运行

支持的推理后端（按优先级）:
  1. DmlExecutionProvider  — DirectML，任意 DX12 GPU（AMD/Intel/NVIDIA 均可）
  2. CPUExecutionProvider  — 纯 CPU 回退
"""
from __future__ import annotations

# ──────────────────── 全局异常捕获（防止 exe 闪退） ────────────────────
import sys
import os

# 打包为 exe 时，确保工作目录为 exe 所在目录
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))

def _global_exception_handler(exc_type, exc_value, exc_traceback):
    """捕获所有未处理的异常，包括模块导入阶段的错误"""
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    input("\n[错误] 程序异常退出，按回车键关闭...")
    sys.exit(1)

sys.excepthook = _global_exception_handler

import time
import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
import torch

from logic.config_watcher import cfg
from logic.checks import run_checks
from logic.logger import logger


# ─────────────────────────── ONNX 推理引擎 ───────────────────────────

class OnnxDetector:
    """封装 ONNX Runtime 的 YOLO 推理，支持 DirectML 加速"""

    def __init__(self, model_source, img_size: int, conf_thresh: float):
        """
        参数:
            model_source: 模型文件路径(str) 或 模型二进制数据(bytes)
        """
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = 0.50

        # 初始化 ONNX Runtime，优先使用 DirectML
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 避免多线程冲突
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 4

        # 支持从文件路径或 bytes 加载模型
        if isinstance(model_source, bytes):
            logger.info("[ONNX] 从内存加载模型")
            self.session = ort.InferenceSession(
                model_source,
                sess_options=sess_options,
                providers=providers,
            )
        else:
            self.session = ort.InferenceSession(
                model_source,
                sess_options=sess_options,
                providers=providers,
            )

        active_providers = self.session.get_providers()
        logger.info(f"[ONNX] 推理后端: {active_providers}")
        if "DmlExecutionProvider" in active_providers:
            logger.info("[ONNX] ✅ 正在使用 DirectML GPU 加速")
        else:
            logger.info("[ONNX] ⚠️ 回退到 CPU 推理（未检测到 DirectML）")

        # 获取模型输入信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        logger.info(f"[ONNX] 模型输入: {self.input_name}, 形状: {self.input_shape}")

        # 预热（第一次推理较慢）
        dummy = np.zeros((1, 3, img_size, img_size), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
        logger.info("[ONNX] 预热完成")

    def detect(self, image: np.ndarray) -> tuple[sv.Detections, dict]:
        """
        输入: BGR 图像 (H, W, 3)
        输出: (supervision.Detections, speed_dict)
        """
        t_start = time.perf_counter()

        # 1. 预处理
        blob, scale, pad_x, pad_y = self._preprocess(image)
        t_pre = time.perf_counter()

        # 2. 推理
        outputs = self.session.run(None, {self.input_name: blob})
        t_inf = time.perf_counter()

        # 3. 后处理
        detections = self._postprocess(outputs[0], scale, pad_x, pad_y)
        t_post = time.perf_counter()

        # 4. 速度信息（毫秒）
        speed = {
            "preprocess": (t_pre - t_start) * 1000,
            "inference": (t_inf - t_pre) * 1000,
            "postprocess": (t_post - t_inf) * 1000,
        }
        return detections, speed

    def _preprocess(self, image: np.ndarray):
        """Letterbox resize + normalize + HWC→NCHW"""
        h, w = image.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 灰色填充 (114, 114, 114)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_y = (self.img_size - new_h) // 2
        pad_x = (self.img_size - new_w) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR → RGB, uint8 → float32, 归一化到 [0, 1]
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        # HWC → CHW → NCHW
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, scale, pad_x, pad_y

    def _postprocess(self, raw_output: np.ndarray, scale: float, pad_x: int, pad_y: int) -> sv.Detections:
        """
        解析 YOLOv8/v10 标准输出格式
        raw_output shape: (1, 4 + num_classes, num_anchors)
        """
        # raw_output: (1, 6, 8400) for 2-class model (player, head)
        output = raw_output[0]  # (6, 8400)
        output = output.T       # (8400, 6) → [cx, cy, w, h, conf_0, conf_1]

        num_classes = output.shape[1] - 4

        # 获取每个 box 的最大类别置信度和对应类别 ID
        class_scores = output[:, 4:]          # (N, num_classes)
        class_ids = class_scores.argmax(axis=1)  # (N,)
        max_confs = class_scores.max(axis=1)     # (N,)

        # 置信度过滤
        keep = max_confs >= self.conf_thresh
        if not keep.any():
            return sv.Detections.empty()

        output = output[keep]
        class_ids = class_ids[keep]
        max_confs = max_confs[keep]

        # cx, cy, w, h → x1, y1, x2, y2
        cx = output[:, 0]
        cy = output[:, 1]
        bw = output[:, 2]
        bh = output[:, 3]

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # 还原 letterbox 偏移和缩放
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(),
            max_confs.tolist(),
            self.conf_thresh,
            self.iou_thresh,
        )

        if len(indices) == 0:
            return sv.Detections.empty()

        indices = indices.flatten()

        return sv.Detections(
            xyxy=xyxy[indices],
            confidence=max_confs[indices].astype(np.float32),
            class_id=class_ids[indices].astype(np.int64),
        )


# ─────────────────────────── 主循环 ───────────────────────────

def init():
    run_checks()
    from logic.capture import capture
    from logic.visual import visuals
    from logic.frame_parser import frameParser
    from logic.hotkeys_watcher import hotkeys_watcher, shutdown_event
    from logic.shooting import shooting
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker() if not cfg.disable_tracker else None

    # 根据配置选择模型加载方式
    if cfg.ai_model_use_enc:
        # 闭源模式: 卡密验证 + 解密 .enc 模型
        from models.model_loader import load_model
        license_key = input("请输入卡(密微信小程序\"西瓜去水印文案配音助手\"免费获取)：").strip()
        model_source = load_model(license_key)
        logger.info("[AI] 从内存加载解密后的模型")
    else:
        # 调试模式: 直接加载 .onnx 明文模型
        model_source = f"models/{cfg.AI_model_name}"
        logger.info(f"[AI] 直接加载明文模型: {model_source}")

    detector = OnnxDetector(
        model_source=model_source,
        img_size=cfg.ai_model_image_size,
        conf_thresh=cfg.AI_conf,
    )

    while not shutdown_event.is_set():
        t0 = time.perf_counter()

        image = capture.get_new_frame()
        t1 = time.perf_counter()

        if image is None:
            continue

        if cfg.circle_capture:
            image = capture.convert_to_circle(image)

        if hotkeys_watcher.app_pause != 0:
            visuals.clear()
            shooting.shoot(False, False)
            if cfg.show_window or cfg.show_overlay:
                visuals.submit_frame(image)
            continue

        # ONNX 推理
        detections, speed = detector.detect(image)
        t2 = time.perf_counter()

        # 显示推理速度
        if cfg.show_window and cfg.show_detection_speed:
            from logic.visual import visuals
            visuals.draw_speed(
                speed["preprocess"],
                speed["inference"],
                speed["postprocess"],
            )

        # ByteTrack 追踪
        if tracker:
            detections = tracker.update(detections)
        t3 = time.perf_counter()

        # 解析目标并执行瞄准
        frameParser.parse(detections, image)
        t4 = time.perf_counter()

        if cfg.show_window or cfg.show_overlay:
            visuals.submit_frame(image)
        t5 = time.perf_counter()

        # 耗时调试输出
        if cfg.log_debug_timing:
            ms = lambda a, b: (b - a) * 1000
            print(
                f"[耗时] "
                f"等待帧到达:{ms(t0, t1):.1f}ms | "
                f"ONNX推理:{ms(t1, t2):.1f}ms | "
                f"ByteTrack:{ms(t2, t3):.1f}ms | "
                f"目标选择+瞄准+鼠标:{ms(t3, t4):.1f}ms | "
                f"可视化提交:{ms(t4, t5):.1f}ms | "
                f"总一轮耗时:{ms(t0, t5):.1f}ms"
            )


if __name__ == "__main__":
    import sys, os

    # 打包为 exe 时，确保工作目录为 exe 所在目录
    # （config.ini、models/ 等相对路径依赖工作目录）
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))

    try:
        init()
    except Exception:
        import traceback
        traceback.print_exc()
        input("\n[错误] 程序异常退出，按回车键关闭...")
