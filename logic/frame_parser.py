import torch
import cv2
import threading
import supervision as sv
import numpy as np

from logic.hotkeys_watcher import hotkeys_watcher
from logic.config_watcher import cfg
from logic.capture import capture
from logic.visual import visuals
from logic.mouse import mouse
from logic.shooting import shooting
from logic.model_classes import HEAD_CLASS_ID, is_head_class

class Target:
    def __init__(self, x, y, w, h, cls):
        self.cls = int(cls)
        self.x = x + cfg.body_x_offset * w if not is_head_class(self.cls) else x
        self.y = y if is_head_class(self.cls) else (y - cfg.body_y_offset * h)
        self.w = w
        self.h = h

class FrameParser:
    def __init__(self):
        self._lock = threading.Lock()
        self.arch = self.get_arch()
        self._center = None
        self._center_cache_key = None
        self._locked_tracker_id = None  # ByteTrack 目标锁定 ID
        self._current_frame = None      # 当前帧图像（用于颜色质心计算）

    def parse(self, result, image=None):
        with self._lock:
            self._current_frame = image
            if result is None:
                self._handle_no_detections()
                return

            if isinstance(result, sv.Detections):
                self._process_sv_detections(result)
            else:
                self._process_yolo_detections(result)

    def _process_sv_detections(self, detections):
        # 显示推理速度（ONNX 模式会附加 speed 属性）
        if (cfg.show_window and cfg.show_detection_speed) and hasattr(detections, 'speed'):
            speed = detections.speed
            if isinstance(speed, dict):
                visuals.draw_speed(
                    speed.get('preprocess', 0),
                    speed.get('inference', 0),
                    speed.get('postprocess', 0),
                )

        if len(detections) > 0:
            if cfg.show_window or cfg.show_overlay:
                visuals.draw_helpers(detections)
            target = self.sort_targets(detections)
            self._handle_target(target)
        else:
            self._handle_no_detections()

    def _process_yolo_detections(self, results):
        frames = (results,) if hasattr(results, "boxes") else results
        processed = False

        for frame in frames:
            processed = True
            boxes = getattr(frame, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                target = self.sort_targets(frame)
                self._handle_target(target)
                self._visualize_frame(frame)
            else:
                self._handle_no_detections()

        if not processed:
            self._handle_no_detections()

    def _handle_target(self, target):
        if target:
            if hotkeys_watcher.clss is None:
                hotkeys_watcher.active_classes()

            if target.cls in hotkeys_watcher.clss:
                # 像素颜色质心修正：用框内像素的加权质心替代框体几何中心
                if (cfg.aim_color_centroid
                        and self._current_frame is not None
                        and not is_head_class(target.cls)):
                    centroid_x = self._compute_color_centroid_x(target)
                    if centroid_x is not None:
                        target.x = centroid_x

                mouse.process_data((target.x, target.y, target.w, target.h, target.cls))

    def _compute_color_centroid_x(self, target):
        """
        像素颜色质心法：用框内像素与背景色的颜色距离作为权重，
        计算水平加权质心，替代框体几何中心。
        角色身体占框内面积 ~70%，武器 ~30%，质心自然偏向身体中线。
        返回修正后的 X 坐标，失败时返回 None。
        """
        frame = self._current_frame
        h_img, w_img = frame.shape[:2]

        # 框体边界（xywh → xyxy）
        x1 = int(max(0, target.x - target.w / 2))
        y1 = int(max(0, target.y - target.h / 2))
        x2 = int(min(w_img, target.x + target.w / 2))
        y2 = int(min(h_img, target.y + target.h / 2))

        if x2 <= x1 or y2 <= y1:
            return None

        # 提取 ROI
        roi = frame[y1:y2, x1:x2]

        # 转为 float 并计算每像素与 YOLO letterbox 灰色的颜色距离
        roi_f = roi.astype(np.float32)
        bg = np.float32([114.0, 114.0, 114.0])
        diff = np.sqrt(np.sum((roi_f - bg) ** 2, axis=2))  # shape: (H_roi, W_roi)

        # 每列权重之和 = 该列包含多少"角色像素"
        col_weights = diff.sum(axis=0)  # shape: (W_roi,)

        # 过滤低权重列（背景/武器等低对比区域）
        threshold = col_weights.max() * 0.2 if col_weights.max() > 0 else 0
        if threshold == 0:
            return None
        col_weights = np.where(col_weights > threshold, col_weights, 0)

        total_weight = col_weights.sum()
        if total_weight < 1.0:
            return None

        # 加权质心
        col_positions = np.arange(len(col_weights), dtype=np.float32)
        centroid_local = np.sum(col_positions * col_weights) / total_weight
        centroid_x = centroid_local + x1

        return float(centroid_x)

    def _visualize_frame(self, frame):
        if cfg.show_window or cfg.show_overlay:
            if cfg.show_boxes or cfg.overlay_show_boxes:
                visuals.draw_helpers(frame.boxes)

            if cfg.show_window and cfg.show_detection_speed:
                visuals.draw_speed(frame.speed['preprocess'], frame.speed['inference'], frame.speed['postprocess'])

    def _handle_no_detections(self):
        self._locked_tracker_id = None
        mouse.reset_aim_state()
        mouse.process_recoil_only()
        if cfg.show_window or cfg.show_overlay:
            visuals.clear()
        if cfg.auto_shoot or cfg.triggerbot:
            shooting.shoot(False, False)

    def sort_targets(self, frame):
        if isinstance(frame, sv.Detections):
            boxes_array, classes_tensor, tracker_ids = self._convert_sv_to_tensor(frame)
        else:
            if frame.boxes is None or len(frame.boxes) == 0:
                return None
            boxes_array = self._to_tensor(frame.boxes.xywh, dtype=torch.float32)
            classes_tensor = self._to_tensor(frame.boxes.cls, dtype=torch.long)
            tracker_ids = None

        if not classes_tensor.numel():
            return None

        return self._find_nearest_target(boxes_array, classes_tensor, tracker_ids)

    def _convert_sv_to_tensor(self, frame):
        xyxy = np.asarray(frame.xyxy, dtype=np.float32)
        xywh = np.empty((xyxy.shape[0], 4), dtype=np.float32)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

        boxes_tensor = torch.as_tensor(xywh, dtype=torch.float32, device=self.arch)
        class_ids = frame.class_id if frame.class_id is not None else np.zeros(xyxy.shape[0], dtype=np.int64)
        classes_tensor = torch.as_tensor(class_ids, dtype=torch.long, device=self.arch)

        tracker_ids = getattr(frame, 'tracker_id', None)

        return boxes_tensor, classes_tensor, tracker_ids

    def _to_tensor(self, value, dtype):
        if hasattr(value, "to"):
            return value.to(device=self.arch, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.arch)

    def _find_nearest_target(self, boxes_array, classes_tensor, tracker_ids=None):
        center = self._get_center(boxes_array.device)
        distances_sq = torch.sum((boxes_array[:, :2] - center) ** 2, dim=1)
        candidate_idxs = self._get_active_candidate_idxs(classes_tensor)

        if candidate_idxs.numel() == 0:
            self._locked_tracker_id = None
            return None

        # ── 目标锁定: 优先跟踪已锁定的目标 ──
        if cfg.mouse_lock_target and tracker_ids is not None and self._locked_tracker_id is not None:
            for idx in candidate_idxs:
                if int(tracker_ids[idx]) == self._locked_tracker_id:
                    nearest_idx = idx.item()
                    target_data = boxes_array[nearest_idx, :4].cpu().numpy()
                    target_class = int(classes_tensor[nearest_idx].item())
                    return Target(*target_data, target_class)
            # 锁定目标已从画面中消失，清除锁定
            self._locked_tracker_id = None

        # ── 正常选择最近目标 ──
        head_candidate_mask = classes_tensor[candidate_idxs] == HEAD_CLASS_ID

        if cfg.disable_headshot:
            non_head_candidate_idxs = candidate_idxs[~head_candidate_mask]

            if non_head_candidate_idxs.numel() == 0:
                return None

            size_factor = (boxes_array[:, 2] * boxes_array[:, 3]).clamp_min(1.0)
            distances_sq = distances_sq / size_factor
            candidate_idxs = non_head_candidate_idxs
            nearest_idx = candidate_idxs[torch.argmin(distances_sq[candidate_idxs])].item()
        else:
            if head_candidate_mask.any():
                candidate_idxs = candidate_idxs[head_candidate_mask]
            nearest_idx = candidate_idxs[torch.argmin(distances_sq[candidate_idxs])].item()

        # 记录新锁定的目标
        if tracker_ids is not None:
            self._locked_tracker_id = int(tracker_ids[nearest_idx])

        target_data = boxes_array[nearest_idx, :4].cpu().numpy()
        target_class = int(classes_tensor[nearest_idx].item())

        return Target(*target_data, target_class)

    def _get_active_candidate_idxs(self, classes_tensor):
        if hotkeys_watcher.clss is None:
            hotkeys_watcher.active_classes()

        active_class_ids = hotkeys_watcher.clss or []
        if not active_class_ids:
            return torch.empty(0, dtype=torch.long, device=classes_tensor.device)

        active_mask = torch.zeros_like(classes_tensor, dtype=torch.bool)
        for class_id in active_class_ids:
            active_mask |= classes_tensor == int(class_id)

        return torch.nonzero(active_mask, as_tuple=False).flatten()

    def _get_center(self, device):
        key = (capture.screen_x_center, capture.screen_y_center, str(device))
        if self._center is None or self._center_cache_key != key:
            self._center = torch.tensor(
                [capture.screen_x_center, capture.screen_y_center],
                dtype=torch.float32,
                device=device
            )
            self._center_cache_key = key
        return self._center

    def update_settings(self):
        with self._lock:
            self.arch = self.get_arch()
            self._center = None
            self._center_cache_key = None
            self._locked_tracker_id = None

    def get_arch(self):
        if cfg.AI_enable_AMD:
            return f'hip:{cfg.AI_device}'
        ai_device = str(cfg.AI_device).lower()
        if 'cpu' in ai_device:
            return 'cpu'
        if ai_device == 'cuda':
            return 'cuda'
        if ai_device.startswith('cuda:'):
            return cfg.AI_device
        else:
            return f'cuda:{cfg.AI_device}'

frameParser = FrameParser()
