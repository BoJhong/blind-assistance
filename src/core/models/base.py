from typing import Optional, Dict, Any

import numpy as np

from .utils import draw_masks, draw_box, draw_text
from ..realsense_camera.utils import get_middle_dist
from ..toml_config import TOMLConfig


class DetectionModel:
    instance = None

    def __init__(
            self,
            config: Any,
            model_path: str,
            confidence_threshold: float = 0.3,
            category_mapping: Any = None,
            load_at_init: bool = True,
    ):
        """
        初始化檢測模型
        :param model_path: 模型檔案路徑
        """
        self.instance = self
        self.load_env(config)
        if load_at_init:
            self.model_path = model_path
            self.confidence_threshold = confidence_threshold
            self.load_model()
            rng = np.random.default_rng(1)
            self.colors = rng.uniform(0, 255, size=(len(self.category), 3))

    def load_env(self, env: Any):
        pass

    def load_model(self):
        """
        載入檢測模型
        （需要用到self.model_path、self.config_path和self.device）
        """
        raise NotImplementedError()

    def set_model(self, model: Any, **kwargs):
        """
        設置檢測模型
        :param model: 檢測模型
        :param kwargs: 額外的參數
        """
        raise NotImplementedError()

    def __call__(self, img: np.ndarray):
        pass

    def draw_detections(
            self,
            image: np.ndarray,
            prediction_list: any,
            depth_data: np.ndarray = None,
            mask_alpha: float = 0.4,
    ):
        """
        繪製預測結果
        :param image: 原始圖片
        :param prediction_list: 預測結果
        :param depth_data: 深度資料
        :param mask_alpha: 遮罩透明度
        """
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        det_img = draw_masks(det_img, prediction_list, self.colors, mask_alpha)

        # Draw bounding boxes and labels of detections
        for p in prediction_list:
            class_id = p[0]
            box = p[1]
            score = p[2]
            track_id = p[3] if len(p) == 4 else None

            color = self.colors[class_id]
            draw_box(det_img, box, color)

            # 如果debug模式為關閉狀態，則不顯示標籤，只顯示框線
            if not TOMLConfig.instance.env["config"]["debug"]:
                continue

            label = self.category[class_id]
            if track_id is not None:
                caption = f"ID: {track_id} {label} {int(score * 100)}%"
            else:
                caption = f"{label} {int(score * 100)}%"

            # 如果有深度數據，則繪製距離
            if depth_data is not None:
                dist = get_middle_dist(det_img, box, depth_data, 24)

                if dist != -1:
                    dist = str(dist / 1000)[:4]
                    caption += f" ({dist}m)"

            draw_text(det_img, caption, box, color, font_size, text_thickness)

        return det_img

    def print_detections(
            self,
            image: np.ndarray,
            prediction_list: any,
            depth_image: np.ndarray = None
    ):
        """
        繪製預測結果
        :param image: 原始圖片
        :param prediction_list: 預測結果
        :param depth_image: 深度資料
        """
        for prediction in prediction_list:
            class_id, box, score = prediction
            label = self.category[class_id]
            print(
                f"Class Name: {label}, Box: {box}, Score: {score}, Depth: {get_middle_dist(image, box, depth_image, 3)}")
