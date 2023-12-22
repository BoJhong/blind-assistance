from typing import Any

import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from .base import DetectionModel


class Yolov8SahiDetectionModel(DetectionModel):
    def load_env(self, config: Any):
        self.yolo_env = config.env["yolo"]
        self.sahi_env = config.env["sahi"]

    def load_model(self):
        self.set_model(
            YOLO(self.model_path),
            AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=self.model_path,
                confidence_threshold=self.yolo_env["confidence_threshold"],
            ),
        )
        """ "
        檢測模型已經初始化並成功設定為 self.model
        （需要利用self.model_path、self.config_path和self.device）
        """

    def set_model(self, model: Any, sahi_model: Any):
        """
        設置底層的YOLOv8模型
        :param model: A YOLOv8 model
        :param sahi_model: A SAHI model
        """

        self.model = model
        self.sahi_model = sahi_model
        self.category = self.model.names

    def _process_object_prediction(self, prediction_list: Any):
        """
        處理物件預測結果
        :param prediction_list: 物件預測結果
        """
        class_ids = []
        boxes = []
        scores = []

        for result in prediction_list:
            try:
                data = result.get_shifted_object_prediction()
                class_id, box, score = (
                    data.category.id,
                    np.array(data.bbox.to_xyxy(), dtype=np.float32),
                    data.score.value,
                )
                if score < self.confidence_threshold:
                    continue
            except Exception:
                continue

            class_ids.append(class_id)
            boxes.append(box)
            scores.append(score)

        return len(class_ids) != 0, list(zip(class_ids, boxes, scores))

    def __call__(
        self,
        img: np.ndarray,
        slice_height: int = None,
        slice_width: int = None,
        overlap_height_ratio: float = None,
        overlap_width_ratio: float = None,
    ):
        """
        預測圖片中的物件
        :param img: 圖片
        :param slice_height: 切片高度
        :param slice_width: 切片寬度
        :param overlap_height_ratio: 高度重疊比例
        :param overlap_width_ratio: 寬度重疊比例
        """
        slice_height = slice_height or self.sahi_env.get("slice_height")
        slice_width = slice_width or self.sahi_env.get("slice_width")
        overlap_height_ratio = overlap_height_ratio or self.sahi_env.get(
            "overlap_height_ratio"
        )
        overlap_width_ratio = overlap_width_ratio or self.sahi_env.get(
            "overlap_width_ratio"
        )

        return self._process_object_prediction(
            get_sliced_prediction(
                img,
                self.sahi_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                verbose=False,
            ).object_prediction_list
        )
