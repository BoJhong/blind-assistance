from typing import Any

import numpy as np
from ultralytics import YOLO

from .base import DetectionModel


class Yolov8DetectionModel(DetectionModel):
    def load_env(self, config: Any):
        self.yolo_env = config.env["yolo"]
        if self.yolo_env["confidence_threshold"] is not None:
            self.confidence_threshold = self.yolo_env["confidence_threshold"]

    def load_model(self):
        """ "
        檢測模型已經初始化並成功設定為 self.model
        （需要利用self.model_path、self.config_path和self.device）
        """
        self.set_model(YOLO(self.model_path))

    def set_model(self, model: Any):
        """
        設置底層的YOLOv8模型
        :param model: A YOLOv8 model
        """

        self.model = YOLO(self.model_path)
        self.category = self.model.names

    @staticmethod
    def _process_object_prediction(prediction_list: Any):
        """
        處理物件預測結果
        :param prediction_list: 物件預測結果
        """
        class_ids = []
        boxes = []
        scores = []

        for result in prediction_list:
            detection_count = result.boxes.shape[0]
            for i in range(detection_count):
                class_id = int(result.boxes.cls[i].item())
                box = result.boxes.xyxy[i].cpu().numpy()
                score = float(result.boxes.conf[i].item())

                class_ids.append(class_id)
                boxes.append(box)
                scores.append(score)

        return len(class_ids) != 0, list(zip(class_ids, boxes, scores))

    def __call__(self, img: np.ndarray):
        """
        預測圖片中的物件
        :param img: 圖片
        """

        return self._process_object_prediction(
            self.model.predict(
                img, conf=self.confidence_threshold, agnostic_nms=True, verbose=False
            )
        )
