from typing import Any

import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from .. import TOMLConfig
from .base import DetectionModel


class Yolov8DetectionModel(DetectionModel):
    def load_env(self):
        self.yolo_env = TOMLConfig.instance.env["yolo"]
        if self.yolo_env["confidence_threshold"] is not None:
            self.confidence_threshold = self.yolo_env["confidence_threshold"]

    def load_model(self):
        """ "
        檢測模型已經初始化並成功設定為self.model
        （需要利用self.model_path、self.config_path和self.device）
        """
        self.set_model(YOLO(self.model_path))

    def set_model(self, model: Any, sahi_model: Any):
        """
        設置底層的YOLOv8模型
        :param model: A YOLOv8 model
        :param sahi_model: A SAHI model
        """

        self.model = YOLO(self.model_path)
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
            class_ids = result.class_ids
            boxes = result.xyxys
            scores = result.scores

        return len(class_ids) != 0, list(zip(class_ids, boxes, scores))

    def __call__(self, img: np.ndarray):
        """
        預測圖片中的物件
        :param img: 圖片
        """

        return self.model.predict(
            img, conf=self.confidence_threshold, agnostic_nms=True
        )
