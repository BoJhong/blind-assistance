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
    def _process_object_prediction(prediction_list: Any, track_history):
        """
        處理物件預測結果
        :param prediction_list: 物件預測結果
        """
        class_ids = []
        boxes = []
        scores = []
        track_ids = []

        for result in prediction_list:
            detection_count = result.boxes.shape[0]
            for i in range(detection_count):
                class_id = int(result.boxes.cls[i].item())
                box = result.boxes.xyxy[i].cpu().numpy()
                score = float(result.boxes.conf[i].item())
                track_id = -1
                if result.boxes.id is not None:
                    track_id = result.boxes.id[i].int().cpu().item()
                    track = track_history[track_id]

                    track.append(box)  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                class_ids.append(class_id)
                boxes.append(box)
                scores.append(score)
                track_ids.append(track_id)

        return len(class_ids) != 0, list(zip(class_ids, boxes, scores, track_ids))

    def __call__(self, img: np.ndarray, track_history):
        """
        預測圖片中的物件（track函數必須傳入 persist=True ，否則畫面都是單獨運算）
        :param img: 圖片
        """

        return self._process_object_prediction(
            self.model.track(
                img,
                conf=self.confidence_threshold,
                iou=0.5,
                agnostic_nms=True,
                verbose=False,
                persist=True,
                tracker='botsort.yaml'
            ),
            track_history
        )
