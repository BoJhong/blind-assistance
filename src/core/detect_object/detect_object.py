import threading
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from src.core.alarm.alarm import Alarm
from src.core.models.yolov8 import Yolov8DetectionModel
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig

track_history = defaultdict(lambda: [])
alarmed_objects_time = defaultdict(lambda: 0)


class DetectObject:
    def __init__(self, config: Any, model_path):
        self.model_path = model_path
        self.yolov8 = Yolov8DetectionModel(config, config.env["yolo"]["model"])
        self.detection_times = {}
        self.last_alarm_time = 0
        self.object_queue = []
        self.speaking = False

        self.class_names = {
            "person": "行人",
            "bicycle": "自行車",
            "car": "汽車",
            "motorcycle": "機車",
            "bus": "公車",
            "truck": "卡車",
        }

    def __call__(self, color_image, depth_frame=None):
        prediction_list = self.yolov8(color_image, track_history)
        closest_object = None

        for class_id, box, score, track_id in prediction_list:
            class_name = self.yolov8.category[class_id]

            track = track_history[track_id]

            '''
            物體警報
            '''

            if len(track) <= 10:
                continue

            if track_id not in alarmed_objects_time:
                alarmed_objects_time[track_id] = int(datetime.now().timestamp() * 1000)
            else:
                date_now = int(datetime.now().timestamp() * 1000)
                alarmed_time = alarmed_objects_time[track_id]
                if date_now - alarmed_time < 5000:
                    continue
                else:
                    alarmed_objects_time[track_id] = date_now

            name = self.class_names[class_name] if class_name in self.class_names else class_name

            object_center = (int((box[2] - box[0]) / 2 + box[0]), int((box[3] - box[1]) / 2 + box[1]))

            if depth_frame is not None:
                depth_pixel = RealsenseCamera.instance.project_color_pixel_to_depth_pixel(
                    depth_frame.get_data(),
                    object_center)
                if not depth_pixel:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())

                result = RealsenseCamera.instance.depth_pixel_to_height(
                    depth_image, depth_pixel, TOMLConfig.instance.env["obstacle_detection"]["camera_height"]
                )

                if not result:
                    continue

                height, dist, lateral_dist, depth_point = result

                if dist > 1500:
                    continue

                if dist == -1:
                    continue  # 消失點不警報

                if lateral_dist < -150:
                    direction = "左側方"
                elif lateral_dist > 150:
                    direction = "右側方"
                else:
                    direction = "前方"

            track_history[track_id] = []

            if closest_object is None or dist < closest_object[3]:
                closest_object = (name, track_id, direction, dist)

        if closest_object is not None:
            self.object_queue.append(closest_object)

        self._alert()

        return prediction_list

    def _alert(self):
        time_now = int(datetime.now().timestamp() * 1000)

        if self.speaking:
            self.last_alarm_time = time_now
            return

        if len(self.object_queue) == 0 or time_now - self.last_alarm_time < 1000:
            return

        # 找出最近的物體
        self.object_queue.sort(key=lambda x: x[3])
        name, track_id, direction, dist = self.object_queue[0]
        self.object_queue = []

        if dist > 1000:
            dist_str = f"{str(round(dist / 100) / 10).replace('.', '點')}公尺"
        elif dist >= 100:
            dist_str = f"{round(dist / 100) * 100}公分"
        else:
            dist_str = f"{round(dist / 10) * 10}公分"

        self.last_alarm_time = time_now
        threading.Thread(target=self._speak, args=(f"{direction}{dist_str}有{name}{track_id}",)).start()

    def _speak(self, message):
        self.speaking = True
        Alarm.instance.speak(message)
        self.speaking = False

    def draw_detections(self, image, prediction_list, depth_image, mask_alpha: float = 0.4):
        # for track_id in track_history:
        #     for box in track_history[track_id]:
        #         x, y, w, h = box
        #         center = (int(x + w / 2), int(y + h / 2))
        #         yolov8_img = cv2.circle(yolov8_img, center, 2, (0, 0, 255), -1)

        return self.yolov8.draw_detections(image, prediction_list, depth_image, mask_alpha)
