import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from src.core.alarm.alarm import Alarm
from src.core.detect_crosswalk_signal.detect_crosswalk_signal import DetectCrosswalkSignal
from src.core.gui.gui import Gui
from src.core.models.class_names import class_names
from src.core.models.yolov8 import Yolov8DetectionModel
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig

track_history = defaultdict(lambda: [])
alarmed_objects_time = defaultdict(lambda: 0)
object_whitelist = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]


class DetectObject:
    def __init__(self, config: TOMLConfig, model_path):
        self.do_env = config.env["detect_object"]
        self.model_path = model_path
        self.yolov8 = Yolov8DetectionModel(config, config.env["yolo"]["model"], self.do_env["confidence_threshold"])
        self.detection_times = {}
        self.last_alarm_object = None
        self.object_queue = []

    def __call__(self, color_image, depth_frame=None):
        prediction_list = self.yolov8(color_image, track_history)
        closest_object = None

        # 使用倒序迴圈避免在移除list裡面的item時，發生以下的錯誤：
        # The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        for i in range(len(prediction_list) - 1, -1, -1):
            class_id, box, score, track_id = prediction_list[i]
            class_name = self.yolov8.category[class_id]
            if class_name not in object_whitelist:
                prediction_list.pop(i)
                continue

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

            name = class_names[class_id]

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

                max_dist = TOMLConfig.instance.env["detect_object"]["max_distance_threshold"]
                if dist > max_dist:
                    continue

                if dist == -1:
                    continue  # 消失點不警報

                max_lateral_dist = TOMLConfig.instance.env["detect_object"]["lateral_distance_threshold"]
                if lateral_dist < -max_lateral_dist:
                    direction = "左側方"
                elif lateral_dist > max_lateral_dist:
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
        time_now = int(time.time() * 1000)

        if Alarm.instance.speaking_count > 0 or len(self.object_queue) == 0:
            return

        if DetectCrosswalkSignal.instance is not None and DetectCrosswalkSignal.instance.is_alarm:
            return

        # 找出最近的物體
        self.object_queue.sort(key=lambda x: x[3])
        name, track_id, direction, dist = self.object_queue[0]
        self.object_queue = []



        if dist > 1000:
            dist_str = f"{str(round(dist / 100) / 10).replace('.', '點')}公尺"  # 毫米轉公尺
        elif dist >= 100:
            dist_str = f"{round(dist / 100) * 10}公分"  # 毫米轉公分，去掉尾數 例如：1372毫米 → 130公分
        elif dist >= 10:
            dist_str = f"{round(dist / 10)}公分"  # 毫米轉公分
        else:
            dist_str = ""
        print(dist)

        if time_now - self.last_alarm_time > 1000:
            threading.Thread(target=self._speak, args=(f"{direction}{dist_str}有{name}{track_id}",)).start()
            self.last_alarm_time = time_now

    def _speak(self, message):
        if Gui.instance is not None:
            Gui.instance.statusbar.showMessage(message)

        Alarm.instance.speak(message)
        self.last_alarm_time = int(time.time() * 1000)

    def draw_detections(self, image, prediction_list, depth_image = None, mask_alpha: float = 0.4):
        # for track_id in track_history:
        #     for box in track_history[track_id]:
        #         x, y, w, h = box
        #         center = (int(x + w / 2), int(y + h / 2))
        #         yolov8_img = cv2.circle(yolov8_img, center, 2, (0, 0, 255), -1)

        return self.yolov8.draw_detections(image, prediction_list, depth_image, mask_alpha)
