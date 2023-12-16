import threading
import time
from enum import Enum

import cv2

from .utils import find_nearest
from .. import TOMLConfig
from ..alarm.alarm import Alarm
from ..detect_obstacle.detect_obstacle import DetectObstacle


class SignalStatus(Enum):
    NONE = 0
    RED = 1
    GREEN = 2


class DetectCrosswalkSignal:
    def __init__(self):
        # self.csd_env = TOMLConfig.instance.env["crosswalk_signal_detection"]
        self.signal_status = SignalStatus.NONE
        self.invalid_time = -1

    def __call__(self, image, prediction_list, names):
        """
        判斷圖片裡面最接近畫面中心的行人號誌
        :param image: 要辨識的圖片
        :param prediction_list: 預測結果
        :param names: 模型所有類別名稱
        :return image: 畫上行人號誌的圖片
        """
        found, nearest_object = find_nearest(image, prediction_list, names)

        if not found:
            self.invalid()
            return

        class_id, box, score = nearest_object
        class_name = names[class_id]

        if class_name == "red":
            signal_status = SignalStatus.RED
        elif class_name == "green":
            signal_status = SignalStatus.GREEN

        self.invalid_time = -1

        if self.signal_status != signal_status:
            self.signal_status = signal_status
            self.alert()

        return box

    def alert(self):
        if self.signal_status == SignalStatus.NONE:
            return

        def _():
            if DetectObstacle.instance:
                DetectObstacle.instance.pause_alarm()
                time.sleep(0.5)

            if self.signal_status == SignalStatus.RED:
                print("紅燈")
                # Alarm.instance.play_sound(1000, 2)
                Alarm.instance.play_notes(["C4", "E4", "G4"], 2)
                self.invalid_time = -1
            else:
                print("綠燈")
                Alarm.instance.play_sound(3000, 2)
                self.invalid_time = -1

            time.sleep(0.5)

            if DetectObstacle.instance:
                DetectObstacle.instance.resume_alarm()

        threading.Thread(target=_).start()

    def draw_line(self, image, box):
        image = image.copy()
        img_height, img_width = image.shape[:2]
        color = (0, 0, 255) if self.signal_status == SignalStatus.RED else (0, 255, 0)
        return cv2.line(
            image,
            (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)),
            (img_width // 2, img_height // 2),
            color,
            2,
        )

    def invalid(self):
        """無法辨識行人號誌時，重置狀態"""
        if self.signal_status == SignalStatus.NONE:
            return

        if self.invalid_time == -1:
            self.invalid_time = time.time() * 1000
        elif time.time() * 1000 - self.invalid_time > 5000:
            self.invalid_time = -1
            self.signal_status = SignalStatus.NONE

            if DetectObstacle.instance:
                DetectObstacle.instance.pause_alarm()
                time.sleep(0.5)

            Alarm.instance.play_sound(2000, 2)
            print("沒有行人號誌，重置狀態")

            time.sleep(0.5)

            if DetectObstacle.instance:
                DetectObstacle.instance.resume_alarm()

    def is_none(self):
        return self.signal_status == SignalStatus.NONE
