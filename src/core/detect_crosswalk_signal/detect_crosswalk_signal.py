import threading
import time
from enum import Enum
from typing import Any

import cv2
from PIL import Image

from .utils import find_nearest
from ..alarm.alarm import Alarm
from ..detect_obstacle.detect_obstacle import DetectObstacle
from ..toml_config import TOMLConfig
from ..vision.vision import Vision


class SignalStatus(Enum):
    NONE = 0
    RED = 1
    GREEN = 2


class DetectCrosswalkSignal:
    instance = None

    def __init__(self, config: Any):
        DetectCrosswalkSignal.instance = self

        self.dcs_env = config.env["detect_crosswalk_signal"]
        self.signal_status = SignalStatus.NONE
        self.invalid_time = -1
        self.is_alarm = False

    def __call__(self, image, prediction_list, names):
        """
        判斷圖片裡面最接近畫面中心的行人號誌
        :param image: 要辨識的圖片
        :param prediction_list: 預測結果
        :param names: 模型所有類別名稱
        :return image: 畫上行人號誌的圖片
        """
        if self.is_alarm:
            return

        # 找出最接近畫面中心和最近的行人號誌
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

    def vision_countdown(self, image):
        start_time = time.time()  # 紀錄開始計算的時間
        prompt = "Please tell me the countdown seconds of the pedestrian signal closest to the center of the screen (only answer with a number)."
        response = Vision.instance.predict(Image.fromarray(image), prompt).strip()
        end_time = time.time()
        second = int(response) if response.isdigit() else 0
        calc_time = round(end_time - start_time)  # 計算時間
        countdown = second - calc_time  # 扣除計算時間後的倒數秒數
        print(f"預測 {second} 秒")
        print(f"計算 {calc_time} 秒")
        print(f"倒數 {countdown} 秒")

    def alert(self):
        """
        播放警示音效
        """
        if self.signal_status == SignalStatus.NONE:
            return

        if TOMLConfig.instance.env["alarm"]["tts_enable"]:
            if self.signal_status == SignalStatus.RED:
                message = "注意前方紅燈"
            else:
                message = "注意前方綠燈"
            threading.Thread(target=self._speak, args=(message,)).start()
            return

        self.is_alarm = True

        if self.signal_status == SignalStatus.RED:
            print("紅燈")
            notes = ["G4", "E4", "D4", "C4"]
        else:
            print("綠燈")
            notes = ["C4", "D4", "E4", "G4"]

        threading.Thread(target=self.play_notes, args=notes).start()

    def draw_line(self, image, box):
        """
        在偵測到的行人號誌與畫面中心點繪製一條線
        :param image: 圖片
        :param box: 行人號誌的框
        :return image: 畫上行人號誌的線的圖片
        """
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
        """
        無法辨識行人號誌時，重置狀態
        """
        if self.signal_status == SignalStatus.NONE:
            return

        time_now = time.time() * 1000
        invalid_time = self.dcs_env["invalid_time"] * 1000

        if self.invalid_time == -1:
            self.invalid_time = time_now
        elif time_now - self.invalid_time > invalid_time:
            self.invalid_time = -1
            self.signal_status = SignalStatus.NONE

            if TOMLConfig.instance.env["alarm"]["tts_enable"]:
                threading.Thread(target=self._speak, args=("行人號誌已離開視線",)).start()

            else:
                print("行人號誌已離開視線")
                notes = ["E4", "D4"]
                threading.Thread(target=self.play_notes, args=notes).start()

    def is_none(self):
        return self.signal_status == SignalStatus.NONE

    def _speak(self, message):
        self.is_alarm = True
        Alarm.instance.speak(message)
        self.is_alarm = False

    def play_notes(self, *args):
        if DetectObstacle.instance:
            DetectObstacle.instance.pause_alarm()
            time.sleep(0.5)

        notes = [note for note in args]

        Alarm.instance.play_notes(notes)
        self.invalid_time = -1

        if DetectObstacle.instance:
            time.sleep(0.5)
            DetectObstacle.instance.resume_alarm()

        self.is_alarm = False
