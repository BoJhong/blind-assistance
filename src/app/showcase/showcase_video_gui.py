import os
import sys
import threading
import time
from threading import Thread

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from src.core.alarm.alarm import Alarm
from src.core.detect_crosswalk_signal.detect_crosswalk_signal import (
    DetectCrosswalkSignal,
)
from src.core.detect_object.detect_object import DetectObject
from src.core.detect_obstacle.detect_obstacle import DetectObstacle
from src.core.gui.gui import Gui
from src.core.models.yolov8sahi import Yolov8SahiDetectionModel
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig
from src.core.vision.vision import Vision
from src.utils.detect_blur import draw_blur_status

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))
alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)
detect_object = DetectObject(config, config.env["yolo"]["model"])

vision = None
yolov8_sahi = None

def init_thread():
    global vision, yolov8_sahi
    yolov8_sahi = Yolov8SahiDetectionModel(config, config.env["yolo"]["cs_model"])
    vision = Vision(config)


threading.Thread(target=init_thread).start()

app = QApplication(sys.argv)

dcs_img = None
last_process_frame = 0
blurry = False
slow_thread = None
finished = True

class SlowProcessThread(Thread):
    def __init__(self, image, n):
        super().__init__()
        self.is_running = False
        self.image = image
        self.n = n

    def run(self):
        self.is_running = True
        while self.is_running:
            global dcs_img
            image = Gui.instance.color_image.copy()
            sahi_prediction_list = yolov8_sahi(image)

            if len(sahi_prediction_list) > 0:
                nearst_box = detect_cs(image, sahi_prediction_list, yolov8_sahi.category)
                if nearst_box is not None:
                    image = detect_cs.draw_line(image, nearst_box)
                image = yolov8_sahi.draw_detections(image, sahi_prediction_list)
            else:
                detect_cs.invalid()

            dcs_img = image

    def stop(self):
        self.is_running = False


count = 0
current_frame_index = 0

def update_frame(cap):
    global count, current_frame_index, dcs_img, slow_thread
    start_time = time.time()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps

    if Gui.instance.is_streaming:
        ret, frame = cap.read()
        if not ret:
            current_frame_index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        Gui.instance.color_image = frame.copy()

    prediction_list = detect_object(frame)
    detect_object_img = frame.copy()

    if len(prediction_list) > 0:
        detect_object_img = detect_object.draw_detections(
            detect_object_img, prediction_list
        )

    if yolov8_sahi and slow_thread is None:
        slow_thread = SlowProcessThread(frame, count)
        slow_thread.start()

    if dcs_img is None:
        dcs_img = frame.copy()

    # main_window.display_image(combined_img, 0)
    # main_window.display_image(combined_depth_colormap, 1)
    Gui.instance.display_image(detect_object_img, 0)
    Gui.instance.display_image(dcs_img, 1)
    # main_window.display_image(heatmap, 3)

    if Gui.instance.is_streaming:
        elapsed_time = time.time() - start_time
        current_frame_index += int(elapsed_time // frame_time) + 1
        print(f"FPS: {current_frame_index}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)


gui = Gui(config, update_frame, is_video_capture=True)
gui.setWindowTitle("盲人輔助系統 Blind Assistance")
gui.show()
sys.exit(app.exec_())
