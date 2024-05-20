import os
from threading import Thread

import numpy as np

from src.core.alarm.alarm import Alarm
from src.core.detect_crosswalk_signal.detect_crosswalk_signal import (
    DetectCrosswalkSignal,
)
from src.core.detect_obstacle.detect_obstacle import DetectObstacle
from src.core.models.yolov8 import Yolov8DetectionModel
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig
from src.utils.detect_blur import detect_blur_fft

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))

rs_camera = RealsenseCamera(config)
yolov8 = Yolov8DetectionModel(config, config.env["yolo"]["cs_model"])
detect_obstacle = DetectObstacle(config)
alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)

last_process_frame = 0
blurry = False
slow_thread = None
finished = True
frame_number = 0


def slow_processing(c_image, d_image, n):
    global blurry, last_process_frame, slow_thread, finished
    if not finished:
        return

    mean = None
    if detect_cs.is_none():
        if n % 15 != 0 and not blurry:
            return

        (mean, blurry) = detect_blur_fft(c_image, size=60, thresh=10)
        if blurry and n // 15 - last_process_frame < 5:
            return

    if slow_thread:
        slow_thread.join()

    last_process_frame = n // 15
    finished = False

    def fn():
        global finished

        prediction_list = yolov8(c_image)
        finished = True

        if len(prediction_list) > 0:
            yolov8.print_detections(c_image, prediction_list, d_image)
            detect_cs(c_image, prediction_list, yolov8.category)
        else:
            detect_cs.invalid()

    slow_thread = Thread(target=fn)
    slow_thread.start()


try:
    while 1:
        frames = rs_camera.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        motion = rs_camera.combined_angle(frames)
        if not motion:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        bottom_point = rs_camera.auto_camera_height(depth_frame)

        detect_obstacle(depth_frame, color_image)

        frame_number = frames.get_frame_number()
        slow_processing(color_image, depth_image, frame_number)
        print(frame_number)
finally:
    rs_camera.pipeline.stop()
    alarm.cleanup()
