import os
from threading import Thread

import numpy as np

from src.core import (
    TOMLConfig,
    Yolov8SahiDetectionModel,
    RealsenseCamera,
    DetectObstacle,
    Alarm,
    DetectCrosswalkSignal,
)
from src.utils.detect_blur import detect_blur_fft

setting = TOMLConfig(os.path.join(__file__, "../config.toml"))
rs_camera = RealsenseCamera()
yolov8_sahi = Yolov8SahiDetectionModel(setting.env["yolo"]["cs_model"])
detect_obstacle = DetectObstacle()
alarm = Alarm()
detect_cs = DetectCrosswalkSignal()

last_process_frame = 0
blurry = False
slow_thread = None
finished = True
frame_number = 0


def slow_processing(image, n):
    global blurry, last_process_frame, slow_thread, finished
    if not finished:
        return

    mean = None
    if detect_cs.is_none():
        if n % 15 != 0 and not blurry:
            return

        (mean, blurry) = detect_blur_fft(image, size=60, thresh=10)
        if blurry and n // 15 - last_process_frame < 5:
            return

    if slow_thread:
        slow_thread.join()

    last_process_frame = n // 15
    finished = False

    def _():
        global finished

        object_exists, prediction_list = yolov8_sahi(image)
        finished = True

        if object_exists:
            detect_cs(image, prediction_list, yolov8_sahi.category)
        else:
            detect_cs.invalid()

    slow_thread = Thread(target=_)
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

        detect_obstacle(depth_frame, color_image)

        frame_number = frames.get_frame_number()
        slow_processing(color_image, frame_number)
finally:
    rs_camera.pipeline.stop()
    alarm.cleanup()
