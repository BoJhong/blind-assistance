import os
from threading import Thread

import cv2
import imutils
import numpy as np

from src.core import (
    TOMLConfig,
    Yolov8SahiDetectionModel,
    RealsenseCamera,
    DetectObstacle,
    Alarm,
    DetectCrosswalkSignal,
)
from src.utils.detect_blur import detect_blur_fft, draw_blur_status

setting = TOMLConfig(os.path.join(__file__, "../config.toml"))
rs_camera = RealsenseCamera()
yolov8_sahi = Yolov8SahiDetectionModel(setting.env["yolo"]["cs_model"])
detect_obstacle = DetectObstacle()
alarm = Alarm()
detect_cs = DetectCrosswalkSignal()

window_name = "Showcase"
csd_window_name = "Crosswalk Signal Detection"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(csd_window_name, cv2.WINDOW_AUTOSIZE)

csd_img = None
last_process_frame = 0
blurry = False
slow_thread = None
finished = True
frame_number = 0


def thread_func(image, n, mean=None):
    global csd_img, blurry, frame_number, finished

    img = draw_blur_status(image, mean, blurry) if mean is not None else image.copy()

    object_exists, prediction_list = yolov8_sahi(image)
    finished = True

    if object_exists:
        nearst_box = detect_cs(image, prediction_list, yolov8_sahi.category)

        if nearst_box is not None:
            img = detect_cs.draw_line(img, nearst_box)
        img = yolov8_sahi.draw_detections(img, prediction_list, depth_data=depth_image)
    else:
        detect_cs.invalid()

    csd_img = img


def slow_processing(image, n):
    global blurry, last_process_frame, slow_thread, finished
    mean = None

    if detect_cs.is_none():
        if n % 15 != 0 and not blurry:
            return
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (mean, blurry) = detect_blur_fft(gray, size=60, thresh=10)

        if blurry and n // 15 - last_process_frame < 5:
            return

    if not finished:
        return

    if slow_thread:
        slow_thread.join()

    last_process_frame = n // 15
    finished = False
    slow_thread = Thread(target=thread_func, args=(image, n, mean))
    slow_thread.start()


try:
    while (
        cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        and cv2.getWindowProperty(csd_window_name, cv2.WND_PROP_VISIBLE) >= 1
    ):
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
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        combined_img, combined_depth_colormap = detect_obstacle(
            depth_frame, color_image, depth_colormap
        )

        frame_number = frames.get_frame_number()
        slow_processing(color_image, frame_number)

        if csd_img is None:
            csd_img = color_image.copy()
        cv2.imshow(csd_window_name, imutils.resize(csd_img, height=480))

        combined_depth_colormap = rs_camera.draw_motion(combined_depth_colormap)
        images = np.hstack(
            (imutils.resize(combined_img, height=480), combined_depth_colormap)
        )

        cv2.imshow(window_name, images)
        key = cv2.waitKey(1)

        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord("h"):
            rs_camera.auto_camera_height(depth_frame)
finally:
    rs_camera.pipeline.stop()
    alarm.cleanup()
    cv2.destroyAllWindows()
