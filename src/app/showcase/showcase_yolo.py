import os
from threading import Thread

import cv2
import imutils
import numpy as np

from src.core import (
    TOMLConfig,
    Yolov8DetectionModel,
    Yolov8SahiDetectionModel,
    RealsenseCamera,
    DetectObstacle,
    Alarm,
    DetectCrosswalkSignal,
)

config = TOMLConfig(os.path.join(__file__, "../config.toml"))
rs_camera = RealsenseCamera(config)
yolov8 = Yolov8DetectionModel(config, config.env["yolo"]["model"])
yolov8_sahi = Yolov8SahiDetectionModel(config, config.env["yolo"]["cs_model"])
detect_obstacle = DetectObstacle(config)
alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)

dcs_window_name = "Detect Crosswalk Signal"
od_window_name = "Object Detection"
cv2.namedWindow(dcs_window_name, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(od_window_name, cv2.WINDOW_AUTOSIZE)

dcs_img = None
finished = True


def slow_processing(image):
    global finished
    if not finished:
        return

    finished = False

    def _():
        global dcs_img, finished
        img = image.copy()

        sahi_object_exists, sahi_prediction_list = yolov8_sahi(image)
        finished = True

        if sahi_object_exists:
            nearst_box = detect_cs(image, sahi_prediction_list, yolov8_sahi.category)
            if nearst_box is not None:
                img = detect_cs.draw_line(img, nearst_box)
            img = yolov8_sahi.draw_detections(img, sahi_prediction_list, depth_image)
        else:
            detect_cs.invalid()

        dcs_img = img

    Thread(target=_).start()


try:
    while (
        cv2.getWindowProperty(dcs_window_name, cv2.WND_PROP_VISIBLE) >= 1
        and cv2.getWindowProperty(od_window_name, cv2.WND_PROP_VISIBLE) >= 1
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

        slow_processing(color_image)

        if dcs_img is None:
            dcs_img = color_image.copy()

        yolov8_img = color_image.copy()
        object_exists, prediction_list = yolov8(color_image)
        if object_exists:
            yolov8_img = yolov8.draw_detections(
                yolov8_img, prediction_list, depth_image
            )

        cv2.imshow(dcs_window_name, imutils.resize(dcs_img, height=480))
        cv2.imshow("Object Detection", imutils.resize(yolov8_img, height=480))
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
