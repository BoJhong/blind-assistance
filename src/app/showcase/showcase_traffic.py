import os
from collections import defaultdict

import cv2
import imutils
import numpy as np

from src.core.alarm.alarm import Alarm
from src.core.detect_crosswalk_signal.detect_crosswalk_signal import (
    DetectCrosswalkSignal,
)
from src.core.models.yolov8 import Yolov8DetectionModel
from src.core.models.yolov8sahi import Yolov8SahiDetectionModel
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))

rs_camera = RealsenseCamera(config)
yolov8 = Yolov8DetectionModel(config, config.env["yolo"]["model"])
yolov8_sahi = Yolov8SahiDetectionModel(config, config.env["yolo"]["cs_model"])
alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)

dcs_window_name = "Detect Crosswalk Signal"
cv2.namedWindow(dcs_window_name, cv2.WINDOW_AUTOSIZE)

dcs_img = None
finished = True

track_history = defaultdict(lambda: [])

try:
    while cv2.getWindowProperty(dcs_window_name, cv2.WND_PROP_VISIBLE) >= 1:
        frames = rs_camera.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        motion = rs_camera.combined_angle(frames)
        if not motion:
            continue

        bottom_point, camera_height = rs_camera.auto_camera_height(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        sahi_prediction_list = yolov8_sahi(color_image)
        dcs_img = color_image.copy()

        if len(sahi_prediction_list) > 0:
            nearst_box = detect_cs(color_image, sahi_prediction_list, yolov8_sahi.category)
            if nearst_box is not None:
                dcs_img = detect_cs.draw_line(dcs_img, nearst_box)
            dcs_img = yolov8_sahi.draw_detections(dcs_img, sahi_prediction_list, depth_image)
        else:
            detect_cs.invalid()

        cv2.imshow(dcs_window_name, imutils.resize(dcs_img, height=480))
        key = cv2.waitKey(1)

        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord("s") and camera_height:
            TOMLConfig.instance.env["obstacle_detection"][
                "camera_height"
            ] = camera_height
finally:
    rs_camera.pipeline.stop()
    alarm.cleanup()
    cv2.destroyAllWindows()
