import os

import cv2
import imutils
import numpy as np

from src.core.alarm.alarm import Alarm
from src.core.detect_object.detect_object import DetectObject
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))

rs_camera = RealsenseCamera(config)
alarm = Alarm(config)
detect_object = DetectObject(config, config.env["yolo"]["model"])

objd_window_name = "Object Detection"
cv2.namedWindow(objd_window_name, cv2.WINDOW_AUTOSIZE)

dcs_img = None
finished = True

try:
    while cv2.getWindowProperty(objd_window_name, cv2.WND_PROP_VISIBLE) >= 1:
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

        yolov8_img = color_image.copy()

        prediction_list = detect_object(color_image, depth_frame)

        if len(prediction_list) > 0:
            yolov8_img = detect_object.draw_detections(
                yolov8_img, prediction_list, depth_image
            )

        cv2.imshow(objd_window_name, imutils.resize(yolov8_img, height=480))
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
