import os

import cv2
import imutils
import numpy as np
import pyrealsense2 as rs

from src.core.alarm.alarm import Alarm
from src.core.detect_crosswalk_signal.detect_crosswalk_signal import (
    DetectCrosswalkSignal,
)
from src.core.detect_obstacle.detect_obstacle import DetectObstacle
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))

rs_setting = rs.config()
rs_setting.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
rs_setting.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
rs_setting.enable_stream(rs.stream.accel)
rs_setting.enable_stream(rs.stream.gyro)
rs_camera = RealsenseCamera(config, rs_setting)

detect_obstacle = DetectObstacle(config)
alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)

window_name = "Showcase"
safe_area_window_name = "Safe Area"
ev_window_name = "Elevation View"

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(safe_area_window_name, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(ev_window_name, cv2.WINDOW_AUTOSIZE)

try:
    while (
        cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        and cv2.getWindowProperty(safe_area_window_name, cv2.WND_PROP_VISIBLE) >= 1
        and cv2.getWindowProperty(ev_window_name, cv2.WND_PROP_VISIBLE) >= 1
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

        (
            combined_img,
            combined_depth_colormap,
            border_img,
            elevation_view_img,
        ) = detect_obstacle(depth_frame, color_image, depth_colormap)
        safe_area_img = detect_obstacle.draw_border(color_image, border_img)

        combined_depth_colormap = rs_camera.draw_motion(combined_depth_colormap)
        images = np.hstack(
            (imutils.resize(combined_img, height=480), combined_depth_colormap)
        )

        cv2.imshow(window_name, images)
        cv2.imshow(safe_area_window_name, imutils.resize(safe_area_img, height=480))
        cv2.imshow(ev_window_name, elevation_view_img)
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
