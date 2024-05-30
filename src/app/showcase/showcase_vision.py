import os
import threading

import cv2
import imutils
import numpy as np
from PIL import Image

from src.core.alarm.alarm import Alarm
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig
from src.core.vision.vision import Vision

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))

rs_camera = RealsenseCamera(config)
alarm = Alarm(config)
vision = Vision(config)

v_window_name = "Vision"
cv2.namedWindow(v_window_name, cv2.WINDOW_AUTOSIZE)

color_image = None
depth_image = None
vision_response = None

def slow_processing():
    global vision, color_image, depth_image, vision_response
    while 1:
        if color_image is not None:
            vision_response = vision.predict(Image.fromarray(color_image))


threading.Thread(target=slow_processing).start()

try:
    while cv2.getWindowProperty(v_window_name, cv2.WND_PROP_VISIBLE) >= 1:
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

        vision_image = color_image.copy()
        if vision_response is not None:
            vision_image = vision.draw(vision_image, vision_response)

        cv2.imshow(v_window_name, imutils.resize(vision_image, height=480))
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
