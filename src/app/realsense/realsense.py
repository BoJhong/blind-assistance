import os

import numpy as np

from src.core.alarm.alarm import Alarm
from src.core.detect_obstacle.detect_obstacle import DetectObstacle
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))
rs_camera = RealsenseCamera(config)
detect_obstacle = DetectObstacle(config)
alarm = Alarm(config)

last_process_frame = 0
blurry = False
slow_thread = None
finished = True
frame_number = 0

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
finally:
    rs_camera.pipeline.stop()
    alarm.cleanup()
