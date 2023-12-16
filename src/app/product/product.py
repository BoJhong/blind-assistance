import os

import numpy as np

from src.core import (
    TOMLConfig,
    YOLOv8,
    RealsenseCamera,
    DetectObstacle,
    Alarm,
    DetectCrosswalkSignal,
)

setting = TOMLConfig(os.path.join(__file__, "../config.toml"))
rs_camera = RealsenseCamera()
yolov8 = YOLOv8(setting.env["yolo"]["traffic_model"])
detect_obstacle = DetectObstacle()
alarm = Alarm()
detect_cs = DetectCrosswalkSignal()

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

        object_exists, prediction_list = yolov8(color_image)
        if object_exists:
            detect_cs(color_image, prediction_list, yolov8.model.names)
        else:
            detect_cs.invalid()
finally:
    rs_camera.pipeline.stop()
    alarm.cleanup()
