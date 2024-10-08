import os
from threading import Thread

import cv2
import imutils
import numpy as np
from PIL import Image

from src.core.alarm.alarm import Alarm
from src.core.detect_crosswalk_signal.detect_crosswalk_signal import (
    DetectCrosswalkSignal,
)
from src.core.detect_object.detect_object import DetectObject
from src.core.detect_obstacle.detect_obstacle import DetectObstacle
from src.core.models.yolov8sahi import Yolov8SahiDetectionModel
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig
from src.core.vision.vision import Vision
from src.utils.detect_blur import draw_blur_status

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))
rs_camera = RealsenseCamera(config)
yolov8_sahi = Yolov8SahiDetectionModel(config, config.env["yolo"]["cs_model"])
detect_obstacle = DetectObstacle(config)
alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)
detect_object = DetectObject(config, config.env["yolo"]["model"])
vision = Vision(config)

window_name = "Showcase"
dcs_window_name = "Detect Crosswalk Signal"
objd_window_name = "Object Detection and Heatmap"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(dcs_window_name, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(objd_window_name, cv2.WINDOW_AUTOSIZE)

dcs_img = None
last_process_frame = 0
blurry = False
slow_thread = None
finished = True
frame_number = 0


def slow_processing(image, depth_image, n):
    global blurry, last_process_frame, finished
    if not finished:
        return

    mean = None
    # if detect_cs.is_none():
    #     if n % 15 != 0 and not blurry:
    #         return
    #
    #     (mean, blurry) = detect_blur_fft(image)
    #     if blurry and n // 15 - last_process_frame < 5:
    #         return

    last_process_frame = n // 15
    finished = False

    def fn():
        global dcs_img, blurry, frame_number, finished
        img = image.copy()

        if mean is not None:
            img = draw_blur_status(image, mean, blurry)

        sahi_prediction_list = yolov8_sahi(image)
        finished = True

        if len(sahi_prediction_list) > 0:
            nearst_box = detect_cs(image, sahi_prediction_list, yolov8_sahi.category)
            if nearst_box is not None:
                img = detect_cs.draw_line(img, nearst_box)
            img = yolov8_sahi.draw_detections(img, sahi_prediction_list, depth_image)
        else:
            detect_cs.invalid()

        dcs_img = img

    Thread(target=fn).start()


try:
    while (
        cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        and cv2.getWindowProperty(dcs_window_name, cv2.WND_PROP_VISIBLE) >= 1
        and cv2.getWindowProperty(objd_window_name, cv2.WND_PROP_VISIBLE) >= 1
    ):
        frames = rs_camera.pipeline.wait_for_frames(60 * 1000)  # timeout時間設為1分鐘
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
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        (
            combined_img,
            combined_depth_colormap,
            border_img,
            elevation_view_img,
            heatmap,
        ) = detect_obstacle(depth_frame, color_image, depth_colormap)

        prediction_list = detect_object(color_image, depth_frame)

        if len(prediction_list) > 0:
            heatmap = detect_object.draw_detections(
                heatmap, prediction_list, depth_image, 0
            )

        frame_number = frames.get_frame_number()
        slow_processing(color_image, depth_image, frame_number)

        if dcs_img is None:
            dcs_img = color_image.copy()

        if bottom_point:
            combined_depth_colormap = rs_camera.draw_bottom_point(
                combined_depth_colormap, bottom_point
            )

        combined_depth_colormap = rs_camera.draw_motion(combined_depth_colormap)
        images = np.hstack(
            (imutils.resize(combined_img, height=480), combined_depth_colormap)
        )

        cv2.imshow(window_name, images)
        cv2.imshow(dcs_window_name, imutils.resize(dcs_img, height=480))
        cv2.imshow(objd_window_name, imutils.resize(heatmap, height=480))
        key = cv2.waitKey(1)

        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()

        if key & 0xFF == ord("s") and camera_height:
            TOMLConfig.instance.env["obstacle_detection"][
                "camera_height"
            ] = camera_height
        if key & 0xFF == ord("v"):
            detect_cs.vision_countdown(Image.fromarray(color_image))
finally:
    rs_camera.pipeline.stop()
    alarm.cleanup()
    cv2.destroyAllWindows()
