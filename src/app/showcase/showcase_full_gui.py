import os
import sys
import threading
from threading import Thread

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from src.core.alarm.alarm import Alarm
from src.core.detect_crosswalk_signal.detect_crosswalk_signal import (
    DetectCrosswalkSignal,
)
from src.core.detect_object.detect_object import DetectObject
from src.core.detect_obstacle.detect_obstacle import DetectObstacle
from src.core.gui.gui import Gui
from src.core.models.yolov8sahi import Yolov8SahiDetectionModel
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig
from src.core.vision.vision import Vision
from src.utils.detect_blur import draw_blur_status

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))
detect_obstacle = DetectObstacle(config)
alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)
detect_object = DetectObject(config, config.env["yolo"]["model"])

vision = None
yolov8_sahi = None

def init_thread():
    global vision, yolov8_sahi
    yolov8_sahi = Yolov8SahiDetectionModel(config, config.env["yolo"]["cs_model"])
    vision = Vision(config)


threading.Thread(target=init_thread).start()

app = QApplication(sys.argv)

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

        sahi_prediction_list = yolov8_sahi(image, config.env["detect_crosswalk_signal"]["confidence_threshold"])
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


def update_frame(main_window: Gui):
    if RealsenseCamera.instance is None:
        return
    frames = RealsenseCamera.instance.pipeline.wait_for_frames(2000)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        return

    motion = RealsenseCamera.instance.combined_angle(frames)
    if not motion:
        return

    bottom_point, camera_height = RealsenseCamera.instance.auto_camera_height(depth_frame)
    # if camera_height is not None:
    #     config.env["obstacle_detection"]["camera_height"] = camera_height
    #     Gui.instance.camera_height_slider.setValue(camera_height)
    #     Gui.instance.statusbar.showMessage(f'攝影機高度已調整為: {camera_height}')

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    Gui.instance.depth_frame = depth_frame
    Gui.instance.color_frame = color_frame
    Gui.instance.depth_image = depth_image.copy()
    Gui.instance.color_image = color_image.copy()

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
    detect_object_img = color_image.copy()

    if len(prediction_list) > 0:
        detect_object_img = detect_object.draw_detections(
            detect_object_img, prediction_list, depth_image, 0
        )

    frame_number = frames.get_frame_number()
    # if yolov8_sahi:
    #     slow_processing(color_image, depth_image, frame_number)

    global dcs_img
    if dcs_img is None:
        dcs_img = color_image.copy()

    if bottom_point:
        combined_depth_colormap = RealsenseCamera.instance.draw_bottom_point(
            combined_depth_colormap, bottom_point
        )

    combined_depth_colormap = RealsenseCamera.instance.draw_motion(combined_depth_colormap)

    main_window.display_image(combined_img, 0)
    main_window.display_image(combined_depth_colormap, 1)
    main_window.display_image(detect_object_img, 2)
    main_window.display_image(heatmap, 3)


gui = Gui(config, update_frame)
gui.setWindowTitle("盲人輔助系統 Blind Assistance")
gui.show()
sys.exit(app.exec_())
