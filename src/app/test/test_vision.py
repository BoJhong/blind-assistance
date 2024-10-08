import os
import threading

import cv2
import imutils
from PIL import Image
from cap_from_youtube import cap_from_youtube

from src.core.alarm.alarm import Alarm
from src.core.detect_crosswalk_signal.detect_crosswalk_signal import (
    DetectCrosswalkSignal,
)
from src.core.models.yolov8sahi import Yolov8SahiDetectionModel
from src.core.toml_config import TOMLConfig
from src.core.vision.vision import Vision

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))

alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)
vision = Vision(config)

config_env = config.env["config"]
clicked = False

window_name = "Video Test"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = not clicked


if config_env["video"].startswith("http"):
    cap = cap_from_youtube(config_env["video"], resolution="720p")
else:
    video_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'resources', config_env["video"])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source path {video_path} does not exist.")
    cap = cv2.VideoCapture(video_path)

count = 0

while cap.isOpened() and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
    if not clicked:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            count = 0
            continue
        count += config.env["config"]["skip_frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    combined_img = imutils.resize(frame.copy(), height=720)

    cv2.imshow(window_name, combined_img)

    cv2.setMouseCallback(window_name, on_mouse)
    key = cv2.waitKey(1)

    if key & 0xFF == ord("q") or key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
    if key & 0xFF == ord("v"):
        vision.predict(Image.fromarray(frame), translate=True)
