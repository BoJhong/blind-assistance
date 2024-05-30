import threading
from pathlib import Path

import cv2
from cap_from_youtube import cap_from_youtube
from PIL import Image
import os

from src.core.alarm.alarm import Alarm
from src.core.toml_config import TOMLConfig
from src.core.vision.vision import Vision

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))

alarm = Alarm(config)
vision = Vision(config)

window_name = "Vision Test"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

config_env = config.env["config"]
clicked = False

frame = None
vision_response = None


def stream_fn(text: str):
    global vision_response
    vision_response = text

def slow_processing():
    global vision, frame, vision_response
    while 1:
        if frame is not None:
            vision_response = vision.predict(Image.fromarray(frame), speak=True, stream=stream_fn)


threading.Thread(target=slow_processing).start()


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = not clicked


if config_env["video"].startswith("http"):
    cap = cap_from_youtube(config_env["video"], resolution="720p")
else:
    video_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'resources', config_env["video"])
    if not Path(video_path).exists():
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

    vision_image = frame.copy()
    if vision_response is not None:
        vision_image = vision.draw(vision_image, vision_response)

    cv2.imshow(window_name, vision_image)

    cv2.setMouseCallback(window_name, on_mouse)
    key = cv2.waitKey(1)

    if key & 0xFF == ord("q") or key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
