import os

import cv2
import imutils
from cap_from_youtube import cap_from_youtube

from src.core import TOMLConfig, Yolov8SahiDetectionModel, Alarm, DetectCrosswalkSignal

config = TOMLConfig(os.path.join(__file__, "../config.toml"))
yolov8_sahi = Yolov8SahiDetectionModel(config, config.env["yolo"]["cs_model"])
alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)

config_env = config.env["config"]
clicked = False

windowName = "Video Test"
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = not clicked


if config_env["video"].startswith("http"):
    cap = cap_from_youtube(config_env["video"], resolution="720p")
else:
    cap = cv2.VideoCapture(config_env["video"])

count = 0

while cap.isOpened() and cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) >= 1:
    if not clicked:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            count = 0
            continue
        count += 10
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    object_exists, prediction_list = yolov8_sahi(frame)
    combined_img = imutils.resize(frame.copy(), height=720)

    if object_exists:
        combined_img = yolov8_sahi.draw_detections(combined_img, prediction_list)
        nearst_box = detect_cs(frame, prediction_list, yolov8_sahi.model.names)
        if nearst_box is not None:
            detect_cs.draw_line(combined_img, nearst_box)
    else:
        detect_cs.invalid()

    cv2.imshow(windowName, combined_img)

    cv2.setMouseCallback(windowName, on_mouse)
    key = cv2.waitKey(1)

    if key & 0xFF == ord("q") or key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
