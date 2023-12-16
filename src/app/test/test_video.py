import os

import cv2
import imutils
from cap_from_youtube import cap_from_youtube

from src.core import TOMLConfig, YOLOv8, Alarm, DetectCrosswalkSignal

setting = TOMLConfig(os.path.join(__file__, "../config.toml"))
yolov8 = YOLOv8(setting.env["yolo"]["traffic_model"])
alarm = Alarm()
detect_cs = DetectCrosswalkSignal()

clicked = False

windowName = "Video Test"
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = not clicked


if setting.env["video"].startswith("http"):
    cap = cap_from_youtube(setting.env["video"], resolution="720p")
else:
    cap = cv2.VideoCapture(setting.env["video"])

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

    object_exists, prediction_list = yolov8(frame)
    combined_img = frame.copy()

    if object_exists:
        combined_img = yolov8.draw_detections(combined_img, prediction_list)
        nearst_box = detect_cs(frame, prediction_list, yolov8.model.names)

        if nearst_box:
            combined_img = detect_cs.draw_line(combined_img, nearst_box)
    else:
        detect_cs.invalid()

    cv2.imshow(windowName, imutils.resize(combined_img, height=480))

    cv2.setMouseCallback(windowName, on_mouse)
    key = cv2.waitKey(1)

    if key & 0xFF == ord("q") or key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
