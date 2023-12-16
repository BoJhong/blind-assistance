import os

import cv2
import imutils
import numpy as np
import requests

from src.core import TOMLConfig, YOLOv8

windowName = "Image Test"
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

setting = TOMLConfig(os.path.join(__file__, "../config.toml"))
yolov8 = YOLOv8(setting.env["yolo"]["cs_model"])

if setting.env["image"].startswith("http"):
    response = requests.get(setting.env["image"]).content
    np_arr = np.frombuffer(response, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
else:
    img = cv2.imread(setting.env["image"])

prediction_list = yolov8(img)
if prediction_list and len(prediction_list) > 0:
    object_exists, combined_img = yolov8.draw_detections(img, prediction_list)
    cv2.imshow(windowName, imutils.resize(combined_img, height=480))
else:
    cv2.imshow(windowName, imutils.resize(img, height=480))

key = cv2.waitKey()
