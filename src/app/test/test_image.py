import os

import cv2
import imutils
import numpy as np
import requests

from src.core.models.yolov8sahi import Yolov8SahiDetectionModel
from src.core.toml_config import TOMLConfig

windowName = "Image Test"
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))
yolov8_sahi = Yolov8SahiDetectionModel(config, config.env["yolo"]["cs_model"])

config_env = config.env["config"]

if config_env["image"].startswith("http"):
    response = requests.get(config_env["image"]).content
    np_arr = np.frombuffer(response, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
else:
    image_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'resources', config_env["image"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source path {image_path} does not exist.")
    img = cv2.imread(image_path)

img = imutils.resize(img, height=720)

prediction_list = yolov8_sahi(img)

if len(prediction_list) > 0:
    combined_img = yolov8_sahi.draw_detections(img, prediction_list)
    cv2.imshow(windowName, combined_img)
else:
    cv2.imshow(windowName, img)

key = cv2.waitKey()
