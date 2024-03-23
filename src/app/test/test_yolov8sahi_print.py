import os

import cv2
import imutils
import numpy as np
import requests

from src.core.models.yolov8sahi import Yolov8SahiDetectionModel
from src.core.toml_config import TOMLConfig

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))

yolov8_sahi = Yolov8SahiDetectionModel(config, config.env["yolo"]["cs_model"])

config_env = config.env["config"]
if config_env["image"].startswith("http"):
    response = requests.get(config_env["image"]).content
    np_arr = np.frombuffer(response, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
else:
    img = cv2.imread(config_env["image"])
img = imutils.resize(img, height=720)

object_exists, prediction_list = yolov8_sahi(img)
print(f"object_exists={object_exists}")
print(f"prediction_list=")
