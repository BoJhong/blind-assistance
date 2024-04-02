import os

import cv2
import imutils
import numpy as np
import requests

from src.core.models.yolov8 import Yolov8DetectionModel
from src.core.toml_config import TOMLConfig

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))
yolov8 = Yolov8DetectionModel(config, config.env["yolo"]["cs_model"])

config_env = config.env["config"]

if config_env["image"].startswith("http"):
    response = requests.get(config_env["image"]).content
    np_arr = np.frombuffer(response, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
else:
    img = cv2.imread(os.path.join(os.path.dirname(__file__), config_env["image"]))

img = imutils.resize(img, height=720)

object_exists, prediction_list = yolov8(img)

if object_exists is not None:
    for prediction in prediction_list:
        class_id, box, score = prediction
        label = yolov8.category[class_id]
        print(f"Class Name: {label}, Box: {box}, Score: {score}")
else:
    print("No object detected")
