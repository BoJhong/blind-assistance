import os
import cv2
from cap_from_youtube import cap_from_youtube

from src.core.alarm.alarm import Alarm
from src.core.detect_crosswalk_signal.detect_crosswalk_signal import (
    DetectCrosswalkSignal,
)
from src.core.models.yolov8 import Yolov8DetectionModel
from src.core.toml_config import TOMLConfig

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))

yolov8 = Yolov8DetectionModel(config, config.env["yolo"]["cs_model"])
alarm = Alarm(config)
detect_cs = DetectCrosswalkSignal(config)

config_env = config.env["config"]

if config_env["video"].startswith("http"):
    cap = cap_from_youtube(config_env["video"], resolution="720p")
else:
    cap = cv2.VideoCapture(os.path.join(os.path.dirname(__file__), config_env["video"]))

count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        count = 0
        continue
    print(count)
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    object_exists, prediction_list = yolov8(frame)

    if object_exists:
        for prediction in prediction_list:
            class_id, box, score = prediction
            label = yolov8.category[class_id]
            print(f"Class Name: {label}, Box: {box}, Score: {score}")
    else:
        print("No object detected")
        
    count += config.env["config"]["skip_frame"]
