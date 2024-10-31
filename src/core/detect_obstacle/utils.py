import math

import cv2
import numpy as np

from src.core.alarm.alarm import Alarm
from src.core.gui.gui import Gui

alarm = False
max_obstacle_dist = 0


# 是否為有效的坑洞
def is_hole(od_env, height: int, lateral_dist: float):
    return abs(lateral_dist) < od_env["lateral_distance_threshold"] and height < od_env["highest_hole_height"]


# 是否為有效的障礙物
def is_obstacle(od_env, height: int, dist: float, lateral_dist: float):
    return (
        abs(lateral_dist) < od_env["lateral_distance_threshold"]
        and od_env["my_height"] + 5 > height > od_env["lowest_obstacle_height"]
        and dist < get_max_obstacle_distance(od_env)
    )


# 是否為有效的安全區域
def is_safe_area(od_env, height):
    return od_env["lowest_obstacle_height"] > height > od_env["highest_hole_height"]


# 初始化偵測點
def init_detect_points(od_env, img_height, img_width, area=None, detect_point_size=60):
    global max_obstacle_dist
    max_obstacle_dist = get_max_obstacle_distance(od_env)

    # area = np.array(od_env["area"], np.int32)
    detect_points = []

    for y in range(0, img_height // detect_point_size + 2):
        temp = []
        for x in range(0, img_width // detect_point_size + 2):
            color_pixel = (x * detect_point_size, y * detect_point_size)
            # if cv2.pointPolygonTest(area, color_pixel, False) == -1:
            #     continue
            temp.append(color_pixel)
        detect_points.append(temp)
    return detect_points


# 處理消失點（通常會是過遠、過近導致無法取得距離的偵測點）
def process_missing_points(missing_points_buffer):  # 消失點防抖處理
    for mp in missing_points_buffer[0]:
        if all(mp in _ for _ in missing_points_buffer):
            return mp
    return -1


# 是否需要警報
def is_warning(warning_preset, distance, message, frequency):
    if distance == math.inf:
        return False
    for preset in warning_preset:
        dist, interval, string = preset["distance"], preset["interval"], preset["name"]

        if distance < dist:
            global alarm
            alarm = True

            Gui.instance.alert_label_1.setText(message.format(distance, string))
            Alarm.instance.start(message.format(distance, string), interval, frequency)
            return True
    return False


# 取得最遠距離的障礙物判斷標準
def get_max_obstacle_distance(od_env):
    return od_env["obstacle_preset"][-1]["distance"]


# 繪製方塊
def draw_square(image, pixel, color, thickness=1, size=30):
    return cv2.rectangle(
        image,
        (pixel[0] - size, pixel[1] - size),
        (pixel[0] + size, pixel[1] + size),
        color,
        thickness,
    )


# 繪製圓形
def draw_circle(image, pixel, size, color):
    return cv2.circle(image, pixel, size, color, -1)


# 繪製文字
def draw_text(image, text, pixel, color, font_scale=0.5, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX

    pos_x, pos_y = pixel
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size
    text_x, text_y = int(pos_x - text_w / 2), int(pos_y + text_h / 2)
    return cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
    )


# 開始警報
def alert(
    od_env,
    min_hole_distance: float,
    min_obstacle_distance: float,
    missing_point: int = None,
):
    global alarm
    if missing_point is not None and missing_point != -1:
        alarm = True
        Alarm.instance.start(
            f"警報：第{missing_point}點缺失", od_env["missing_point_alarm_interval"], 2000
        )
    elif is_warning(
        od_env["hole_preset"], min_hole_distance, "距離最近的坑洞\n{}mm ({})", 2500
    ):
        pass
    elif is_warning(
        od_env["obstacle_preset"], min_obstacle_distance, "距離最近的障礙\n{}mm ({})", 3000
    ):
        pass
    elif alarm:
        alarm = False
        Alarm.instance.stop()
