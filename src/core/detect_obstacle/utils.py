import math

import cv2
import numpy as np

from ..alarm.alarm import Alarm

alarm = False


def is_hole(od_env, height):
    return height < od_env["highest_hole_height"]


def is_obstacle(od_env, height, dist):
    return od_env["my_height"] > height > od_env[
        "lowest_obstacle_height"
    ] and dist < get_max_obstacle_distance(od_env)


def init_detect_points(od_env, img_height, img_width, area, spilt_count=60):
    area = np.array(od_env["area"], np.int32)
    detect_points = []

    for y in range(0, int(img_height / spilt_count) + 2):
        temp = []
        for x in range(0, int(img_width / spilt_count) + 2):
            global alarm
            alarm = True
            color_pixel = (x * spilt_count, y * spilt_count)
            if cv2.pointPolygonTest(area, color_pixel, False) == -1:
                continue
            temp.append(color_pixel)
        detect_points.append(temp)
    return detect_points


def process_missing_points(missing_points_buffer):  # 消失點防抖處理
    for mp in missing_points_buffer[0]:
        if all(mp in _ for _ in missing_points_buffer):
            return mp
    return -1


def is_warning(warning_preset, distance, message, frequency):
    if distance == math.inf:
        return False
    for preset in warning_preset:
        dist, interval, string = preset["distance"], preset["interval"], preset["name"]

        if distance < dist:
            Alarm.instance.start(message.format(distance, string), interval, frequency)
            return True
    return False


def get_max_obstacle_distance(od_env):  # 取得最遠距離的障礙物判斷標準
    return od_env["obstacle_preset"][-1]["distance"]


def draw_square(image, pixel, color, thickness=1, size=30):
    return cv2.rectangle(
        image,
        (pixel[0] - size, pixel[1] - size),
        (pixel[0] + size, pixel[1] + size),
        color,
        thickness,
    )


def draw_circle(image, pixel, color, thickness=1):
    return cv2.circle(image, pixel, 3, color, -1)


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


def alert(od_env, min_hole_distance, min_obstacle_distance, missing_point=None):
    global alarm
    if missing_point is not None and missing_point != -1:
        alarm = True
        Alarm.instance.start(
            f"警報：第{missing_point}點缺失", od_env["missing_point_alarm_interval"], 2000
        )
    elif is_warning(
        od_env["hole_preset"], min_hole_distance, "警報：距離最近的坑洞 {}m ({})", 2500
    ):
        pass
    elif is_warning(
        od_env["obstacle_preset"], min_obstacle_distance, "警報：距離最近的障礙 {}m ({})", 3000
    ):
        pass
    elif alarm:
        alarm = False
        Alarm.instance.stop()
