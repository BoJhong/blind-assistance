import math

import cv2

from .. import TOMLConfig
from ..alarm.alarm import Alarm


def process_missing_points(missing_points_buffer):  # 消失點防抖處理
    for mp in missing_points_buffer[0]:
        if all(mp in _ for _ in missing_points_buffer):
            return mp
    return -1


def is_warning(warning_preset, distance, message):
    if distance == math.inf:
        return False
    for preset in warning_preset:
        dist, interval, string = (
            preset["distance"],
            preset["interval"],
            preset["name"],
        )

        if distance < dist:
            Alarm.instance.start(
                message.format(distance, string),
                interval,
            )
            return True
    return False


def get_max_obstacle_distance():  # 取得最遠距離的障礙物判斷標準
    return TOMLConfig.instance.env["obstacle_detection"]["obstacle_preset"][-1][
        "distance"
    ]


def draw_square(image, pixel, color, thickness=1):
    return cv2.rectangle(
        image,
        (pixel[0] - 30, pixel[1] - 30),
        (pixel[0] + 30, pixel[1] + 30),
        color,
        thickness,
    )


def draw_circle(image, pixel, color, thickness=1):
    return cv2.circle(image, pixel, 3, color, -1)


def draw_text(image, text, pixel, color, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

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


def alert(od_env, missing_point, min_hole_distance, min_obstacle_distance):
    if missing_point != -1:
        Alarm.instance.start(
            f"警報：第{missing_point}點缺失",
            od_env["missing_point_alarm_interval"],
        )
    elif is_warning(od_env["hole_preset"], min_hole_distance, "警報：距離最近的坑洞 {}m ({})"):
        pass
    elif is_warning(
        od_env["obstacle_preset"],
        min_obstacle_distance,
        "警報：距離最近的障礙 {}m ({})",
    ):
        pass
    else:
        Alarm.instance.stop()
