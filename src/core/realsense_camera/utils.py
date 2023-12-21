import math
import random
from typing import Tuple

import numpy as np
import pyrealsense2 as rs


def default_setting():
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    return config


def intrin_and_extrin(profile: any):
    depth_intrin = (
        profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    )
    color_intrin = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )
    depth_to_color_extrin = (
        profile.get_stream(rs.stream.depth)
        .as_video_stream_profile()
        .get_extrinsics_to(profile.get_stream(rs.stream.color))
    )
    color_to_depth_extrin = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_extrinsics_to(profile.get_stream(rs.stream.depth))
    )
    return depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin


def is_pixel_inside_image(image: np.ndarray, pixel: Tuple[int, int]) -> bool:
    img_height, img_width = image.shape[:2]
    return 0 <= pixel[0] < img_width and 0 <= pixel[1] < img_height


def get_rotation_matrix(pitch: int, yaw: int, roll: int):
    pitch = math.radians(pitch)
    yaw = math.radians(0)
    roll = math.radians(0)
    return np.array(
        [
            [
                math.cos(yaw) * math.cos(pitch),
                math.cos(yaw) * math.sin(pitch) * math.sin(roll)
                - math.sin(yaw) * math.cos(roll),
                math.cos(yaw) * math.sin(pitch) * math.cos(roll)
                + math.sin(yaw) * math.sin(roll),
            ],
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(yaw) * math.sin(pitch) * math.sin(roll)
                + math.cos(yaw) * math.cos(roll),
                math.sin(yaw) * math.sin(pitch) * math.cos(roll)
                - math.cos(yaw) * math.sin(roll),
            ],
            [
                -math.sin(pitch),
                math.cos(pitch) * math.sin(roll),
                math.cos(pitch) * math.cos(roll),
            ],
        ]
    )


def project_color_pixel_to_depth_pixel(
    depth_data: np.ndarray, color_data: np.ndarray, color_pixel: Tuple[int, int]
):
    depth_shape, color_shape = depth_data.shape[:2], color_data.shape[:2]
    w, h = depth_shape[1] / color_shape[1], depth_shape[0] / color_shape[0]
    return int(color_pixel[0] * w), int(color_pixel[1] * h)


def get_middle_dist(image: np.ndarray, box, depth_data, rand_num):
    distance_list = []
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]  # 確定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))  # 確定深度搜索範圍

    for i in range(rand_num):
        bias = random.randint(-min_val // 4, min_val // 4)
        color_pixel = mid_pos[1] + bias, mid_pos[0] + bias
        depth_pixel = project_color_pixel_to_depth_pixel(depth_data, image, color_pixel)
        if not is_pixel_inside_image(depth_data, depth_pixel):
            continue

        dist = depth_data[depth_pixel]

        if dist:
            distance_list.append(dist)

    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[
        rand_num // 2 - rand_num // 4 : rand_num // 2 + rand_num // 4
    ]  # 冒泡排序+中值濾波

    if len(distance_list) > 0:
        return np.mean(distance_list)
    else:
        return -1
