import math
from typing import Tuple, Any

import numpy as np
import pyrealsense2 as rs

from .motion import get_motion, draw_motion
from .utils import (
    is_pixel_inside_image,
    get_rotation_matrix,
    intrin_and_extrin,
    default_setting,
)
from .. import TOMLConfig


class RealsenseCamera:
    instance = None

    def __init__(self, config: Any, setting=None):
        RealsenseCamera.instance = self

        self.rs_env = config.env["realsense"]
        self.pipeline = rs.pipeline()
        self.config = setting or default_setting()
        self.profile = self.pipeline.start(self.config)

        self.pitch = 0
        self.yaw = 0
        self.roll = 0

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        (
            self.depth_intrin,
            self.color_intrin,
            self.depth_to_color_extrin,
            self.color_to_depth_extrin,
        ) = intrin_and_extrin(self.profile)

    def __exit__(self):
        self.pipeline.stop()

    def combined_angle(self, frames):
        motion = get_motion(frames)
        if motion:
            self.pitch, self.yaw, self.roll = motion
            return motion

    # 計算RGB像素點距離地板的高度
    def depth_pixel_to_height(
        self, depth_image: np.ndarray, depth_pixel: Tuple[int, int], base_height
    ):
        if not is_pixel_inside_image(depth_image, depth_pixel):
            return
        dist = depth_image[depth_pixel[1], depth_pixel[0]]
        depth_point = rs.rs2_deproject_pixel_to_point(
            self.depth_intrin, depth_pixel, dist
        )

        # 計算角度（俯仰角和偏航角）
        elevation = np.arctan2(depth_point[1], depth_point[2])
        azimuth = np.arctan2(depth_point[0], depth_point[2])
        pitch_angle_radians = math.radians(self.pitch + 90) - elevation

        # 距離計算攝影機到點的水平距離
        horizontal_distance = dist * math.cos(pitch_angle_radians)

        # 使用三角函數計算高低差
        height = dist * math.sin(pitch_angle_radians) / 10 + base_height
        lateral_distance = abs(horizontal_distance * math.sin(azimuth))

        return height, int(horizontal_distance), int(lateral_distance), depth_point

    def project_color_pixel_to_depth_pixel(
        self, data: np.ndarray, from_pixel: Tuple[int, int]
    ):
        depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
            data,
            self.depth_scale,
            self.rs_env["depth_min"],
            self.rs_env["depth_max"],
            self.depth_intrin,
            self.color_intrin,
            self.depth_to_color_extrin,
            self.color_to_depth_extrin,
            from_pixel,
        )

        if depth_pixel:
            return int(depth_pixel[0]), int(depth_pixel[1])

    def draw_motion(self, image):
        return draw_motion(image, self.pitch, self.yaw, self.roll)

    def auto_camera_height(self, depth_frame):
        world_point = np.array([0, 0, 10])  # [前後, 左右, 上下]
        world_point = np.dot(
            get_rotation_matrix(self.pitch, self.yaw, self.roll), world_point
        )
        world_point = rs.rs2_project_point_to_pixel(
            self.depth_intrin,
            world_point,
        )
        if math.isnan(world_point[0]) or math.isnan(world_point[1]):
            return
        world_point = (round(world_point[1]), round(world_point[0]))

        depth_image = np.asanyarray(depth_frame.get_data())
        img_height, img_width = depth_image.shape[:2]
        if (
            img_width > world_point[0] > 0
            and img_height > world_point[1] > 0
            and (self.pitch < -90 or self.pitch > 90)
        ):
            height, dist, lateral_dist, depth_point = self.depth_pixel_to_height(
                depth_image, world_point, 0
            )
            camera_height = round(-height)
            TOMLConfig.instance.env["obstacle_detection"][
                "camera_height"
            ] = camera_height
            print(f"已將攝影機高度設置為: {camera_height}")
        else:
            print(f"請先將深度攝影機朝正下方看！")
