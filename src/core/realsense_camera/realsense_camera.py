import math
from typing import Tuple, Any

import cv2
import numpy as np
import pyrealsense2 as rs

from src.core.realsense_camera.motion import get_motion, draw_motion
from src.core.realsense_camera.utils import (
    is_pixel_inside_image,
    get_rotation_matrix,
    intrin_and_extrin,
    default_setting,
)


class RealsenseCamera:
    instance = None

    def __init__(self, config: Any, file=None, setting=None):
        """
        初始化RealSense相機
        :param config: toml設定檔
        :param setting: 深度攝影機設置
        """
        RealsenseCamera.instance = self

        self.rs_env = config.env["realsense"]
        self.pipeline = rs.pipeline()
        self.config = setting or default_setting(file)
        self.profile = self.pipeline.start(self.config)

        if file:
            device = self.config.resolve(self.pipeline).get_device()
            playback = device.as_playback()
            playback.set_real_time(False)

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

    @property
    def motion(self):
        return self.pitch, self.yaw, self.roll

    @property
    def motion_radians(self):
        return math.radians(self.pitch), math.radians(self.yaw), math.radians(self.roll)

    def __exit__(self):
        """
        停止深度攝影機
        """
        self.pipeline.stop()

    def combined_angle(self, frames):
        """
        計算深度攝影機的姿態
        @param frames: 深度攝影機幀
        """
        camera_motion = get_motion(frames)
        if camera_motion:
            self.pitch, self.yaw, self.roll = camera_motion
            return camera_motion

    def depth_pixel_to_height(
        self, depth_image: np.ndarray, depth_pixel: Tuple[int, int], base_height
    ):
        """
        計算RGB像素點距離地板的高度
        :param depth_image: 深度圖片
        :param depth_pixel: 深度像素點
        :param base_height: 攝影機的基本高度
        """
        # 檢查像素是否在圖片內
        if not is_pixel_inside_image(depth_image, depth_pixel):
            return

        # 從深度圖片中獲取深度
        dist = depth_image[depth_pixel[1], depth_pixel[0]]
        # 將彩度圖像素點轉換為深度像素點
        depth_point = rs.rs2_deproject_pixel_to_point(
            self.depth_intrin, depth_pixel, dist
        )

        # 計算角度（俯仰角和偏航角）
        elevation = np.arctan2(depth_point[1], depth_point[2])
        azimuth = np.arctan2(depth_point[0], depth_point[2])
        pitch_angle_radians = math.radians(self.pitch + 90) - elevation

        # 距離計算攝影機到點的水平距離
        horizontal_dist = dist * math.cos(pitch_angle_radians)

        # 使用三角函數計算高低差
        height = dist * math.sin(pitch_angle_radians) / 10 + base_height
        # 計算橫向距離
        lateral_distance = horizontal_dist * math.sin(azimuth)

        return height, int(abs(horizontal_dist)), int(lateral_distance), depth_point

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
        """
        自動設置攝影機高度
        :param depth_frame: 深度幀
        return: 像素座標
        """

        # 設置相對座標
        world_point = np.array([0, 0, 10])  # [前後, 左右, 上下]
        # 使用攝影機姿態將相對座標轉換為世界座標
        world_point = np.dot(get_rotation_matrix(self.motion_radians), world_point)
        # 將世界座標轉換為畫面像素座標
        pixel = rs.rs2_project_point_to_pixel(
            self.depth_intrin,
            world_point,
        )

        # 檢查世界座標是否為有效的數字
        if math.isnan(pixel[0]) or math.isnan(pixel[1]):
            return
        # 四捨五入世界座標
        pixel = (round(pixel[1]), round(pixel[0]))

        depth_image = np.asanyarray(depth_frame.get_data())
        img_height, img_width = depth_image.shape[:2]

        # 檢查像素點是否在圖片內，並且攝影機是否朝下
        if (
            img_width > pixel[0] > 0
            and img_height > pixel[1] > 0
            and (self.pitch < -90 or self.pitch > 90)
        ):
            # 計算像素點的高度
            height, dist, lateral_dist, depth_point = self.depth_pixel_to_height(
                depth_image, pixel, 0
            )

            # 將高度四捨五入並保存為攝影機高度
            camera_height = round(-height)

            # 回傳像素座標
            return pixel, camera_height
        return None, None

    def draw_bottom_point(self, image, bottom_point):
        return cv2.circle(image, bottom_point, 10, (255, 255, 255), -1)
