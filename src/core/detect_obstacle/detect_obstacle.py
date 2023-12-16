import math

import cv2
import numpy as np

from .utils import (
    process_missing_points,
    get_max_obstacle_distance,
    draw_square,
    draw_circle,
    alert,
    draw_text,
)
from ..toml_config import TOMLConfig
from ..alarm.alarm import Alarm
from ..realsense_camera.realsense_camera import RealsenseCamera


class DetectObstacle:
    instance = None

    def __init__(self):
        DetectObstacle.instance = self

        self.od_env = TOMLConfig.instance.env["obstacle_detection"]
        self.missing_points_buffer = []
        self.alarm = True
        self.detect_points = None

    def __call__(
        self,
        depth_frame: any,
        color_img: np.asanyarray = None,
        depth_img: np.asanyarray = None,
    ):
        """
        :param depth_frame: 深度影像
        :param color_img: 彩度圖片
        :param depth_img: 深度圖片
        :return: 繪製過後的彩度圖片與深度圖片
        """

        draw = color_img is not None and depth_img is not None
        if draw:
            color_img, depth_img = color_img.copy(), depth_img.copy()
        min_hole_distance = math.inf
        min_obstacle_distance = math.inf
        index = 0
        max_obstacle_dist = get_max_obstacle_distance()
        missing_points = []

        depth_image = np.asanyarray(depth_frame.get_data())

        area = np.array(self.od_env["area"], np.int32)
        color_img = cv2.polylines(color_img, [area], True, (255, 255, 255), 2)
        img_height, img_width = color_img.shape[:2]

        if self.detect_points is None:
            self.init_detect_points(img_height, img_width, area)

        for color_pixel in self.detect_points:
            depth_pixel = (
                RealsenseCamera.instance.rs2_project_color_pixel_to_depth_pixel(
                    depth_frame.get_data(), color_pixel
                )
            )
            if not depth_pixel:
                continue

            result = RealsenseCamera.instance.depth_pixel_to_height(
                depth_image, depth_pixel, self.od_env["camera_height"]
            )

            if result:
                height, dist, depth_point = result
                color = (0, 255, 0)

                if dist > 0:
                    # 如果高度低於最高坑洞的判斷標準，則代表有坑洞
                    if height < self.od_env["highest_hole_height"]:
                        color = (0, 0, 255)
                        if dist < min_hole_distance:
                            min_hole_distance = dist

                    # 如果高度低於身高，又高於最低障礙物的判斷標準，則代表有障礙物
                    if (
                        self.od_env["my_height"]
                        > height
                        > self.od_env["lowest_obstacle_height"]
                        and dist < get_max_obstacle_distance()
                    ):
                        color = (0, 255, 255)
                        if dist < min_obstacle_distance:
                            min_obstacle_distance = dist

                if self.od_env["debug"] and draw:
                    x, y = color_pixel
                    color_img = draw_text(
                        color_img, str(int(height)), (x, y + 10), (255, 255, 255)
                    )
                    color_img = draw_text(
                        color_img, str(dist), (x, y - 10), (255, 200, 255)
                    )
                    if color_img is not None:
                        color_img = draw_square(color_img, color_pixel, color)
                    if depth_img is not None:
                        depth_img = draw_circle(depth_img, depth_pixel, color)
            else:
                missing_points.append(index)
            index += 1

        # 緩存消失點，如果所有緩存的消失點都有該點，則警告
        self.missing_points_buffer.append(missing_points)
        if len(self.missing_points_buffer) > self.od_env["missing_point_threshold"]:
            self.missing_points_buffer = self.missing_points_buffer[1:]

        missing_point = process_missing_points(self.missing_points_buffer)

        if Alarm.instance is not None and self.alarm:
            alert(
                self.od_env,
                missing_point,
                min_hole_distance,
                min_obstacle_distance,
            )

        return color_img, depth_img

    def init_detect_points(self, img_height, img_width, area):
        area = np.array(self.od_env["area"], np.int32)
        self.detect_points = []

        for x in range(0, int(img_width / 60) + 2):
            for y in range(0, int(img_height / 60) + 2):
                color_pixel = (x * 60, y * 60)
                if cv2.pointPolygonTest(area, color_pixel, False) == -1:
                    continue
                self.detect_points.append(color_pixel)

    def pause_alarm(self):
        self.alarm = False
        Alarm.instance.stop()

    def resume_alarm(self):
        self.alarm = True
