import math
from typing import Any

import cv2
import numpy as np

from .elevation_view import draw_elevation_view, draw_elevation_view_points
from .utils import (
    process_missing_points,
    draw_square,
    draw_circle,
    alert,
    draw_text,
    init_detect_points,
    is_hole,
    is_obstacle,
    is_safe_area,
)
from ..alarm.alarm import Alarm
from ..realsense_camera.realsense_camera import RealsenseCamera


class DetectObstacle:
    instance = None

    def __init__(self, config: Any):
        DetectObstacle.instance = self
        self.config_env = config.env["config"]
        self.do_env = config.env["obstacle_detection"]
        if self.do_env["missing_point_alarm"]:
            self.missing_points_buffer = []
        self.alarm = True
        self.detect_points = None
        self.detect_point_size = -1

    def init_detect_points(self):
        self.detect_points = init_detect_points(
            # self.do_env, img_height, img_width, area, split_count
            self.do_env, self.img_height, self.img_width, detect_point_size=self.detect_point_size
        )

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
        debug = self.config_env["debug"]
        missing_point_alarm = self.do_env["missing_point_alarm"]

        if debug:
            orig_color_img = color_img.copy()
            color_img, depth_img = color_img.copy(), depth_img.copy()

        min_hole_distance = math.inf
        min_obstacle_distance = math.inf

        if missing_point_alarm:
            missing_points = []

        depth_image = np.asanyarray(depth_frame.get_data())

        # area = np.array(self.do_env["area"], np.int32)
        # color_img = cv2.polylines(color_img, [area], True, (255, 255, 255), 2)
        img_height, img_width = color_img.shape[:2]
        self.img_height, self.img_width = img_height, img_width

        if self.detect_point_size != self.do_env["detect_point_size"]:
            self.detect_point_size = self.do_env["detect_point_size"]
            self.init_detect_points()

        if debug:
            border_img = np.zeros((img_height, img_width, 3), np.uint8)
            elevation_view_img = draw_elevation_view(
                RealsenseCamera.instance.pitch,
                640,
                480,
            )

        index = -1
        elevation_view_points = []

        heatmap_data = np.zeros((len(self.detect_points), len(self.detect_points[0])))

        border_circle_size = 25 if img_height == 480 else 50
        for i in range(len(self.detect_points)):
            closest_point = None
            for j in range(len(self.detect_points[i])):
                index += 1
                color_pixel = self.detect_points[i][j]
                depth_pixel = (
                    RealsenseCamera.instance.project_color_pixel_to_depth_pixel(
                        depth_frame.get_data(), color_pixel
                    )
                )
                if not depth_pixel:
                    continue

                result = RealsenseCamera.instance.depth_pixel_to_height(
                    depth_image, depth_pixel, self.do_env["camera_height"]
                )

                if not result:
                    if missing_point_alarm:
                        missing_points.append(index)
                    continue

                height, dist, lateral_dist, depth_point = result
                color = (0, 255, 0)

                if dist > 0:
                    is_safe = True

                    if closest_point is None or dist < closest_point[1]:
                        closest_point = result

                    # 如果高度低於最高坑洞的判斷標準，則代表有坑洞
                    if is_hole(self.do_env, height, lateral_dist):
                        is_safe = False
                        color = (0, 0, 255)
                        heatmap_data[i, j] = 1
                        if dist < min_hole_distance:
                            min_hole_distance = dist
                    # 如果高度低於身高，又高於最低障礙物的判斷標準，則代表有障礙物
                    if is_obstacle(self.do_env, height, dist, lateral_dist):
                        is_safe = False
                        color = (0, 255, 255)
                        heatmap_data[i, j] = 0.5
                        if dist < min_obstacle_distance:
                            min_obstacle_distance = dist

                    if debug and is_safe and is_safe_area(self.do_env, height):
                        border_img = cv2.circle(border_img, color_pixel, 50, color, -1)

                if debug:
                    x, y = color_pixel

                    square_size = self.detect_point_size // 2
                    color_img = draw_square(
                        color_img, color_pixel, color, size=square_size
                    )

                    if self.detect_point_size >= 30:
                        font_scale = self.detect_point_size / 100 * 0.75
                        color_img = draw_text(
                            color_img,
                            str(int(height)),
                            (x, y + font_scale * 40),
                            (255, 255, 255),
                            font_scale,
                        )
                        color_img = draw_text(
                            color_img, str(dist), (x, y), (255, 200, 255), font_scale
                        )
                        color_img = draw_text(
                            color_img,
                            str(lateral_dist),
                            (x, y - font_scale * 40),
                            (255, 0, 0),
                            font_scale,
                        )
                    depth_img = draw_circle(depth_img, depth_pixel, self.detect_point_size // 10, color)

            if closest_point is not None and debug:
                elevation_view_points.append(closest_point)
        if debug:
            elevation_view_img = draw_elevation_view_points(
                elevation_view_img, elevation_view_points
            )

        if missing_point_alarm:
            # 緩存消失點，如果所有緩存的消失點都有該點，則警告
            self.missing_points_buffer.append(missing_points)
            if len(self.missing_points_buffer) > self.do_env["missing_point_threshold"]:
                self.missing_points_buffer = self.missing_points_buffer[1:]

            missing_point = process_missing_points(self.missing_points_buffer)
        else:
            missing_point = None

        if Alarm.instance is not None and self.alarm:
            alert(self.do_env, min_hole_distance, min_obstacle_distance, missing_point)

        if debug:
            # 放大熱力圖資料
            large_data = cv2.resize(
                heatmap_data, (img_width, img_height), interpolation=cv2.INTER_LINEAR
            )
            large_data[0, 0] = 1

            # 將資料轉換為8位元灰階影像
            heatmap_data_normalized = cv2.normalize(
                large_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )

            # 將灰階圖轉換為熱力圖
            heatmap = cv2.applyColorMap(heatmap_data_normalized, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(heatmap, 0.1, np.zeros_like(heatmap), 1, 0)

            heatmap = cv2.addWeighted(orig_color_img, 1, heatmap, 1, 0)

            return color_img, depth_img, border_img, elevation_view_img, heatmap

    @staticmethod
    def draw_border(image, border_img, color=(0, 255, 0), mask_alpha: float = 0.2):
        mask_img = image.copy()
        border_img = cv2.cvtColor(border_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            border_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )

        # 平滑邊緣
        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            mask_img = cv2.drawContours(mask_img, [approx], -1, color, cv2.FILLED)

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

    def pause_alarm(self):
        self.alarm = False
        Alarm.instance.stop()

    def resume_alarm(self):
        self.alarm = True
