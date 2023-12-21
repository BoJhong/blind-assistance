import math

import cv2
import numpy as np

camera_center = (0, 0)
camera_pitch = 0


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    angle_radians = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle_radians) * (px - ox) - math.sin(angle_radians) * (py - oy)
    qy = oy + math.sin(angle_radians) * (px - ox) + math.cos(angle_radians) * (py - oy)
    return qx, qy


def draw_elevation_view(pitch, width, height):
    global camera_center, camera_pitch
    camera_pitch = pitch
    shape = (height, width, 3)
    image = np.full(shape, 255).astype(np.uint8)
    x = 50
    y = int(height / 2 - height / 2 * math.sin(math.radians(abs(pitch) - 90)) * 0.9)
    camera_center = (x, y)
    point1 = rotate(camera_center, (x + 1000, y), -pitch - 58)
    point2 = rotate(camera_center, (x + 1000, y), -pitch - 119)
    points = np.array([camera_center, point1, point2], np.int32)
    cv2.fillPoly(image, [points], (0, 255, 255))
    cv2.circle(image, camera_center, 20, (0, 0, 255), -1)
    return image


def draw_elevation_view_points(image, elevation_view_points):
    global camera_center, camera_pitch
    image = image.copy()
    points = []
    x, y = camera_center
    for point in elevation_view_points:
        height, dist, depth_point = point
        elevation = np.arctan2(depth_point[1], depth_point[2])
        point = rotate(
            camera_center,
            (x + dist / 2, y),
            -camera_pitch - 90 + math.degrees(elevation),
        )
        points.append(point)
    background_points = points.copy()
    points = np.array(points, np.int32)

    background_points.append(rotate(camera_center, (x + 1000, y), -camera_pitch - 58))
    background_points.append(rotate(camera_center, (x + 1000, y), -camera_pitch - 119))
    background_points = np.array(background_points, np.int32)
    cv2.fillPoly(image, pts=[background_points], color=(255, 255, 255))

    cv2.polylines(image, pts=[points], isClosed=False, color=(0, 0, 0), thickness=3)
    return image
