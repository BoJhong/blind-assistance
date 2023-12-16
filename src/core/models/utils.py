from typing import Tuple

import cv2
import numpy as np

from .. import TOMLConfig
from ..realsense_camera.utils import get_middle_dist


def draw_detections(
    image, prediction_list, category, colors, depth_data=None, mask_alpha=0.3
):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, prediction_list, colors, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in prediction_list:
        color = colors[class_id]
        draw_box(det_img, box, color)
        if TOMLConfig.instance.env["yolo"]["debug"]:
            label = category[class_id]
            caption = f"{label} {int(score * 100)}%"
            if depth_data is not None:
                distance = get_middle_dist(det_img, box, depth_data, 12)
                if distance != -1:
                    distance = str(distance / 1000)[:4]
                    caption += f" ({distance}m)"

            draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(
    image: np.ndarray,
    text: str,
    box: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    font_size: float = 0.001,
    text_thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(
        text=text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_size,
        thickness=text_thickness,
    )
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

    return cv2.putText(
        image,
        text,
        (x1, y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_masks(
    image: np.ndarray, prediction_list, colors, mask_alpha: float = 0.3
) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for class_id, box, score in prediction_list:
        color = colors[class_id]
        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
