from typing import Tuple, Any, Optional, Dict

import cv2
import numpy as np

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

    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 + th), color, -1)

    return cv2.putText(
        image,
        text,
        (x1, y1 + th),
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
    for p in prediction_list:
        class_id = p[0]
        box = p[1]
        score = p[2]
        color = colors[class_id]
        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
