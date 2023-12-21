import numpy as np


def find_nearest(image, prediction_list, names):
    img_height, img_width = image.shape[:2]
    img_mid_pos = img_width // 2, img_height // 2

    nearest_index, nearest_dist = -1, 0

    for index, p in enumerate(prediction_list):
        class_id, box, score = p
        class_name = names[class_id]

        if class_name != "red" and class_name != "green":
            continue

        mid_pos = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2

        dist = distance(img_mid_pos, mid_pos)

        if nearest_index == -1 or dist < nearest_dist:
            nearest_index = index
            nearest_dist = dist

    return nearest_index != -1, prediction_list[nearest_index]


def distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
