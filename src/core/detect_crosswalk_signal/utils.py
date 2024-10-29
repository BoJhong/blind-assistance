import numpy as np


def find_nearest(image, prediction_list, names):
    """
    找出最接近畫面中心的行人號誌
    :param image: 要辨識的圖片
    :param prediction_list: 預測結果
    :param names: 模型所有類別名稱
    """
    img_height, img_width = image.shape[:2]
    img_mid_pos = img_width // 2, img_height // 2

    nearest_index, nearest_dist = -1, 0

    for index, p in enumerate(prediction_list):
        class_id, box, score = p
        class_name = names[class_id]

        if class_name != "red" and class_name != "green":
            continue

        width = box[2] - box[0]
        height = box[3] - box[1]

        if height / width > 2.5 or height / width < 0.7:
            print(f"行人號誌比例不對，已略過\n高度 / 寬度 = {height / width}")
            continue

        mid_pos = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2

        # 計算行人號誌中心點和畫面中心的距離
        dist = np.linalg.norm(np.array(img_mid_pos) - np.array(mid_pos))

        if nearest_index == -1 or dist < nearest_dist:
            nearest_index = index
            nearest_dist = dist

    return nearest_index != -1, prediction_list[nearest_index]

