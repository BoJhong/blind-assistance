from typing import Optional, Dict, Any

import numpy as np

from .utils import draw_detections
from .. import TOMLConfig


class DetectionModel:
    instance = None

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        load_at_init: bool = True,
    ):
        """
        初始化檢測模型
        :param model_path: 模型檔案路徑
        """
        self.instance = self
        self.load_env()
        if load_at_init:
            self.model_path = model_path
            self.confidence_threshold = confidence_threshold
            self.load_model()
            rng = np.random.default_rng(1)
            self.colors = rng.uniform(0, 255, size=(len(self.category), 3))

    def load_env(self):
        pass

    def load_model(self):
        """
        這個函數的實現方式必須是初始化檢測模型並將其設定為self.model
        （需要利用self.model_path、self.config_path和self.device）
        """
        raise NotImplementedError()

    def set_model(self, model: Any, **kwargs):
        """
        這個函數的實現能夠從已載入的模型實例化一個DetectionModel
        Args:
            model: Any
                已載入的模型
        """
        raise NotImplementedError()

    def __call__(self, img: np.ndarray):
        pass

    def draw_detections(
        self,
        image: np.ndarray,
        prediction_list: any,
        depth_data: np.ndarray = None,
        mask_alpha: float = 0.4,
    ):
        """
        繪製預測結果
        :param image: 原始圖片
        :param prediction_list: 預測結果
        :param depth_data: 深度資料
        :param mask_alpha: 遮罩透明度
        """
        return draw_detections(
            image, prediction_list, self.category, self.colors, depth_data, mask_alpha
        )
