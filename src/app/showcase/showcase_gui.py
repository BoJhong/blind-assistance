import os
import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from src.core.gui.gui import Gui
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.toml_config import TOMLConfig

config = TOMLConfig(os.path.join(os.path.dirname(__file__), "config.toml"))
app = QApplication(sys.argv)


def update_frame(main_window: Gui):
    if RealsenseCamera.instance is None:
        return
    frames = RealsenseCamera.instance.pipeline.wait_for_frames(60 * 1000)  # timeout時間設為1分鐘
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )

    main_window.display_image(color_image, 0)
    main_window.display_image(depth_image, 1)
    main_window.display_image(depth_colormap, 2)


gui = Gui(config, update_frame)
gui.setWindowTitle("Showcase")
gui.show()
sys.exit(app.exec_())
