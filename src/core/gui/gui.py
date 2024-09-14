import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QLabel
from PyQt5.uic import loadUi
import cv2
import numpy as np

from src.core.realsense_camera.realsense_camera import RealsenseCamera

class Gui(QMainWindow):
    instance = None

    def __init__(self, config, update_frame_func, stop_func=None):
        Gui.instance = self
        super(Gui, self).__init__()
        loadUi("gui/main_window.ui", self)
        self.config = config
        self.update_frame_func = update_frame_func
        print(self.update_frame_func)

        self.is_zoomed_in = False
        self.labels = [self.image_label_1, self.image_label_2, self.image_label_3, self.image_label_4]
        for label in self.labels:
            label.mousePressEvent = self.create_click_handler(label)

        # 設置 QTimer 來更新所有 QLabel
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(int(1000 / 15))  # 15FPS

        self.stop = stop_func

    def create_click_handler(self, label):
        """返回一個處理點擊事件的函數"""
        def handle_click(event):
            self.on_label_clicked(label)

        return handle_click

    def update_frames(self):
        """每次 QTimer 超時後更新所有 QLabel"""
        self.update_frame_func(self)

    def closeEvent(self, event):
        """視窗關閉時，停止所有 Realsense 摄像機的執行"""
        if self.stop:
            self.stop()

        event.accept()

    def on_label_clicked(self, clicked_label):
        """當點擊某個 QLabel 時，放大該 QLabel 並隱藏其他 QLabel"""
        if self.is_zoomed_in:
            # 如果已經有一個圖像被放大了，恢復原狀
            for label in self.labels:
                label.show()
            self.is_zoomed_in = False
        else:
            # 隱藏其他 QLabel，僅顯示被點擊的
            for label in self.labels:
                if label != clicked_label:
                    label.hide()
            self.is_zoomed_in = True

    def display_image(self, img, window=0):
        label = None

        if window == 0:
            label = self.image_label_1
        elif window == 1:
            label = self.image_label_2
        elif window == 2:
            label = self.image_label_3
        elif window == 3:
            label = self.image_label_4

        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR to RGB
        outImage = outImage.rgbSwapped()

        pixmap = QPixmap.fromImage(outImage)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

