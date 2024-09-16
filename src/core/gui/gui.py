import threading
from time import sleep

from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.uic import loadUi

import gui.compiled_resources  # type: ignore
from src.core.alarm.tts import TTS
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.vision.vision import Vision


class RealSenseThread(QThread):
    frame_data_signal = pyqtSignal(str)

    def __init__(self, config, run_func, file=None):
        super().__init__()
        self.config = config
        self.run_func = run_func
        self.file = file
        self.is_running = False

    def run(self):
        try:
            self.is_running = True
            self.rs_camera = RealsenseCamera(self.config, file=self.file)
            Gui.instance.toggle_camera_btn.setText('停止')
            Gui.instance.statusbar.showMessage('深度攝影機啟動完畢！')
            Gui.instance.is_streaming = True
            Gui.instance.toggle_camera_btn.setEnabled(True)
            Gui.instance.toggle_camera_btn.setChecked(True)

            while self.is_running:
                try:
                    self.run_func(Gui.instance)
                except Exception as e:
                    print(e)
                    self.frame_data_signal.emit(f'錯誤: {e}')

        except Exception as e:
            print(e)
            self.frame_data_signal.emit(f'錯誤: {e}')

    def stop(self):
        self.is_running = False
        self.rs_camera.pipeline.stop()
        sleep(0.5)
        RealsenseCamera.instance = None
        Gui.instance.toggle_camera_btn.setText('啟動')
        Gui.instance.statusbar.showMessage('深度攝影機已停止！')
        Gui.instance.is_streaming = False
        Gui.instance.toggle_camera_btn.setEnabled(True)
        Gui.instance.toggle_camera_btn.setChecked(False)


class VisionThread(QThread):
    def __init__(self, image, prompt_input, speak):
        super().__init__()
        self.image = image
        self.prompt_input = prompt_input
        self.speak = speak

    def run(self):
        Vision.instance.predict(self.image, self.prompt_input, self.stream_fn, self.speak)

    def stream_fn(self, text):
        Gui.instance.vision_response_label.setText(text)


class Gui(QMainWindow):
    instance = None

    def __init__(self, config, update_frame_func):
        Gui.instance = self
        super(Gui, self).__init__()
        loadUi("gui/main_window.ui", self)
        self.config = config
        self.update_frame_func = update_frame_func

        self.playback_widget.setVisible(False)

        self.setting_btn.clicked.connect(self.setting)

        self.is_zoomed_in = False
        self.image_labels = [self.image_label_1, self.image_label_2, self.image_label_3, self.image_label_4]
        for image_label in self.image_labels:
            image_label.mousePressEvent = self.create_click_handler(image_label)

        self.toggle_camera_btn.clicked.connect(self.toggle_camera)
        self.replay_btn.clicked.connect(self.replay)
        self.prompt_input.returnPressed.connect(self.send_prompt)
        self.green_light_btn.clicked.connect(self.set_green_light)
        self.red_light_btn.clicked.connect(self.set_red_light)
        self.adjust_camera_height_btn.clicked.connect(self.adjust_camera_height)

        self.is_streaming = False
        self.realsense_thread = RealSenseThread(self.config, self.update_frame_func)
        self.realsense_thread.frame_data_signal.connect(self.update_status)

        self.body_height_slider.valueChanged.connect(self.set_body_height)
        self.camera_height_slider.valueChanged.connect(self.set_camera_height)
        self.body_height_slider.setValue(self.config.env["obstacle_detection"]["my_height"])
        self.camera_height_slider.setValue(self.config.env["obstacle_detection"]["camera_height"])

    @pyqtSlot(str)
    def update_status(self, message):
        self.statusbar.showMessage(message)

    def start_camera(self):
        if not self.is_streaming:
            self.realsense_thread.start()

    def stop_camera(self):
        if self.is_streaming:
            self.realsense_thread.stop()

    def create_click_handler(self, label):
        def handle_click(event):
            self.on_image_label_clicked(label)

        return handle_click

    def closeEvent(self, event):
        if self.stop:
            self.stop()

        event.accept()

    def on_image_label_clicked(self, clicked_image_label):
        """當點擊某個 QLabel 時，放大該 QLabel 並隱藏其他 QLabel"""
        if self.is_zoomed_in:
            # 如果已經有一個圖像被放大了，恢復原狀
            for image_label in self.image_labels:
                image_label.show()
            self.is_zoomed_in = False
        else:
            # 隱藏其他 QLabel，僅顯示被點擊的
            for image_label in self.image_labels:
                if image_label != clicked_image_label:
                    image_label.hide()
            self.is_zoomed_in = True

    def display_image(self, img, window=0):
        image_label = self.image_labels[window]

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
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)

    def toggle_camera(self):
        self.toggle_camera_btn.setEnabled(False)
        if self.toggle_camera_btn.isChecked():
            self.toggle_camera_btn.setText('啟動中...')
            self.statusbar.showMessage('正在啟動深度攝影機...')
            threading.Thread(target=self.start_camera).start()
        else:
            self.toggle_camera_btn.setText('停止中...')
            self.statusbar.showMessage('正在停止深度攝影機...')
            threading.Thread(target=self.stop_camera).start()

    def replay(self):
        try:
            if self.replay_btn.isChecked():

                # 使用 QFileDialog 讓用戶選擇 bag 檔案
                file_name, _ = QFileDialog.getOpenFileName(self, "選擇要回放的檔案", "", "Bag Files (*.bag)")

                if not file_name:
                    self.replay_btn.setChecked(False)
                    return
                # 如果用戶選擇了文件，啟動回放模式
                self.statusbar.showMessage(f'正在回放: {file_name}')

                if self.realsense_thread and self.realsense_thread.is_running:
                    self.realsense_thread.stop()
                    self.realsense_thread.wait()

                self.realsense_thread = RealSenseThread(self.config, self.update_frame_func, file=file_name)
                self.realsense_thread.start()
                self.replay_btn.setText('播放回放')
            else:
                if self.realsense_thread and self.realsense_thread.is_running:
                    self.realsense_thread.stop()
                    self.realsense_thread.wait()

                self.realsense_thread = RealSenseThread(self.config, self.update_frame_func)
                self.replay_btn.setText('開始回放')
        except Exception as e:
            print(e)

    def send_prompt(self):
        threading.Thread(target=Vision.instance.predict, args=(
            Image.fromarray(self.color_image),
            self.prompt_input.text(),
            prompt_stream_fn, True)
        ).start()
        # response = Vision.instance.predict(
        #     Image.fromarray(self.color_image),
        #     self.prompt_input.text(),
        #     prompt_stream_fn,
        #     True,
        # )

    # def send_prompt(self):
    #     if Vision.instance is not None:
    #         try:
    #             image = Image.fromarray(self.color_image)
    #             text = self.prompt_input.text()
    #             thread = VisionThread(
    #                 image,
    #                 text,
    #                 True,
    #             )
    #             thread.start()
    #         except Exception as e:
    #             print(e)

    def setting(self):
        if self.setting_btn.isChecked():
            self.setting_btn.setText('關閉設定')
            self.stackedWidget.setCurrentWidget(self.setting_page)
        else:
            self.setting_btn.setText('開啟設定')
            self.stackedWidget.setCurrentWidget(self.main_page)

    def set_green_light(self):
        if self.green_light_btn.isChecked():
            self.red_light_btn.setChecked(False)

    def set_red_light(self):
        if self.red_light_btn.isChecked():
            self.green_light_btn.setChecked(False)

    def set_body_height(self):
        self.body_height_label.setText(f'身高: {self.body_height_slider.value()}')
        self.config.env["obstacle_detection"]["my_height"] = self.body_height_slider.value()

    def set_camera_height(self):
        self.camera_height_label.setText(f'攝影機高度: {self.camera_height_slider.value()}')
        self.config.env["obstacle_detection"]["camera_height"] = self.camera_height_slider.value()

    def adjust_camera_height(self):
        if RealsenseCamera.instance is None:
            self.statusbar.showMessage('請先啟動深度攝影機！')
            return
        _, camera_height = RealsenseCamera.instance.auto_camera_height(self.depth_frame)
        if camera_height is None:
            self.statusbar.showMessage('請先將攝影機視角朝正下方以校準攝影機高度！')
        else:
            self.statusbar.showMessage(f'攝影機高度已調整為: {camera_height}')
            self.camera_height_slider.setValue(camera_height)


def prompt_stream_fn(text):
    Gui.instance.vision_response_label.setText(text)
