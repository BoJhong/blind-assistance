import os
import threading
from time import sleep

import cv2
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.uic import loadUi

import gui.compiled_resources  # type: ignore
from src.core.alarm.alarm import Alarm
from src.core.alarm.tts import TTS
from src.core.realsense_camera.realsense_camera import RealsenseCamera
from src.core.speech_recognition.speech_recognition import SpeechRecognition
from src.core.vision.vision import Vision
import time

class SpeechThread(QThread):
    frame_data_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.is_running = False

    def run(self):
        try:
            self.is_running = True
            speech_recognition = SpeechRecognition()

            while self.is_running:
                try:
                    text = speech_recognition()
                    if text != '':
                        print(text)
                        Gui.instance.prompt_input.setText(text)
                except Exception as e:
                    pass
        except Exception as e:
            self.frame_data_signal.emit(f'錯誤: {e}')

    def stop(self):
        self.is_running = False


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

            exposure = Gui.instance.exposure_slider.value()
            if exposure == 0:
                Gui.instance.exposure_label.setText(f'曝光: 自動曝光')
                self.rs_camera.enable_auto_exposure()
            else:
                Gui.instance.exposure_label.setText(f'曝光: {exposure}')
                self.rs_camera.set_exposure(Gui.instance.exposure_slider.value())

            while self.is_running:
                try:
                    self.run_func(Gui.instance)
                except Exception as e:
                    print(f"錯誤: {e}")
                    self.frame_data_signal.emit(f'錯誤: {e}')

                    Alarm.instance.stop()

                    # 屎山
                    while self.is_running:
                        try:
                            # 嘗試重新啟動攝影機
                            self.rs_camera.profile = self.rs_camera.pipeline.start(self.rs_camera.config)
                            print("攝影機重新連線成功")
                            reconnected = True
                            break
                        except Exception:
                            time.sleep(1)

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
        Alarm.instance.stop()

class VideoCaptureThread(QThread):
    frame_data_signal = pyqtSignal(str)

    def __init__(self, config, run_func, file=None):
        super().__init__()
        self.config = config
        self.run_func = run_func
        self.file = file
        self.is_running = False

        if self.file is None:
            self.file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'resources',
                                     self.config.env["config"]["video"])

    def run(self):
        try:
            if not os.path.exists(self.file):
                Gui.instance.statusbar.showMessage(f"影片檔案不存在：{self.file}")
                Gui.instance.toggle_camera_btn.setText('啟動')
                Gui.instance.toggle_camera_btn.setEnabled(True)
                Gui.instance.toggle_camera_btn.setChecked(False)
                return

            self.is_running = True
            self.cap = cv2.VideoCapture(self.file)

            Gui.instance.toggle_camera_btn.setText('停止')
            Gui.instance.statusbar.showMessage(f"正在播放 {self.config.env['config']['video']}")
            Gui.instance.is_streaming = True
            Gui.instance.toggle_camera_btn.setEnabled(True)
            Gui.instance.toggle_camera_btn.setChecked(True)

            while self.is_running and self.cap.isOpened():
                try:
                    self.run_func(self.cap)
                except Exception as e:
                    print(e)
                    self.frame_data_signal.emit(f'錯誤: {e}')

        except Exception as e:
            print(e)
            self.frame_data_signal.emit(f'錯誤: {e}')

    def stop(self):
        self.is_running = False
        Gui.instance.toggle_camera_btn.setText('啟動')
        Gui.instance.statusbar.showMessage('已停止播放影片！')
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

    def __init__(self, config, update_frame_func, is_video_capture=False):
        Gui.instance = self
        super(Gui, self).__init__()
        loadUi("gui/main_window.ui", self)
        self.config = config
        self.update_frame_func = update_frame_func
        self.is_video_capture = is_video_capture

        self.playback_widget.setVisible(False)
        self.red_light_btn.setVisible(False)
        self.green_light_btn.setVisible(False)

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
        self.chat_btn.clicked.connect(self.toggle_chat)
        self.speech_btn.clicked.connect(self.toggle_speech)
        self.alarm_btn.clicked.connect(self.toggle_alarm)
        self.tts_btn.clicked.connect(self.toggle_tts)

        self.chat_btn.setChecked(False)
        self.chat_widget.setVisible(False)
        self.chat_btn.setText('開啟對話欄')

        self.is_streaming = False
        if is_video_capture:
            self.video_capture_thread = VideoCaptureThread(self.config, self.update_frame_func)
            self.video_capture_thread.frame_data_signal.connect(self.update_status)
        else:
            self.realsense_thread = RealSenseThread(self.config, self.update_frame_func)
            self.realsense_thread.frame_data_signal.connect(self.update_status)

        self.body_height_slider.valueChanged.connect(self.set_body_height)
        self.camera_height_slider.valueChanged.connect(self.set_camera_height)
        self.exposure_slider.valueChanged.connect(self.set_exposure)
        self.lateral_distance_threshold_slider.valueChanged.connect(self.set_lateral_distance_threshold)
        self.detect_point_size_slider.valueChanged.connect(self.set_detect_point_size)
        self.body_height_slider.setValue(self.config.env["obstacle_detection"]["my_height"])
        self.camera_height_slider.setValue(self.config.env["obstacle_detection"]["camera_height"])
        self.exposure_slider.setValue(self.config.env["realsense"]["exposure"])
        self.lateral_distance_threshold_slider.setValue(self.config.env["obstacle_detection"]["lateral_distance_threshold"])
        self.detect_point_size_slider.setValue(self.config.env["obstacle_detection"]["detect_point_size"])


    @pyqtSlot(str)
    def update_status(self, message):
        self.statusbar.showMessage(message)

    def start_camera(self):
        if not self.is_streaming:
            if self.is_video_capture:
                self.video_capture_thread.start()
            else:
                self.realsense_thread.start()

    def stop_camera(self):
        if self.is_streaming:
            if self.is_video_capture:
                self.video_capture_thread.stop()
            else:
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
                if self.is_video_capture:
                    file_name, _ = QFileDialog.getOpenFileName(self, "選擇要回放的檔案", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
                else:
                    file_name, _ = QFileDialog.getOpenFileName(self, "選擇要回放的檔案", "", "Bag Files (*.bag)")

                if not file_name:
                    self.replay_btn.setChecked(False)
                    return
                # 如果用戶選擇了文件，啟動回放模式
                self.statusbar.showMessage(f'正在回放: {file_name}')

                if self.is_video_capture:
                    if self.video_capture_thread and self.video_capture_thread.is_running:
                        self.video_capture_thread.stop()
                        self.video_capture_thread.wait()

                    self.video_capture_thread = VideoCaptureThread(self.config, self.update_frame_func, file=file_name)
                    self.video_capture_thread.start()
                else:
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

    def update_crosswalk_signal_status(self, status):
        if status == 0:
            self.alert_label_3.setText("行人號誌狀態: 無")
        elif status == 1:
            self.alert_label_3.setText("行人號誌狀態: 紅燈")
        elif status == 2:
            self.alert_label_3.setText("行人號誌狀態: 綠燈")

    def send_prompt(self):
        threading.Thread(target=Vision.instance.predict, args=(
            Image.fromarray(self.color_image),
            self.prompt_input.text(),
            prompt_stream_fn,
            True,
            True,)
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
        self.body_height_label.setText(f'身高: {self.body_height_slider.value()}cm')
        self.config.env["obstacle_detection"]["my_height"] = self.body_height_slider.value()

    def set_camera_height(self):
        self.camera_height_label.setText(f'攝影機高度: {self.camera_height_slider.value()}cm')
        self.config.env["obstacle_detection"]["camera_height"] = self.camera_height_slider.value()

    def set_exposure(self):
        exposure = self.exposure_slider.value()
        if exposure == 0:
            self.exposure_label.setText(f'曝光: 自動曝光')
            if RealsenseCamera.instance is not None:
                RealsenseCamera.instance.enable_auto_exposure()
        else:
            self.exposure_label.setText(f'曝光: {self.exposure_slider.value()}')
            if RealsenseCamera.instance is not None:
                RealsenseCamera.instance.disable_auto_exposure()
                RealsenseCamera.instance.set_exposure(self.exposure_slider.value())

    def set_lateral_distance_threshold(self):
        self.config.env["obstacle_detection"]["lateral_distance_threshold"] = self.lateral_distance_threshold_slider.value()
        self.lateral_distance_threshold_label.setText(f'障礙物橫向距離閥值: {self.lateral_distance_threshold_slider.value()}mm')

    def set_detect_point_size(self):
        self.config.env["obstacle_detection"]["detect_point_size"] = self.detect_point_size_slider.value()
        self.detect_point_size_label.setText(f'偵測點大小: {self.detect_point_size_slider.value()}')

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

    def toggle_chat(self):
        if self.chat_btn.isChecked():
            self.chat_btn.setText('關閉對話欄')
            self.chat_widget.setVisible(True)
        else:
            self.chat_btn.setText('開啟對話欄')
            self.chat_widget.setVisible(False)

    def toggle_speech(self):
        if self.speech_btn.isChecked():
            self.speech_thread = SpeechThread()
            self.speech_thread.start()
            self.speech_thread.frame_data_signal.connect(self.update_status)
        else:
            if self.speech_thread and self.speech_thread.is_running:
                self.speech_thread.stop()

    def toggle_alarm(self):
        if self.alarm_btn.isChecked():
            self.alarm_btn.setText('關閉警示音')
            Alarm.instance.disable = False
        else:
            self.alarm_btn.setText('開啟警示音')
            Alarm.instance.disable = True

    def toggle_tts(self):
        if self.tts_btn.isChecked():
            self.tts_btn.setText('關閉語音播報')
        else:
            self.tts_btn.setText('開啟語音播報')

        if TTS.instance is not None:
            TTS.exec_status = self.tts_btn.isChecked()
        self.config.env["alarm"]["tts_enable"] = self.tts_btn.isChecked()




def prompt_stream_fn(text):
    Gui.instance.vision_response_label.setText(text)
