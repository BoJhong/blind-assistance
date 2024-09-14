import threading
import time
from typing import Any

import winsound

from .pitches import pitches
from .tts import TTS
import logging

class Alarm:
    instance = None
    exec_status = False

    def __init__(self, config: Any):
        Alarm.instance = self

        self.alarm_env = config.env["alarm"]
        self.message = "警報！"
        self.duration = 1
        self.frequency = 2500
        self.loop = False

        # self.pwm = buzzer.setup()

        if self.alarm_env["tts_enable"]:
            self.tts = TTS(self.alarm_env)

    def __exit__(self):
        self.cleanup()

    def play_sound(self, frequency: int = 2500, duration: float = 1):
        def fn():
            # self.pwm.start(frequency)
            if self.alarm_env["windows_sound"]:
                threading.Thread(
                    target=winsound.Beep, args=(frequency, int(duration * 1000))
                ).start()
            time.sleep(duration)  # 聲音持續時間
            # pwm.stop()

        threading.Thread(target=fn).start()

    def speak(self, message: str):
        if self.alarm_env["print"]:
            logging.info(message)

        if self.tts is not None:
            self.tts(message)

    def speak_async(self, message: str):
        threading.Thread(target=self.speak, args=(message,)).start()

    def start(self, message: str = "警報！", duration: float = 1, frequency: int = 2500):
        self.duration = duration
        self.frequency = frequency
        self.message = message

        if self.exec_status:
            return
        self.exec_status = True

        # 建立並執行子執行緒
        if not self.loop:
            threading.Thread(target=self._exec_loop).start()

    def _exec_loop(self):
        self.loop = True
        while self.exec_status:
            self.play_sound(self.frequency, self.duration)
            if self.alarm_env["print"] and self.message is not None:
                logging.info(self.message)
            time.sleep(self.duration * 2)
        self.loop = False

    def stop(self):
        if not self.exec_status:
            return
        self.exec_status = False
        if self.alarm_env["print"]:
            print("停止警報")

    def cleanup(self):
        self.exec_status = False
        # pwm.stop()
        # buzzer.cleanup()

    @staticmethod
    def note_to_frequency(note):
        return pitches[note]

    def play_notes(self, notes: list, duration: float = 0.5):
        for note in notes:
            self.play_sound(self.note_to_frequency(note), duration)
            time.sleep(duration)
