import threading
import time

import winsound

from src.core import TOMLConfig
from .pitches import pitches


class Alarm:
    instance = None
    exec_status = False

    def __init__(self):
        Alarm.instance = self

        self.alarm_env = TOMLConfig.instance.env["alarm"]

        self.message = "警報！"
        self.duration = 1
        self.frequency = 2500
        self.loop = False

        # self.pwm = buzzer.setup()

    def __exit__(self):
        self.cleanup()

    def play_sound(self, frequency: int = 2500, duration: float = 1):
        def _():
            # self.pwm.start(frequency)
            if self.alarm_env["windows_sound"]:
                threading.Thread(
                    target=winsound.Beep, args=(frequency, int(duration * 1000))
                ).start()
            time.sleep(duration)  # 聲音持續時間
            # pwm.stop()

        threading.Thread(target=_).start()

    def start(self, message: str = None, duration: float = 1):
        self.duration = duration
        if message:
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
            if self.alarm_env["print"]:
                print(self.message)
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
