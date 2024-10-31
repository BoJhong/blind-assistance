import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd

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
        self.disable = False
        self.speaking_count = 0

        # self.pwm = buzzer.setup()

        if self.alarm_env["tts_enable"]:
            self.tts = TTS(self.alarm_env)

    def __exit__(self):
        self.cleanup()

    def play_sound(self, frequency: int = 2500, duration: float = 1):
        def fn():
            # self.pwm.start(frequency)
            self.bip(freq=frequency, dur=duration, volume=0.1)
            time.sleep(duration)  # 聲音持續時間
            # pwm.stop()

        threading.Thread(target=fn).start()

    def bip(self, freq, dur, a=0, d=0, s=1, r=0, volume=1.0):
        t = np.arange(0, dur, 1 / 44100)
        env = np.interp(t, [0, a, (a + d), dur - r, dur], [0, 1, s, s, 0])
        sound = np.sin(2 * np.pi * freq * t) * env * volume
        sd.play(sound, samplerate=44100)

    def speak(self, message: str):
        if self.alarm_env["print"]:
            logging.info(message)

        if self.tts is not None and TTS.exec_status:
            self.speaking_count += 1
            self.tts(message)
            self.speaking_count -= 1

    def speak_async(self, message: str):
        threading.Thread(target=self.speak, args=(message,)).start()

    def start(self, message: str = "警報！", duration: float = 1, frequency: int = 2500):
        if self.disable:
            return

        self.duration = duration
        self.frequency = frequency
        self.message = message

        if Alarm.exec_status:
            return

        Alarm.exec_status = True
        if not self.loop:
            threading.Thread(target=self._exec_loop).start()

    def _exec_loop(self):
        self.loop = True
        while Alarm.exec_status and not self.disable:
            if self.alarm_env["print"] and self.message is not None:
                logging.info(self.message)

            self.bip(freq=self.frequency,
                     dur=self.duration,
                     volume=0.1)

            next_loop = False

            for _ in range(2):
                for i in range(int(self.duration * 10)):
                    time.sleep(0.1)
                    if i > int(self.duration * 10):
                        print(i)
                        print(self.duration)
                        next_loop = True
                        break
                if next_loop:
                    break

            if next_loop:
                continue

        self.loop = False

        if self.disable:
            self.stop()

    def stop(self):
        if not Alarm.exec_status:
            return
        Alarm.exec_status = False
        if self.alarm_env["print"]:
            print("停止警報")

    def cleanup(self):
        Alarm.exec_status = False
        # pwm.stop()
        # buzzer.cleanup()

    @staticmethod
    def note_to_frequency(note):
        return pitches[note]

    def play_notes(self, notes: list, duration: float = 0.5):
        for note in notes:
            self.play_sound(self.note_to_frequency(note), duration)
            time.sleep(duration)
