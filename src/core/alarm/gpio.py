import threading
import time

# import Jetson.GPIO as GPIO


class Gpio:
    interval_time = float
    _warning_thread = any

    _warning = False
    _print = True
    _buzzer_pin = 18  # 根據實際連接的引腳設定

    warning_message = "warning"

    def __init__(self):
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(self._buzzer_pin, GPIO.OUT)
        pass

    # 發出警告聲
    @staticmethod
    def sound_warning(sound_time=0.5):
        # GPIO.output(self._buzzer_pin, GPIO.HIGH)
        time.sleep(sound_time)  # 聲音持續時間
        # GPIO.output(self._buzzer_pin, GPIO.LOW)

    def start_warning(self, interval_time, warning_message=None):
        self.interval_time = interval_time
        if warning_message:
            self.warning_message = warning_message

        if self._warning:
            return
        self._warning = True

        # 建立並執行子執行緒
        self._warning_thread = threading.Thread(target=self.keep_warning)
        self._warning_thread.start()

    def keep_warning(self):
        while self._warning:
            # self.sound_warning(self.interval_time)
            if self._print:
                print(self.warning_message)
            time.sleep(self.interval_time)

    def stop_warning(self):
        if not self._warning:
            return
        self._warning = False
        if self._print:
            print("stop warning")

    def cleanup(self):
        self.stop_warning()
        # GPIO.cleanup()
