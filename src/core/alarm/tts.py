import io
import threading
import wave
from typing import Any

import pyaudio
import requests


class TTS:
    def __init__(self, alarm_env: Any):
        self.alarm_env = alarm_env
        self.pyAudio = pyaudio.PyAudio()

    def __call__(self, message):
        self.get_audio(message)

    def get_audio(self, message):
        data = {
            "refer_wav_path": self.alarm_env['ref_wav_path'],
            "prompt_text": self.alarm_env['prompt_text'],
            "prompt_language": self.alarm_env['prompt_language'],
            "text": message,
            "text_language": self.alarm_env['text_language']
        }

        try:
            self.response = requests.post(self.alarm_env["tts_host"], json=data, timeout=3)
        except requests.exceptions.RequestException as e:
            print(f"錯誤：{e}")
            return

        audio_data = self.response.content

        self.play_audio(audio_data,)

    def play_audio(self, audio_data):
        chunk = 1024
        wf = wave.open(io.BytesIO(audio_data), 'rb')

        stream = self.pyAudio.open(format=self.pyAudio.get_format_from_width(wf.getsampwidth()),
                                   channels=wf.getnchannels(),
                                   rate=wf.getframerate(),
                                   output=True)

        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
