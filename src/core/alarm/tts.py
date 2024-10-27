import io
import queue
import re
import threading
import wave
from typing import Any

import pyaudio
import pyttsx3
import requests

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

class TTS:
    instance = None
    exec_status = True

    def __init__(self, alarm_env: Any):
        TTS.instance = self

        self.alarm_env = alarm_env
        self.pyAudio = pyaudio.PyAudio()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def __call__(self, message):
        if not TTS.exec_status:
            return

        if self.alarm_env["tts_mode"] == "ai":
            threading.Thread(target=self.get_audio, args=(message,)).start()
        else:
            self.engine.say(message)
            self.engine.runAndWait()

    def get_audio(self, message):
        data = {
            "refer_wav_path": self.alarm_env['ref_wav_path'],
            "prompt_text": self.alarm_env['prompt_text'],
            "prompt_language": self.alarm_env['prompt_language'],
            "text": message,
            "text_language": self.alarm_env['text_language']
        }
        try:
            response = requests.post(self.alarm_env["tts_host"], json=data)
            audio_data = response.content
            self.play_audio(audio_data)
        except Exception as e:
            print(e)

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
