import io
import queue
import re
import threading
import wave
from typing import Any

import pyaudio
import requests

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

class TTS:
    instance = None

    def __init__(self, alarm_env: Any):
        TTS.instance = self

        self.get_audio_queue = queue.Queue()  # 創建一個獲取音頻隊列
        self.done_queue = queue.Queue()  # 創建一個完成隊列
        self.getting_audio = False  # 是否正在獲取音頻
        self.playing = False  # 是否正在播放音頻

        self.alarm_env = alarm_env
        self.pyAudio = pyaudio.PyAudio()

    def __call__(self, message):
        playback_event = threading.Event()
        self.get_audio(message, playback_event)
        playback_event.wait()  # 等待當前語音播放完成

    def process_get_audio_queue(self):
        if self.getting_audio is True:
            return
        self.getting_audio = True
        while not self.get_audio_queue.empty():
            data, playback_event = self.get_audio_queue.get()
            try_times = 0  # 嘗試次數
            while 1:
                try:
                    self.response = requests.post(self.alarm_env["tts_host"], json=data, timeout=20)
                except requests.exceptions.RequestException as e:
                    print(f"錯誤：{e}")
                    try_times += 1
                    if try_times == 3:
                        self.cancel()
                        self.getting_audio = False
                        return
                    continue

                if self.getting_audio is False:
                    break

                audio_data = self.response.content
                # 啟動音頻播放執行緒
                threading.Thread(target=self.play_audio_thread, args=(audio_data, playback_event)).start()
                break
        self.getting_audio = False

    def play_audio_thread(self, audio_data, playback_event):
        wf = wave.open(io.BytesIO(audio_data), 'rb')
        self.done_queue.put((wf, playback_event))
        threading.Thread(target=self.process_done_queue).start()

    @staticmethod
    def split(todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break
            if todo_text[i_split_head] in splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    def cut(self, inp):
        # 定義要保留的字元範圍：數字、英文字母、中文、日文
        pattern = re.compile(r'^[^0-9a-zA-Z\u4e00-\u9fa5\u3040-\u30FF]+|[^0-9a-zA-Z\u4e00-\u9fa5\u3040-\u30FF]+$')
        # 使用正規表示式刪除開頭和結尾處的非指定字元範圍的字符
        inp = re.sub(pattern, '', inp)
        if not re.search(r'[^\w\s]', inp[-1]):
            inp += '。'
        punctuation = r'[,.;?!、，。？！;：\n]'
        opt = re.split(f'({punctuation})', inp)
        sentences = []
        for i in range(0, len(opt), 2):
            if i + 1 < len(opt):
                symbol = opt[i + 1]
                sentence = opt[i] + symbol  # 將文字與標點符號組合在一起
                if len(sentence.strip()) == 1 and re.search(punctuation, symbol):  # 如果只有一个标点符号则跳过
                    continue
                sentences.append(sentence)

        return sentences

    def process_done_queue(self):
        if self.done_queue.empty() or self.playing is True:
            return

        self.playing = True
        while not self.done_queue.empty():
            chunk = 1024
            wf, playback_event = self.done_queue.get()

            stream = self.pyAudio.open(format=self.pyAudio.get_format_from_width(wf.getsampwidth()),
                                       channels=wf.getnchannels(),
                                       rate=wf.getframerate(),
                                       output=True)

            data = wf.readframes(chunk)
            while data and self.playing is True:
                stream.write(data)
                data = wf.readframes(chunk)

            stream.stop_stream()
            stream.close()

            playback_event.set()  # 設置播放完成事件

        self.playing = False

    def cancel(self):
        self.get_audio_queue.queue.clear()
        self.done_queue.queue.clear()
        self.response = None
        self.getting_audio = False
        self.playing = False

    def get_audio(self, message, playback_event):
        cut_texts = self.cut(message)

        for t in cut_texts:
            if t.strip():
                data = {
                    "refer_wav_path": self.alarm_env['ref_wav_path'],
                    "prompt_text": self.alarm_env['prompt_text'],
                    "prompt_language": self.alarm_env['prompt_language'],
                    "text": t,
                    "text_language": self.alarm_env['text_language']
                }
                event = playback_event if t == cut_texts[-1] else threading.Event()
                self.get_audio_queue.put((data, event))
                threading.Thread(target=self.process_get_audio_queue).start()

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
