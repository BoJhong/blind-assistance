import threading
from threading import Thread

import cv2
import torch
from deep_translator import GoogleTranslator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from src.core.alarm.tts import TTS
from src.utils.opencv import draw_multiline_text_with_border

class Vision:
    instance = None

    def __init__(self, config):
        Vision.instance = self

        # self.config = config.env["vision"]

        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("推理裝置: {}".format(device))
        dtype = torch.float32 if device == "cpu" else torch.float16  # CPU doesn't support float16
        print("精度: {}".format(dtype))

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
            torch_dtype=dtype,
            # low_cpu_mem_usage=True,
            # use_safetensors=True,
            # attn_implementation="flash_attention_2",
            device_map=device,
        ).to(device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        print(f"Model loaded: {model_id}")

    def predict(self, image, prompt="Describe this image.", stream=None, speak=False, translate=False):
        if translate:
            prompt = GoogleTranslator(target='en').translate(prompt)
        # print(prompt)

        enc_image = self.model.encode_image(image)
        response = self.answer_question(enc_image, prompt)
        result = ''
        for txt in response:
            if txt.strip() == '':
                continue

            result = txt
            if stream is not None:
                stream(txt)

        if translate:
            print(result)
            result = GoogleTranslator(source='en', target='zh-TW').translate(result)

        if stream is not None:
            stream(result)

        print(result)

        if speak and TTS.instance is not None:
            threading.Thread(target=TTS.instance, args=(result,)).start()

        return result


    def answer_question(self, enc_image, prompt):
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        # 啟動生成，使用 streamer 作為 callback
        generation_kwargs = {
            "image_embeds": enc_image,
            "question": prompt,
            "tokenizer": self.tokenizer,
            "streamer": streamer
        }
        generation_thread = threading.Thread(target=self.model.answer_question, kwargs=generation_kwargs)
        generation_thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

    def draw(self, color_image, response):
        image = color_image.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 2
        border_color = (0, 0, 0)  # 黑色
        border_thickness = thickness + 2

        max_width = image.shape[1]
        position = (0, 30)  # 距離頂部 30 像素
        draw_multiline_text_with_border(image, response, position, font, font_scale, font_color, thickness, border_color, border_thickness, max_width)
        return image
