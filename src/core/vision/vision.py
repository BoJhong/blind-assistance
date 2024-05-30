from threading import Thread

import cv2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.utils.opencv import draw_multiline_text_with_border

class Vision:
    instance = None

    def __init__(self, config):
        Vision.instance = self

        # self.config = config.env["vision"]

        model_id = "vikhyatk/moondream2"
        revision = "2024-05-20"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16  # CPU doesn't support float16

        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            revision=revision,
            torch_dtype=dtype
        ).to(device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def predict(self, image, prompt="Describe the image in detail.", stream=None, speak=False):
        enc_image = self.model.encode_image(image)
        response = self.answer_question(enc_image, prompt)
        result = ''
        for txt in response:
            if txt.strip() == '':
                continue
            result = txt
            if stream is not None:
                stream(txt)

        return result


    def answer_question(self, enc_image, prompt):
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        thread = Thread(
            target=self.model.answer_question,
            kwargs={
                "image_embeds": enc_image,
                "question": prompt,
                "tokenizer": self.tokenizer,
                "streamer": streamer,
            },
        )
        thread.start()

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
