import speech_recognition

class SpeechRecognition:
    instance = None

    def __init__(self) -> None:
        SpeechRecognition.instance = self

    def __call__(self, timeout=1) -> str:
        r = speech_recognition.Recognizer()
        with speech_recognition.Microphone() as source:
            audio = r.listen(source, timeout=timeout)

        try:
            return r.recognize_google(audio, language='zh-tw')
        except speech_recognition.UnknownValueError:
            return ''
        except speech_recognition.RequestError as e:
            # return f'錯誤: {e}'
            return ''
