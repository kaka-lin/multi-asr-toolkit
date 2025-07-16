from typing import Any

import speech_recognition as sr

from .base import ASRBackend
from utils.dataclasses import ASRResult
from utils.mp3_utils import convert_mp3_to_wav


class SpeechRecognitionBackend(ASRBackend):
    def __init__(self, language="zh"):
        super().__init__(language=language)
        self.recognizer = sr.Recognizer()

    def mp3_to_wav(self, audio_path: str) -> str:
        if audio_path.endswith(".mp3"):
            wav_path = audio_path.replace(".mp3", ".wav")
            convert_mp3_to_wav(audio_path, wav_path)
            return wav_path
        return audio_path

    def transcribe(self, audio_path: str, word_timestamps: bool = True) -> Any:
        """
        Transcribes using Google Speech Recognition and returns the raw text.
        """
        # This backend does not support word timestamps.
        audio_path = self.mp3_to_wav(audio_path)
        with sr.AudioFile(audio_path) as source:
            audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(
                    audio, language=self.language
                )
                return text
            except sr.UnknownValueError:
                return "無法辨識語音內容"
            except sr.RequestError as e:
                return f"無法連接 Google 語音服務：{e}"

    def to_asr_result(self, result: Any) -> ASRResult:
        """
        Converts the raw text result into our standard ASRResult object.
        This backend does not support timestamps, so the segments list will
        be empty.
        """
        return ASRResult(
            text=result,
            segments=[],
            language=self.language
        )
