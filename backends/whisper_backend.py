from typing import Any

import whisper

from .base import ASRBackend
from utils.dataclasses import ASRResult, ASRSegment, ASRToken


class WhisperBackend(ASRBackend):
    def __init__(self, model_size="base", language="auto"):
        super().__init__(language, model_size)
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str, word_timestamps: bool = True) -> Any:
        """
        Calls the original whisper model and returns the raw dictionary result.
        """
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=word_timestamps
        )
        return result

    def to_asr_result(self, result: Any) -> ASRResult:
        """
        Converts the raw dictionary result from whisper
        into our standard ASRResult object.
        """
        segments = []
        for seg_data in result.get("segments", []):
            words_data = seg_data.get("words", [])
            tokens = [
                ASRToken(
                    start=word.get("start"),
                    end=word.get("end"),
                    token=word.get("word"),
                    probability=word.get("probability")
                ) for word in words_data if word.get("start") is not None
            ]

            segment = ASRSegment(
                start=seg_data.get("start"),
                end=seg_data.get("end"),
                text=seg_data.get("text", ""),
                tokens=tokens
            )
            segments.append(segment)

        return ASRResult(
            text=result.get("text", "").strip(),
            segments=segments,
            language=result.get("language")
        )
