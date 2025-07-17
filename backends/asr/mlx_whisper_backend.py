from typing import Any

import mlx_whisper

from .base import ASRBackend
from utils.dataclasses import ASRResult, ASRSegment, ASRToken


class MLXWhisperBackend(ASRBackend):
    def __init__(self, model_size="base", language="auto"):
        super().__init__(language, model_size)

    def transcribe(self, audio_path: str, word_timestamps: bool = True) -> Any:
        """
        Calls mlx_whisper.transcribe and returns the raw dictionary.
        """
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.model_size,
            word_timestamps=word_timestamps,
        )
        return result

    def to_asr_result(self, result: Any) -> ASRResult:
        """
        Converts the raw dictionary result from mlx-whisper
        into our standard ASRResult object.
        """
        segments = []
        for seg_data in result.get("segments", []):
            # Handle absence of 'words' key, as it might not be present.
            words_data = seg_data.get("words", [])
            tokens = [
                ASRToken(
                    start=word.get("start"),
                    end=word.get("end"),
                    token=word.get("word"),
                    # Add default probability
                    probability=word.get("probability", 0.0)
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
