from typing import Any

import torch
from faster_whisper import WhisperModel

from .base import ASRBackend
from utils.dataclasses import ASRToken, ASRSegment, ASRResult


class FasterWhisperBackend(ASRBackend):
    def __init__(self,
                 model_size="base",
                 device="cpu",
                 compute_type="float32",
                 language="auto"):
        super().__init__(language, model_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(self, audio_path: str, word_timestamps: bool = True) -> Any:
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=5,
            word_timestamps=word_timestamps
        )

        return list(segments), info

    def to_asr_result(self, result: Any) -> ASRResult:
        raw_segments, info = result
        language = info.language

        segments = []
        full_text = ""
        for seg in raw_segments:
            if hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.9:
                continue

            tokens = []
            if hasattr(seg, 'words'):
                tokens = [
                    ASRToken(
                        start=word.start,
                        end=word.end,
                        token=word.word,
                        probability=word.probability
                    ) for word in seg.words
                ]

            segment = ASRSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                tokens=tokens
            )
            segments.append(segment)
            full_text += seg.text

        return ASRResult(
            text=full_text.strip(),
            segments=segments,
            language=language
        )
