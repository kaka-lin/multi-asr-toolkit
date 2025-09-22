from typing import Any

from transformers import pipeline

from .base import ASRBackend
from utils.dataclasses import ASRResult, ASRSegment


class TransformersBackend(ASRBackend):
    def __init__(
        self,
        model_size="openai/whisper-small",
        device=-1,
        language="auto"
    ):
        super().__init__(language)
        self.asr = pipeline(
            task="automatic-speech-recognition",
            model=model_size,
            device=device,   # -1 for CPU, 0 for GPU
            framework="pt",  # "tf" for TensorFlow, "pt" for PyTorch
        )

    def transcribe(self, audio_path: str, word_timestamps: bool = True) -> Any:
        """
        Calls the transformers pipeline and returns the raw dictionary result.
        """
        result = self.asr(
            audio_path,
            generate_kwargs={"language": self.language},
            return_timestamps="word" if word_timestamps else True
        )
        return result

    def to_asr_result(self, result: Any) -> ASRResult:
        """
        Converts the raw dictionary result from transformers
        into our standard ASRResult object.
        """
        segments = []
        for chunk in result.get("chunks", []):
            # Transformers pipeline doesn't provide word-level timestamps in the same way as other backends,
            # so the tokens list will be empty. The "chunk" itself represents either a word or a segment.
            segment = ASRSegment(
                start=chunk["timestamp"][0],
                end=chunk["timestamp"][1],
                text=chunk["text"],
                tokens=[]  # No detailed word-level tokens from this backend in our current structure
            )
            segments.append(segment)

        return ASRResult(
            text=result.get("text", "").strip(),
            segments=segments,
            language=self.language  # Transformers doesn't return language
        )
