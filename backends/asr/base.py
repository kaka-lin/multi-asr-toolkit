from typing import Any
from abc import ABC, abstractmethod

from utils.dataclasses import ASRResult


class ASRBackend(ABC):
    def __init__(self, language="auto", model_size=None):
        self.language = None if language == "auto" else language
        self.model_size = model_size

    @abstractmethod
    def transcribe(self, audio_path: str, word_timestamps: bool = True) -> Any:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def to_asr_result(self, result: Any) -> ASRResult:
        raise NotImplementedError("must be implemented in the child class")
