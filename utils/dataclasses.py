from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ASRToken:
    """Represents a single token from an ASR model."""
    start: float
    end: float
    token: str
    probability: float


@dataclass
class ASRSegment:
    """Represents a segment of transcribed audio."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    tokens: List[ASRToken] = field(default_factory=list)


@dataclass
class ASRResult:
    """Represents the final result from an ASR backend."""
    text: str
    segments: List[ASRSegment]
    language: Optional[str] = None
