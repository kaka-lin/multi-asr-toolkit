import platform

from .asr.transformers_backend import TransformersBackend
from .asr.speech_recognition_backend import SpeechRecognitionBackend
from .asr.whisper_backend import WhisperBackend
from .asr.faster_whisper_backend import FasterWhisperBackend


def get_asr_backend(name: str, **kwargs):
    name = name.lower()
    if name == "transformers":
        return TransformersBackend(**kwargs)
    elif name == "speech-recognition":
        return SpeechRecognitionBackend(**kwargs)
    elif name == "whisper":
        return WhisperBackend(**kwargs)
    elif name == "faster-whisper":
        return FasterWhisperBackend(**kwargs)
    elif name == "mlx-whisper" and platform.system() == "Darwin":
        from .asr.mlx_whisper_backend import MLXWhisperBackend
        return MLXWhisperBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")
