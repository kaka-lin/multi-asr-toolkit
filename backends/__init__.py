import os
import sys
import platform

proj_root = sys.path[0]
model_dir = os.path.join(proj_root, "models", "korean_zipformer")


from .asr.transformers_backend import TransformersBackend
from .asr.speech_recognition_backend import SpeechRecognitionBackend
from .asr.whisper_backend import WhisperBackend
from .asr.faster_whisper_backend import FasterWhisperBackend
from .asr.sherpa_onnx_beckend import SherpaOnnxBackend


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
    elif name == "sherpa-onnx":
        enc = os.path.join(model_dir, "encoder-epoch-99-avg-1.int8.onnx")
        dec = os.path.join(model_dir, "decoder-epoch-99-avg-1.int8.onnx")
        joi = os.path.join(model_dir, "joiner-epoch-99-avg-1.int8.onnx")
        tok = os.path.join(model_dir, "tokens.txt")

        # Zipformer-Transducer
        cfg = {
            "encoder_filename": enc,
            "decoder_filename": dec,
            "joiner_filename":  joi,
            "tokens":           tok,
            "sampling_rate": 16000,
            "feature_dim": 80,
            "provider": "cpu",
            "num_threads": 2,
            "decoding_method": "modified_beam_search",
            "max_active_paths": 4,
        }

        return SherpaOnnxBackend(cfg)
    else:
        raise ValueError(f"Unknown backend: {name}")
