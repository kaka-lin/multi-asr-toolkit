import sherpa_onnx
import numpy as np


class SherpaOnnxBackend:
    def __init__(self, model_configs: dict):
        """
        Initialize the ASR recognizer using Transducer models.

        Args (in model_configs):
            encoder_filename: Path to encoder.onnx
            decoder_filename: Path to decoder.onnx
            joiner_filename: Path to joiner.onnx
            tokens: Path to tokens.txt
            sampling_rate: Expected sample rate of input audio (default: 16000)
            feature_dim: Feature dimension (default: 80 for fbank)
            provider: "cuda" or "cpu" (default: "cuda")
            num_threads: int (default: 2)
            decoding_method: "greedy_search" | "modified_beam_search" | "fast_beam_search"
            max_active_paths: int (beam size for modified_beam_search)
        """
        self.encoder=model_configs["encoder_filename"]
        self.decoder=model_configs["decoder_filename"]
        self.joiner=model_configs["joiner_filename"]
        self.tokens=model_configs["tokens"]
        self.sample_rate=int(model_configs.get("sampling_rate", 16000))
        self.feature_dim=int(model_configs.get("feature_dim", 80))
        self.provider=model_configs.get("provider", "cuda")
        self.num_threads=int(model_configs.get("num_threads", 2))
        self.decoding_method=model_configs.get("decoding_method", "modified_beam_search")
        self.max_active_paths=int(model_configs.get("max_active_paths", 4))

        # Create the recognizer
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=self.encoder,
            decoder=self.decoder,
            joiner=self.joiner,
            tokens=self.tokens,
            sample_rate=self.sample_rate,
            feature_dim=self.feature_dim,
            provider=self.provider,
            num_threads=self.num_threads,
            decoding_method=self.decoding_method,
            max_active_paths=self.max_active_paths,
        )
        self.stream = self.recognizer.create_stream()
        self.expected_sr = self.sample_rate

    def _to_text(self, result) -> str:
        """Handle sherpa-onnx versions where get_result() may return str or an object."""
        if isinstance(result, str):
            return result
        return getattr(result, "text", "") or ""

    # --- Simple inputs ---
    def accept_pcm16_bytes(self, raw_bytes: bytes, sample_rate: int = 16000) -> None:
        """假設 raw_bytes 是 mono s16le，且已對齊（長度為 2 的倍數）"""
        if sample_rate != self.expected_sr:
            raise ValueError(f"Expected {self.expected_sr} Hz, got {sample_rate}")
        f32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        f32 = np.ascontiguousarray(f32, dtype=np.float32)
        self.stream.accept_waveform(sample_rate, f32)

    def accept_waveform_float32(self, f32: np.ndarray, sample_rate: int = 16000) -> None:
        """已經是 [-1,1] 的 float32（單聲道）"""
        if sample_rate != self.expected_sr:
            raise ValueError(f"Expected {self.expected_sr} Hz, got {sample_rate}")
        if f32.dtype != np.float32:
            f32 = f32.astype(np.float32, copy=False)
        f32 = np.ascontiguousarray(f32, dtype=np.float32)
        self.stream.accept_waveform(sample_rate, f32)

    def input_finished(self) -> None:
        self.stream.input_finished()

    def is_endpoint(self) -> bool:
        return self.recognizer.is_endpoint(self.stream)

    # --- Decoding ---
    def decode(self) -> str:
        """Perform decoding and return the current result text."""
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        result = self.recognizer.get_result(self.stream)
        return self._to_text(result)

    def get_partial(self) -> str:
        result = self.recognizer.get_result(self.stream)
        return self._to_text(result)

    # --- Reset ---
    def reset(self) -> str:
        """Reset stream for a new utterance."""
        final_res = self.recognizer.get_result(self.stream)
        final_text = self._to_text(final_res)
        self.recognizer.reset(self.stream)
        return final_text
