import os
import time
import wave
from typing import Tuple, Optional, List

from backends import get_asr_backend
from utils.subtitle_generator import save_transcription_results
from utils.model import resolve_model_name


def transcribe_audio(audio_input, backend, language, model_size, word_timestamps=True):
    """
    使用指定的後端對音訊檔案進行離線轉錄。
    """
    audio = None
    if isinstance(audio_input, str):
        audio = audio_input
    else:
        raise ValueError("無效的音訊輸入類型，預期為檔案路徑。")

    model_name = resolve_model_name(backend, model_size)
    kwargs = {"language": language}
    if model_name:
        kwargs["model_size"] = model_name
    asr = get_asr_backend(backend, **kwargs)
    result = asr.transcribe(audio, word_timestamps=word_timestamps)

    asr_result = asr.to_asr_result(result)
    subtitle_results = save_transcription_results(asr_result, audio)

    return asr_result.text, subtitle_results


class StreamingTranscriber:
    """
    一個處理音訊流的控制器。

    使用純串流 ASR 引擎（如 Sherpa-ONNX）進行即時語音辨識，
    同時根據語音的停頓或標點符號，自動將音訊流切分成段落並存檔。

    主要功能：
    1. 接收連續的音訊塊 (audio chunks)。
    2. 將音訊塊餵給底層的 ASR 引擎進行即時辨識。
    3. 管理初步 (partial) 和最終 (finalized) 的辨識結果。
    4. 根據設定的規則（如句子長度、停頓時間）決定切分點。
    5. 當一個句子結束時，將該句對應的音訊儲存為 WAV 檔案。
    """
    puncts = {".", "?", "!", "。", "？", "！", ",", "，", ";", "；"}
    FinalizedSegment = Optional[Tuple[float, float, str, str]]

    def __init__(
        self,
        streaming_asr,
        sample_rate: int = 16000,
        chunk_sec: float = 1.2,
        overlap_sec: float = 0.25,
        max_utt_sec: float = 6.0,
        stall_ms: int = 900,
        partial_min_interval: float = 0.25,
        min_utt_sec: float = 2.0,
    ):
        """
        初始化 StreamingTranscriber。

        Args:
            streaming_asr: 一個符合 ASR 協議的串流辨識引擎實例。
            sample_rate (int): 音訊取樣率。
            chunk_sec (float): 內部處理音訊塊的長度（秒）。
            overlap_sec (float): 內部處理音訊塊之間的重疊長度（秒）。
            max_utt_sec (float): 一個音訊段落的最長持續時間（秒），超過則強制切分。
            stall_ms (int): 當辨識結果停滯超過此毫秒數時，視為句子結束並切分。
            partial_min_interval (float): 初步辨識結果的最小更新間隔（秒），用於UI節流。
            min_utt_sec (float): 觸發停頓切分的最小句子長度（秒）。
        """
        self.streaming_asr = streaming_asr

        # Audio parameters
        self.sample_rate = sample_rate
        self.bytes_per_sample = 2  # s16le
        self.chunk_bytes = int(self.sample_rate * self.bytes_per_sample * chunk_sec)
        self.overlap_bytes = int(self.sample_rate * self.bytes_per_sample * overlap_sec)
        self.step_time_sec = (self.chunk_bytes - self.overlap_bytes) / (self.sample_rate * self.bytes_per_sample)
        self._ring: bytes = b""

        # Segmentation control parameters
        self.max_utt_sec = max_utt_sec
        self.stall_ms = stall_ms
        self.partial_min_interval = partial_min_interval
        self.min_utt_sec = min_utt_sec

        # State variables
        self._last_partial = ""
        self._last_change_t = time.monotonic()
        self._last_emit_t = 0.0
        self._seg_bytes: List[bytes] = []
        self._seg_start_time = 0.0
        self.estimated_end_time = 0.0

        # File saving settings
        self.out_dir = "audios"
        self.filename_prefix = "seg"
        self._seg_index = 1
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

    def _feed_asr_engine(self, raw_bytes: bytes):
        """將 PCM 音訊數據餵給底層的 ASR 引擎。"""
        self.streaming_asr.accept_pcm16_bytes(raw_bytes, sample_rate=self.sample_rate)

    def _concat_seg_pcm(self) -> bytes:
        """將當前段落累積的 PCM bytes 串接起來。"""
        return b"".join(self._seg_bytes) if self._seg_bytes else b""

    def _write_wav(self, pcm_bytes: bytes) -> str:
        """將 PCM bytes 寫入 WAV 檔案並回傳路徑。"""
        fname = f"{self.filename_prefix}_{self._seg_index:04d}.wav"
        path = os.path.join(self.out_dir, fname)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.bytes_per_sample)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        self._seg_index += 1
        return path

    def _finalize_segment(self, text: str) -> FinalizedSegment:
        """
        結束目前音訊段落的處理。

        此方法會將累積的音訊儲存到檔案，並重設內部狀態以準備下一個段落。
        """
        pcm_data = self._concat_seg_pcm()
        if not pcm_data:
            return None

        audio_path = None
        if self.out_dir:
            audio_path = self._write_wav(pcm_data)

        seg_start = self._seg_start_time
        seg_end = self.estimated_end_time

        # Reset for the next segment
        self._seg_start_time = seg_end
        self._seg_bytes = []
        self.streaming_asr.reset()
        now = time.monotonic()
        self._last_partial = ""
        self._last_change_t = now
        self._last_emit_t = now

        return (seg_start, seg_end, text, audio_path)

    def flush(self) -> FinalizedSegment:
        """
        處理剩餘的音訊緩衝區，強制結束當前的段落。
        """
        if not self._seg_bytes:
            return None

        final_text = ""
        if self._ring:
            self._feed_asr_engine(self._ring)
            final_text = self.streaming_asr.decode().strip()
            self._ring = b""

        return self._finalize_segment(final_text)

    def stream_transcribe(
        self, audio_bytes: bytes
    ) -> Tuple[Optional[str], FinalizedSegment]:
        """
        處理傳入的音訊流塊。

        Args:
            audio_bytes: s16le 格式的 PCM 音訊數據。

        Returns:
            一個元組 (partial_result, finalized_segment)。
            - partial_result: 最新的初步辨識結果 (用於UI顯示)。
            - finalized_segment: 如果一個段落剛結束，則為 (開始時間, 結束時間, 文字)，否則為 None。
        """
        if not audio_bytes:
            return None, self.flush()

        self._seg_bytes.append(audio_bytes)
        buf = self._ring + audio_bytes
        buffer_position = 0
        step_size = max(1, self.chunk_bytes - self.overlap_bytes)
        emitted_partial: Optional[str] = None

        while buffer_position + self.chunk_bytes <= len(buf):
            # 1. 提取 chunk 並餵給 ASR 引擎
            piece = buf[buffer_position:buffer_position + self.chunk_bytes]
            self._feed_asr_engine(piece)

            # 2. 更新估算的時間
            # 我們的處理進度向前推進了一個步長的時間
            self.estimated_end_time += self.step_time_sec

            # 3. 獲取並處理初步辨識結果
            now = time.monotonic()
            partial = self.streaming_asr.decode()
            stripped = partial.strip()

            # --- 初步結果更新邏輯 (UI 優化) ---
            if partial != self._last_partial:
                self._last_partial = partial
                self._last_change_t = now  # 記錄結果變化的時間點，用於後續偵測停頓

                # 更新節流：避免過於頻繁地更新畫面導致閃爍
                if stripped and (now - self._last_emit_t) >= self.partial_min_interval:
                    emitted_partial = partial
                    self._last_emit_t = now

            # 4. 判斷是否需要切分段落
            # 計算目前段落的持續時間
            current_utt_duration = self.estimated_end_time - self._seg_start_time

            # 定義分段條件
            too_long = current_utt_duration >= self.max_utt_sec
            stalled = (now - self._last_change_t) * 1000.0 >= self.stall_ms
            stalled_trigger = stalled and (current_utt_duration >= self.min_utt_sec)
            punct_trigger = any(p in stripped for p in self.puncts)

            # 如果滿足任一分段條件，則結束目前段落
            if too_long or stalled_trigger or (punct_trigger and current_utt_duration > 1.0):
                finalized_segment = self._finalize_segment(stripped)

                # 將本次處理後剩餘的音訊存入 ring 緩衝區，供下次使用
                self._ring = buf[buffer_position + step_size:]
                return emitted_partial, finalized_segment

            # 5. 向前滑動視窗
            buffer_position += step_size

        self._ring = buf[buffer_position:]
        return emitted_partial, None