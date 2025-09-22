import os
from typing import List, Union

import gradio as gr

from utils.preprocess import extract_audio_chunks_from_video
from backends import get_asr_backend
from core.transcription import StreamingTranscriber


def load_audio(selected_row):
    # selected_row 是一個包含該列所有值的列表
    # 例如：['audios/sample.wav', '範例音訊 1 (440 Hz)']
    return selected_row[3], selected_row[4]


def ui_stream_process(
    video,
    chunk_sec,
    overlap_sec,
    max_utt_sec,
    stall_ms,
    min_utt_sec,
    partial_min_interval,
):
    """
    Gradio 事件函式（generator）：上傳影片後立即開始「串流式」轉寫。
    """
    streaming_asr = get_asr_backend("sherpa-onnx")

    transcriber = StreamingTranscriber(
        streaming_asr,
        chunk_sec=float(chunk_sec),
        overlap_sec=float(overlap_sec),
        max_utt_sec=float(max_utt_sec),
        stall_ms=int(stall_ms),
        min_utt_sec=float(min_utt_sec),
        partial_min_interval=float(partial_min_interval),
    )

    samples: List[List[Union[int, float, str]]] = []
    idx = 1
    latest_audio = None

    # 初次回傳空資料集，避免 UI 還沒資料時報型別
    yield "", gr.Dataset(samples=[]), None

    for pcm in extract_audio_chunks_from_video(video, chunk_sec=float(chunk_sec)):
        partial, finalized_segment = transcriber.stream_transcribe(pcm)
 
        # 即時更新初步辨識結果
        if partial:
            yield partial, None, latest_audio

        # 當一個段落結束時，更新 Dataset
        if finalized_segment:
            t0, t1, text, audio_path = finalized_segment
            row = [
                idx,
                f"{t0:.2f}",
                f"{t1:.2f}",
                text,
                audio_path,
            ]
            samples.append(row)
            idx += 1
            latest_audio = audio_path
            yield partial or "", gr.update(samples=samples), latest_audio


def create_split_audio_tab():
    with gr.TabItem("Split Audio Tab"):
        with gr.Row():
            with gr.Column(scale=2):
                video = gr.Video(
                    sources=["upload"], label="📂 上傳檔案", interactive=True
                )
                with gr.Accordion("進階參數", open=False):
                    chunk_sec = gr.Slider(0.4, 3.0, value=1.2, step=0.1, label="外部輸入 chunk（秒）")
                    overlap_sec = gr.Slider(0.0, 1.2, value=0.25, step=0.05, label="內部滑窗重疊（秒）")
                    max_utt_sec = gr.Slider(2, 20, value=6.0, step=0.5, label="最長一句（秒）")
                    stall_ms = gr.Slider(200, 2000, value=900, step=50, label="久未變強制切段（毫秒）")
                    partial_min_interval = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="初步結果更新間隔（秒）")
                    min_utt_sec = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="停頓觸發的最小句長")

                transcribe_btn = gr.Button("開始處理")

            with gr.Column(scale=4):
                partial = gr.Textbox(label="即時辨識結果", lines=4)
                player = gr.Audio(label="當前播放的音訊")

                segments = gr.Dataset(
                    components=[
                        gr.Number(label="#", precision=0, visible=False),
                        gr.Textbox(label="開始(s)", visible=False),
                        gr.Textbox(label="結束(s)", visible=False),
                        gr.Textbox(label="辨識文字", visible=False),
                        gr.Audio(label="Audio", type="filepath", visible=False),
                    ],
                    headers=["#", "開始", "結束", "文字", "音訊"],
                    samples=[],
                    label="已切分音訊段落 (可直接在表格中播放)",
                )

        # --- Event Listeners ---
        transcribe_btn.click(
            fn=ui_stream_process,
            inputs=[
                video,
                chunk_sec,
                overlap_sec,
                max_utt_sec,
                stall_ms,
                min_utt_sec,
                partial_min_interval,
            ],
            outputs=[partial, segments, player],
        )

        segments.click(
            fn=load_audio,
            inputs=[segments],
            outputs=[partial, player]
        )