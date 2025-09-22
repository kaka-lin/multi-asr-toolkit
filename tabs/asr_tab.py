from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from core.demucs import demix_audio
from core.transcription import transcribe_audio
from utils.preprocess import get_media_path

refresh_symbol = '🔄'


@dataclass
class TranscriptionOptions:
    """Options for transcription."""
    backend: str
    language: str
    model_size: str
    word_timestamps: bool


def run_demixing(audio_path, model_name):
    if not audio_path:
        return (None, None, gr.Textbox("錯誤：音訊檔案路徑為空", visible=True))

    audio_path = Path(audio_path)
    output_dir = Path("demix_output")

    separated_files, vocal_audio_path = demix_audio(
        audio_path, model_name, output_dir
    )

    # Convert Path objects to strings for Gradio
    file_paths = [str(p) for p in separated_files]

    # If no vocal track is found, return an error message
    if vocal_audio_path is None:
        return (
            file_paths,
            str(audio_path),
            gr.Textbox("警告：未找到人聲音軌，請檢查音訊檔案。", visible=True)
        )

    return (file_paths, vocal_audio_path, gr.Textbox("", visible=False))


def transcribe_and_update_video(
    audio_path: str,
    options: TranscriptionOptions,
    video_path: str
):
    """
    Transcribes the audio and updates the video with subtitles.

    Args:
        audio_path (str): The path to the audio file.
        options (TranscriptionOptions): The transcription options.
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing the transcribed text, subtitle results,
               and the video path with subtitles.
    """
    text, subtitle_results = transcribe_audio(
        audio_path,
        options.backend,
        options.language,
        options.model_size,
        options.word_timestamps
    )
    srt_path = subtitle_results[0]
    if video_path and srt_path:
        return text, subtitle_results, (video_path, srt_path)
    return text, subtitle_results, video_path


def create_transcription_options_and_transcribe(
    audio_path, backend, language, model_size, word_timestamps, video_path
):
    """Helper function to create TranscriptionOptions and call transcribe."""
    options = TranscriptionOptions(
        backend=backend,
        language=language,
        model_size=model_size,
        word_timestamps=word_timestamps,
    )
    return transcribe_and_update_video(audio_path, options, video_path)


def create_asr_tab(
    demix_choices, asr_backend_choices, language_choices,
    model_size_options, update_backend_options,
):
    """
    Creates the ASR (Automatic Speech Recognition) tab for the Gradio
    interface.

    This tab includes components for uploading audio/video files, recording
    from a microphone, providing a YouTube URL, demixing audio, and
    performing speech-to-text transcription.å
    """
    # Group arguments for clarity
    options = {
        "demix_choices": demix_choices,
        "asr_backend_choices": asr_backend_choices,
        "language_choices": language_choices,
        "model_size_options": model_size_options,
        "refresh_symbol": refresh_symbol,
    }
    callbacks = {
        "update_backend_options": update_backend_options,
    }

    # Dictionary to hold all Gradio components
    c = {}

    with gr.TabItem("ASR Tab"):
        with gr.Row():
            with gr.Column(scale=3):
                c["file_input"] = gr.Audio(
                    sources=["upload"], type="filepath", label="📂 上傳檔案"
                )
                c["mic_input"] = gr.Audio(
                    sources=["microphone"], type="filepath", label="🎙️ 麥克風錄音"
                )
                c["youtube_url"] = gr.Textbox(label="YouTube URL")
                c["yt_quality"] = gr.Radio(
                    choices=["low", "good", "best"], value="best",
                    label="YouTube Video Quality", interactive=True
                )
                c["audio_format"] = gr.Radio(
                    choices=["wav", "flac", "mp3"], value="mp3",
                    label="Audio Format", interactive=True
                )
                c["submit_btn"] = gr.Button("上傳")

            with gr.Column(scale=4):
                c["video_preview"] = gr.Video(label="Video", interactive=False)
                c["audio_preview"] = gr.Audio(
                    label="Audio", interactive=False, type="filepath"
                )
                c["output_text"] = gr.Textbox(label="📝 辨識結果")
                c["error_box"] = gr.Textbox(label="錯誤訊息", visible=False)

            with gr.Column(scale=3):
                # Demixing section
                c["demix_mode_dropdown"] = gr.Dropdown(
                    choices=options["demix_choices"],
                    value=options["demix_choices"][2],
                    label="MDX Models", interactive=True
                )
                c["demix_audio_format"] = gr.Radio(
                    choices=["wav", "flac", "mp3"], value="mp3",
                    label="Audio Format", interactive=True
                )
                c["demix_results"] = gr.File(
                    label="Demixing", interactive=False, file_count="multiple"
                )
                with gr.Row():
                    c["refresh_btn"] = gr.Button(
                        f"Refresh Model {options['refresh_symbol']}",
                        interactive=True
                    )
                    c["demix_btn"] = gr.Button("Demixing")

                # ASR section
                c["asr_backend_dropdown"] = gr.Dropdown(
                    choices=options["asr_backend_choices"], label="選擇引擎",
                    value=options["asr_backend_choices"][0]
                )
                c["language_dropdown"] = gr.Dropdown(
                    choices=options["language_choices"], label="語言",
                    value=options["language_choices"][0]
                )
                c["modelsize_dropdown"] = gr.Dropdown(
                    choices=options["model_size_options"]["transformers"],
                    label="模型大小", value="small"
                )
                c["word_timestamps_check"] = gr.Checkbox(
                    label="Word Timestamps - Highlight Words", value=True,
                    interactive=True
                )
                c["subtitle_results"] = gr.File(
                    label="Subtitles", interactive=False, file_count="multiple"
                )
                c["transcribe_btn"] = gr.Button("轉錄")

    # Action parts
    c["submit_btn"].click(  # pylint: disable=no-member
        fn=get_media_path,
        inputs=[
            c["file_input"], c["mic_input"], c["youtube_url"],
            c["yt_quality"], c["audio_format"]
        ],
        outputs=[c["video_preview"], c["audio_preview"], c["error_box"]],
    )

    c["demix_btn"].click(  # pylint: disable=no-member
        fn=run_demixing,
        inputs=[c["audio_preview"], c["demix_mode_dropdown"]],
        outputs=[
            c["demix_results"], c["audio_preview"], c["error_box"],
        ],
    )

    c["asr_backend_dropdown"].change(  # pylint: disable=no-member
        fn=callbacks["update_backend_options"],
        inputs=c["asr_backend_dropdown"],
        outputs=[c["modelsize_dropdown"], c["language_dropdown"]]
    )

    c["transcribe_btn"].click(  # pylint: disable=no-member
        fn=create_transcription_options_and_transcribe,
        inputs=[
            c["audio_preview"], c["asr_backend_dropdown"],
            c["language_dropdown"], c["modelsize_dropdown"],
            c["word_timestamps_check"], c["video_preview"],
        ],
        outputs=[c["output_text"], c["subtitle_results"], c["video_preview"]]
    )