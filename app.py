import argparse
import platform
from pathlib import Path

import gradio as gr

from cli import run_cli
from utils.model import transcribe_audio, demix_audio
from utils.preprocess import get_media_path

demix_choices = ["UVR-MDX-NET Inst HQ4", "UVR-MDX-NET-Voc_FT", "htdemucs"]
asr_backend_choices = ["transformers", "faster-whisper", "whisper", "speech-recognition"]
language_choices = ["zh", "en", "ja", "ko", "auto"]

model_size_options = {
    "transformers": ["tiny", "base", "small", "medium", "large"],
    "faster-whisper": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
    "whisper": ["tiny", "base", "small", "medium", "large"],
    "speech-recognition": ["google"],
}

if platform.system() == "Darwin":
    # macOS specific settings
    asr_backend_choices.append("mlx-whisper")
    model_size_options["mlx-whisper"] = [
        "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"
    ]

refresh_symbol = '🔄'


def run_demixing(audio_path, model_name):
    if not audio_path:
        return (None, None, gr.Textbox("錯誤：音訊檔案路徑為空", visible=True))

    audio_path = Path(audio_path)
    output_dir = Path("demix_output")

    separated_files, vocal_audio_path = demix_audio(audio_path, model_name, output_dir)

    # Convert Path objects to strings for Gradio
    file_paths = [str(p) for p in separated_files]

    # If no vocal track is found, return an error message
    if vocal_audio_path is None:
        return (file_paths, str(audio_path), gr.Textbox("警告：未找到人聲音軌，請檢查音訊檔案。", visible=True))

    return (file_paths, vocal_audio_path, gr.Textbox("", visible=False))


def update_model_size_dropdown(backend):
    return gr.Dropdown(
        choices=model_size_options[backend],
        value=model_size_options[backend][0],
    )


def transcribe_and_update_video(
    audio_path, backend, language, model_size, word_timestamps, video_path
):
    subtitle_results = []

    text, subtitle_results = transcribe_audio(
        audio_path, backend, language, model_size, word_timestamps
    )
    srt_path = subtitle_results[0]
    if video_path and srt_path:
        return text, subtitle_results, (video_path, srt_path)
    return text, subtitle_results, video_path


def build_demo():
    # Create a Gradio interface
    with gr.Blocks() as app:
        # UI parts
        gr.Markdown("## 🧠 Multi-ASR Toolkit - 語音轉文字平台")

        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.Audio(
                    sources=["upload"], type="filepath", label="📂 上傳檔案"
                )
                mic_input = gr.Audio(
                    sources=["microphone"], type="filepath", label="🎙️ 麥克風錄音"
                )
                youtube_url = gr.Textbox(label="YouTube URL")
                yt_quality = gr.Radio(
                    choices=["low", "good", "best"],
                    value="best",
                    label="YouTube Video Quality",
                    interactive=True
                )
                audio_format = gr.Radio(
                    choices=["wav", "flac", "mp3"],
                    value="mp3",
                    label="Audio Format",
                    interactive=True
                )
                submit_btn = gr.Button("上傳")
            
            with gr.Column(scale=4):
                video_preview = gr.Video(label="Video", interactive=False)
                audio_preview = gr.Audio(
                    label="Audio", interactive=False, type="filepath"
                )
                output_text = gr.Textbox(label="📝 辨識結果")
                error_box = gr.Textbox(label="錯誤訊息", visible=False)

            with gr.Column(scale=3):
                # Demixing section
                demix_mode_dropdown = gr.Dropdown(
                    choices=demix_choices,
                    value=demix_choices[2],
                    label="MDX Models",
                    interactive=True
                )
                demix_audio_format = gr.Radio(
                    choices=["wav", "flac", "mp3"],
                    value="mp3",
                    label="Audio Format",
                    interactive=True
                )
                demix_results = gr.File(label="Demixing", interactive=False, file_count="multiple")
                with gr.Row():
                    refresh_btn = gr.Button(
                        f"Refresh Model {refresh_symbol}", interactive=True
                    )
                    demix_btn = gr.Button("Demixing")
        
                # ASR section
                backend_dropdown = gr.Dropdown(
                    choices=asr_backend_choices, label="選擇引擎", value=asr_backend_choices[0]
                )
                language_dropdown = gr.Dropdown(
                    choices=language_choices, label="語言", value=language_choices[0]
                )
                modelsize_dropdown = gr.Dropdown(
                    choices=model_size_options["transformers"],
                    label="模型大小",
                    value="small"
                )
                word_timestamps_check = gr.Checkbox(
                    label="Word Timestamps - Highlight Words", value=True, interactive=True
                )
                subtitle_results = gr.File(label="Subtitles", interactive=False, file_count="multiple")
                transcribe_btn = gr.Button("轉錄")
                
        
        # Action parts
        submit_btn.click(
            fn=get_media_path,
            inputs=[file_input, mic_input, youtube_url, yt_quality, audio_format],
            outputs=[video_preview, audio_preview, error_box],
        )

        demix_btn.click(
            fn=run_demixing,
            inputs=[audio_preview, demix_mode_dropdown],
            outputs=[
                demix_results,
                audio_preview,
                error_box,
            ],
        )

        backend_dropdown.change(
            fn=update_model_size_dropdown,
            inputs=backend_dropdown,
            outputs=modelsize_dropdown
        )

        transcribe_btn.click(
            fn=transcribe_and_update_video,
            inputs=[
                audio_preview, backend_dropdown, language_dropdown, 
                modelsize_dropdown, word_timestamps_check, video_preview,
            ],
            outputs=[output_text, subtitle_results, video_preview]
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-ASR Toolkit", add_help=True
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="web",
        help="選擇執行模式：cli 或 web (Gradio UI)，預設為 web"
    )
    known_args, remaining_args = parser.parse_known_args()
    return known_args, remaining_args


def main():
    known_args, remaining_args = parse_args()
    if known_args.mode == "cli":
        run_cli(remaining_args)
    else:
        # Launch the app
        # Gradio defaults to localhost:7860
        demo = build_demo()
        demo.queue()
        # To create a public link, set share=True
        demo.launch()


if __name__ == "__main__":
    main()
