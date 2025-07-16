import sys
import argparse
import platform

import gradio as gr

from cli import run_cli
from utils.model import transcribe_audio
from utils.preprocess import get_media_path

backend_choices = ["transformers", "faster-whisper", "whisper", "speech-recognition"]
language_choices = ["zh", "en", "ja", "ko", "auto"]

model_size_options = {
    "transformers": ["tiny", "base", "small", "medium", "large"],
    "faster-whisper": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
    "whisper": ["tiny", "base", "small", "medium", "large"],
    "speech-recognition": ["google"],
}

if platform.system() == "Darwin":
    # macOS specific settings
    backend_choices.append("mlx-whisper")
    model_size_options["mlx-whisper"] = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]


def update_model_size_dropdown(backend):
    return gr.Dropdown(
        choices=model_size_options[backend],
        value=model_size_options[backend][0],
    )


def transcribe_and_update_video(audio_path, backend, language, model_size, word_timestamps, video_path):
    text, srt_path = transcribe_audio(audio_path, backend, language, model_size, word_timestamps)
    if video_path and srt_path:
        return text, (video_path, srt_path)
    return text, video_path


def build_demo():
    # Create a Gradio interface
    with gr.Blocks() as app:
        # UI parts
        gr.Markdown("## 🧠 Multi-ASR Toolkit - 語音轉文字平台")

        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.Audio(sources=["upload"], type="filepath", label="📂 上傳檔案")
                mic_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ 麥克風錄音")
                youtube_url = gr.Textbox(label="YouTube URL")
                yt_quality = gr.Radio(
                    choices=["low", "good", "best"],
                    value="best",
                    label="YouTube Video Quality",
                    interactive=True
                )
                audio_format = gr.Radio(
                    choices=["wav", "flac", "mp3"],
                    value="mp3", label="Audio Format",
                    interactive=True
                )
                submit_btn = gr.Button("上傳")
            
            with gr.Column(scale=4):
                video_preview = gr.Video(label="Video", interactive=False)
                audio_preview = gr.Audio(label="Audio", interactive=False, type="filepath")
                output_text = gr.Textbox(label="📝 辨識結果")
                error_box = gr.Textbox(label="錯誤訊息", visible=False)

            with gr.Row(scale=3):
                backend_dropdown = gr.Dropdown(choices=backend_choices, label="選擇引擎", value=backend_choices[0])
                language_dropdown = gr.Dropdown(choices=language_choices, label="語言", value=language_choices[0])
                modelsize_dropdown = gr.Dropdown(
                    choices=model_size_options["transformers"],
                    label="模型大小",
                    value="small"
                )
                word_timestamps_check = gr.Checkbox(label="Word Timestamps - Highlight Words", value=True, interactive=True)
                transcribe_btn = gr.Button("轉錄")
        
        # Action parts
        submit_btn.click(
            fn=get_media_path,
            inputs=[file_input, mic_input, youtube_url, yt_quality, audio_format],
            outputs=[video_preview, audio_preview, error_box],
        )

        backend_dropdown.change(
            fn=update_model_size_dropdown,
            inputs=backend_dropdown,
            outputs=modelsize_dropdown
        )

        transcribe_btn.click(
            fn=transcribe_and_update_video,
            inputs=[audio_preview, backend_dropdown, language_dropdown, modelsize_dropdown, word_timestamps_check, video_preview],
            outputs=[output_text, video_preview]
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-ASR Toolkit", add_help=True)
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
