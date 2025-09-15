import argparse
import platform
from pathlib import Path

import gradio as gr

from cli import run_cli
from utils.model import transcribe_audio, demix_audio
from utils.preprocess import get_media_path
from tabs.asr_tab import create_asr_tab

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
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v1", "large-v2", "large-v3", 
        "large-v3-turbo", "large"
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


def update_backend_options(backend):
    model_update = gr.update(
        choices=model_size_options[backend],
        value=model_size_options[backend][0],
    )
    language_update = gr.update(
        visible=(backend != "mlx-whisper")
    )
    return model_update, language_update


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
        # UI Parts
        gr.Markdown("## 🧠 Multi-ASR Toolkit - 語音轉文字平台")
        with gr.Tabs():
            create_asr_tab(
                demix_choices, asr_backend_choices, language_choices, model_size_options,
                refresh_symbol, get_media_path, run_demixing, update_backend_options,
                transcribe_and_update_video
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
