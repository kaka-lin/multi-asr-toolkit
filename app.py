import argparse
import platform
from pathlib import Path

import gradio as gr

from cli import run_cli

from tabs import create_asr_tab, create_split_audio_tab

demix_choices = ["UVR-MDX-NET Inst HQ4", "UVR-MDX-NET-Voc_FT", "htdemucs"]
asr_backend_choices = ["transformers", "faster-whisper", "whisper", "speech-recognition", "sherpa-onnx"]
language_choices = ["zh", "en", "ja", "ko", "auto"]

model_size_options = {
    "transformers": ["tiny", "base", "small", "medium", "large"],
    "faster-whisper": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
    "whisper": ["tiny", "base", "small", "medium", "large"],
    "speech-recognition": ["google"],
    "sherpa-onnx": ["zipformer"]
}

if platform.system() == "Darwin":
    # macOS specific settings
    asr_backend_choices.append("mlx-whisper")
    model_size_options["mlx-whisper"] = [
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v1", "large-v2", "large-v3",
        "large-v3-turbo", "large"
    ]


def update_backend_options(backend):
    model_update = gr.update(
        choices=model_size_options[backend],
        value=model_size_options[backend][0],
    )
    language_update = gr.update(
        visible=(backend != "mlx-whisper")
    )
    return model_update, language_update


def build_demo():
    # Create a Gradio interface
    with gr.Blocks() as app:
        # UI Parts
        gr.Markdown("## 🧠 Multi-ASR Toolkit - 語音轉文字平台")
        with gr.Tabs():
            create_asr_tab(
                demix_choices, asr_backend_choices, language_choices,
                model_size_options, update_backend_options,
            )

            create_split_audio_tab()

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
