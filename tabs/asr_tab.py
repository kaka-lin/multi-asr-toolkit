import gradio as gr


def create_asr_tab(
    demix_choices, asr_backend_choices, language_choices, model_size_options,
    refresh_symbol, get_media_path, run_demixing, update_backend_options,
    transcribe_and_update_video
):
    """
    Creates the ASR (Automatic Speech Recognition) tab for the Gradio
    interface.

    This tab includes components for uploading audio/video files, recording
    from a microphone, providing a YouTube URL, demixing audio, and
    performing speech-to-text transcription.

    Args:
        demix_choices (list): A list of available demixing models.
        asr_backend_choices (list): A list of available ASR backend engines.
        language_choices (list): A list of supported languages for
                                 transcription.
        model_size_options (dict): A dictionary of available model sizes for
                                   different backends.
        refresh_symbol (str): The symbol/emoji for the refresh button.
        get_media_path (callable): Callback to process input media.
        run_demixing (callable): Callback to run the audio demixing process.
        update_backend_options (callable): Callback to update model and
                                           language options based on the
                                           selected ASR backend.
        transcribe_and_update_video (callable): Callback to run transcription
                                                and update the video with
                                                subtitles.
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
        "get_media_path": get_media_path,
        "run_demixing": run_demixing,
        "update_backend_options": update_backend_options,
        "transcribe_and_update_video": transcribe_and_update_video,
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
        fn=callbacks["get_media_path"],
        inputs=[
            c["file_input"], c["mic_input"], c["youtube_url"],
            c["yt_quality"], c["audio_format"]
        ],
        outputs=[c["video_preview"], c["audio_preview"], c["error_box"]],
    )

    c["demix_btn"].click(  # pylint: disable=no-member
        fn=callbacks["run_demixing"],
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
        fn=callbacks["transcribe_and_update_video"],
        inputs=[
            c["audio_preview"], c["asr_backend_dropdown"],
            c["language_dropdown"], c["modelsize_dropdown"],
            c["word_timestamps_check"], c["video_preview"],
        ],
        outputs=[c["output_text"], c["subtitle_results"], c["video_preview"]]
    )