from pathlib import Path

import torch

from backends import get_asr_backend
from backends.demucs.api import Separator, save_audio
from utils.subtitle_generator import save_transcription_results


def translate_model_name(model_name: str) -> str:
    """Translate the model name from OpenAI Whisper to MLX format."""
    model_mapping = {
        "tiny.en": "mlx-community/whisper-tiny.en-mlx",
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base.en": "mlx-community/whisper-base.en-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small.en": "mlx-community/whisper-small.en-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium.en": "mlx-community/whisper-medium.en-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v1": "mlx-community/whisper-large-v1-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "large": "mlx-community/whisper-large-mlx"
    }
    
    # Retrieve the corresponding MLX model path
    mlx_model_path = model_mapping.get(model_name)

    if mlx_model_path:
        return mlx_model_path
    else:
        raise ValueError(
            f"Model name '{model_name}' is not recognized or not supported."
        )


def resolve_model_name(backend: str, model_size: str) -> str:
    if backend == "transformers":
        return f"openai/whisper-{model_size}"
    elif backend == "speech-recognition":
        return None
    elif backend == "mlx-whisper":
        return translate_model_name(model_size)
    return model_size


def transcribe_audio(audio_input, backend, language, model_size, word_timestamps=True):
    """
    Transcribes audio using the specified backend.
    The audio can be a file path (str) or a tuple of (sample_rate, numpy_array).
    """
    audio = None
    # Check if input is a file path
    if isinstance(audio_input, str):
        audio = audio_input
    else:
        raise ValueError("Invalid audio input type. Expected a file path or a tuple.")

    model_name = resolve_model_name(backend, model_size)
    kwargs = {"language": language}
    if model_name:
        kwargs["model_size"] = model_name
    asr = get_asr_backend(backend, **kwargs)
    result = asr.transcribe(audio, word_timestamps=word_timestamps)

    # Convert the result to ASRResult format
    asr_result = asr.to_asr_result(result)

    # Extract the subtitle text
    subtitle_results = save_transcription_results(asr_result, audio)

    return asr_result.text, subtitle_results


def demix_audio(audio_path: Path, model_name: str, output_dir: Path):
    """
    使用指定的 demucs 模型分離音訊檔案，並將分離後的音軌儲存到指定目錄。

    Args:
        audio_path (Path): 要處理的音訊檔案路徑。
        model_name (str): 要使用的 demucs 模型名稱。
        output_dir (Path): 儲存分離音軌的目錄。
    
    Returns:
        list[Path]: 分離後音軌檔案的路徑列表。
    """
    if not audio_path.is_file():
        print(f"錯誤：找不到指定的音訊檔案 -> {audio_path}")
        return []

    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在初始化 Demucs Separator (模型: {model_name})...")
    try:
        separator = Separator(
            model=model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            progress=True
        )
    except Exception as e:
        print(f"初始化 Separator 時發生錯誤: {e}")
        print("請確認您的環境已正確安裝 demucs 及其相依套件。")
        return []

    print(f"正在使用模型 '{separator._name}' 在裝置 '{separator._device}' 上進行分離...")
    print(f"音訊檔案: {audio_path}")

    try:
        origin, separated = separator.separate_audio_file(audio_path)
    except Exception as e:
        print(f"音訊分離過程中發生錯誤: {e}")
        return []

    print("音訊分離完成，正在儲存檔案...")
    
    output_paths = []
    vocal_only_audio_path = None
    for stem_name, stem_tensor in separated.items():
        output_filename = f"{audio_path.stem}_{stem_name}.mp3"
        output_path = output_dir / output_filename

        print(f"  - 正在儲存: {output_path}")

        save_audio(
            stem_tensor,
            str(output_path),
            samplerate=separator.samplerate
        )
        output_paths.append(output_path)

        if stem_name == "vocals":
            vocal_only_audio_path = str(output_path)
            print(f"  - 偵測到人聲音軌，已儲存為: {output_path}")
        
    print("\n所有音軌已成功儲存！")
    return output_paths, vocal_only_audio_path