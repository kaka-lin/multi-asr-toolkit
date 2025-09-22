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
