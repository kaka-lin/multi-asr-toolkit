import argparse
from utils.model import transcribe_audio


def run_cli(args):
    parser = argparse.ArgumentParser(description="Run Multi-ASR Toolkit in CLI mode")

    parser.add_argument("audio_path", help="輸入的音訊檔案路徑（支援 mp3, wav 等）")
    parser.add_argument("--backend", choices=["transformers", "faster-whisper", "whisper", "speech-recognition"],
                        default="transformers", help="選擇辨識後端")
    parser.add_argument("--language", default="auto", help="語音語言（如 zh, en, ja, ko）")
    parser.add_argument("--model-size", default="small", help="模型大小，例如 small、medium、large")

    args = parser.parse_args()

    # transcribe audio
    print(f"[INFO] 使用 {args.backend} 後端進行辨識...")
    result = transcribe_audio(
        audio_input=args.audio_path,
        backend=args.backend,
        language=args.language,
        model_size=args.model_size
    )
    print("\n📝 辨識結果：\n")
    print(result)
