import os
import json

from .dataclasses import ASRResult


def _format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT/VTT timestamp format."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def save_as_srt(result: ASRResult, audio_path: str):
    """Saves the transcription result in SRT format."""
    if not result.segments:
        print("Warning: No segments found, cannot generate SRT file.")
        return None

    output_path = os.path.splitext(audio_path)[0] + ".srt"
    with open(output_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result.segments):
            start_time = _format_timestamp(segment.start)
            end_time = _format_timestamp(segment.end)
            text = segment.text.strip()
            if segment.speaker:
                text = f"[{segment.speaker}] {text}"
            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")
    return output_path


def save_as_vtt(result: ASRResult, audio_path: str):
    """Saves the transcription result in VTT format."""
    if not result.segments:
        print("Warning: No segments found, cannot generate VTT file.")
        return

    output_path = os.path.splitext(audio_path)[0] + ".vtt"
    with open(output_path, "w", encoding="utf-8") as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        for segment in result.segments:
            start_time = _format_timestamp(segment.start).replace(",", ".")
            end_time = _format_timestamp(segment.end).replace(",", ".")
            text = segment.text.strip()
            if segment.speaker:
                text = f"[{segment.speaker}] {text}"
            vtt_file.write(f"{start_time} --> {end_time}\n")
            vtt_file.write(f"{text}\n\n")


def save_as_txt(result: ASRResult, audio_path: str):
    """Saves the transcription result in TXT format."""
    output_path = os.path.splitext(audio_path)[0] + ".txt"
    with open(output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(result.text.strip())


def save_as_json(result: ASRResult, audio_path: str):
    """Saves the transcription result in JSON format."""
    output_path = os.path.splitext(audio_path)[0] + ".json"
    # Convert ASRResult to a dictionary for JSON serialization
    result_dict = {
        "text": result.text,
        "language": result.language,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker,
                "tokens": [
                    {
                        "start": tok.start,
                        "end": tok.end,
                        "token": tok.token,
                        "probability": tok.probability,
                    }
                    for tok in seg.tokens
                ],
            }
            for seg in result.segments
        ],
    }
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(result_dict, json_file, indent=4, ensure_ascii=False)


def save_transcription_results(result: ASRResult, audio_path: str):
    """Saves the transcription result in all supported formats."""
    srt_path = save_as_srt(result, audio_path)
    save_as_vtt(result, audio_path)
    save_as_txt(result, audio_path)
    save_as_json(result, audio_path)

    return srt_path
