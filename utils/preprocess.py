import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

import yt_dlp


def extract_audio(video_path, audio_format="mp3"):
    """
    從影片檔案中提取音訊。

    Args:
        video_path (str): 影片檔案路徑。
        audio_format (str): 要提取的音訊格式。

    Returns:
        str: 提取出的音訊檔案路徑，若失敗則為 None。
    """
    try:
        audio_path = os.path.splitext(video_path)[0] + f".{audio_format}"
        if os.path.exists(audio_path):
            return audio_path
    
        codec = "copy" if audio_format == "m4a" else "libmp3lame"
        command = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", codec, "-y", audio_path,
        ]
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return audio_path
    except Exception:
        # 如果 'copy' 編碼器失敗，嘗試不使用它
        try:
            audio_path = os.path.splitext(video_path)[0] + f".{audio_format}"
            command = ["ffmpeg", "-i", video_path, "-vn", "-y", audio_path]
            subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return audio_path
        except Exception as e:
            print(f"提取音訊失敗: {e}")
            return None


def extract_audio_chunks_from_video(
    video_path: str,
    chunk_sec: float = 1.2,
) -> Iterable[bytes]:
    """以串流方式，從影片解出 s16le/16k/mono PCM，並以 chunk_sec 為單位 yield。"""
    assert Path(video_path).is_file(), f"File not found: {video_path}"
    
    SR = 16000
    bytes_per_sec = SR * 2  # s16le, 1ch
    frame_bytes = int(bytes_per_sec * chunk_sec)

    command = [
        "ffmpeg",
        "-hide_banner", "-nostdin", "-loglevel", "warning",
        "-i", video_path,
        "-vn",
        "-ac", "1", "-ar", str(SR),
        "-acodec", "pcm_s16le",
        "-f", "s16le",
        "-"
    ]
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**6
    )
    
    try:
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                break
            yield chunk
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


def download_youtube_video(
    url: str,
    output_dir: str = "downloads",
    filename: str = "",
    cookies_from_browser: str = "chrome",
    yt_quality: str = "best",
    extract_audio_format: str = "mp3"
) -> (str, str, str):
    """
    下載 YouTube 影片並提取音訊。

    Args:
        url (str): YouTube 影片網址。
        output_dir (str): 檔案儲存資料夾。
        filename (str): 下載後的檔名（不含副檔名）。
        cookies_from_browser (str): 要從哪個瀏覽器讀取 cookies。
        yt_quality (str): 影片品質 ('low', 'good', 'best')。
        extract_audio_format (str): 要提取的音訊格式 (e.g., 'mp3', 'wav')。

    Returns:
        video_path (str): 下載後影片的完整路徑。
        audio_path (str): 提取出音訊的完整路徑。
        error_message (str): 錯誤訊息，若成功則為 None。
    """
    # 解析 YouTube video id
    pattern = (
        r"(?:https?://)?"
        r"(?:www\.)?"
        r"(?:youtube\.com/watch\?v=|youtu\.be/)"
        r"([\w\-]{11})"
    )
    match = re.match(pattern, url.strip())
    if not match:
        return None, None, "請輸入合法的 YouTube 連結"

    # File output path
    video_id = match.group(1)
    os.makedirs(output_dir, exist_ok=True)

    base_filename = filename if filename != "" else video_id
    video_filepath = os.path.join(output_dir, f"{base_filename}.mp4")
    audio_filepath = os.path.join(
        output_dir, f"{base_filename}.{extract_audio_format}"
    )

    if os.path.exists(video_filepath) and os.path.exists(audio_filepath):
        return video_filepath, audio_filepath, None

    quality_map = {
        "best": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "good": (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
            "best[height<=720][ext=mp4]/best"
        ),
        "low": (
            "worstvideo[ext=mp4]+worstaudio[ext=m4a]/"
            "worst[ext=mp4]/worst"
        ),
    }
    format_string = quality_map.get(yt_quality, quality_map["best"])

    ydl_opts = {
        'format': format_string,
        'outtmpl': video_filepath,
        'quiet': True,
        'merge_output_format': 'mp4',
        'cookiefile': 'youtube_cookies.txt',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(video_filepath):
            return None, None, "影片下載失敗，找不到檔案。"

        extracted_audio_path = extract_audio(video_filepath, extract_audio_format)
        if not extracted_audio_path:
            return video_filepath, None, "音訊提取失敗。"

        return video_filepath, extracted_audio_path, None

    except Exception as e:
        return None, None, f"下載或處理失敗：{str(e)}"


def get_media_path(
    file_input, mic_input=None, youtube_url=None, yt_quality="best", audio_format="mp3"
):
    """
    從檔案、麥克風或 YouTube URL 獲取媒體路徑。

    Args:
        file_input (str): 上傳的音訊檔案路徑。
        mic_input (str): 麥克風錄音的音訊檔案路徑。
        youtube_url (str): YouTube 影片網址。
        yt_quality (str): YouTube 影片品質。
        audio_format (str): YouTube 音訊格式。

    Returns:
        tuple: (video_path, audio_path, error_message)
               video_path: 用於更新影片預覽的路徑，如果沒有則為 None。
               audio_path: 用於更新音訊預覽的路徑。
               error_message: 錯誤訊息，如果沒有則為空字串。
    """
    if file_input:
        return None, file_input, ""

    if mic_input:
        return None, mic_input, ""

    if youtube_url:
        video_path, audio_path, err = download_youtube_video(
            youtube_url,
            yt_quality=yt_quality,
            extract_audio_format=audio_format
        )
        # Return paths even if there's a partial error
        if err:
            return video_path, audio_path, err

        return video_path, audio_path, ""

    return None, None, "請錄音、上傳檔案或提供 YouTube 連結"