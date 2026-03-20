"""音頻前處理工具"""

import logging

import numpy as np
import librosa
import soundfile as sf

from clonevoice.config import (
    MIN_AUDIO_DURATION,
    MAX_AUDIO_DURATION,
    TARGET_SAMPLE_RATE,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)


class AudioValidationError(Exception):
    """音頻驗證錯誤"""


def get_audio_duration(audio_path: str) -> float:
    """取得音頻長度（秒），支援 WAV/MP3/AAC/FLAC 等格式"""
    return librosa.get_duration(path=audio_path)


def validate_audio(audio_path: str) -> str:
    """驗證音頻檔案是否符合要求

    檢查項目：
    - 檔案可讀取
    - 長度介於 MIN_AUDIO_DURATION 到 MAX_AUDIO_DURATION 之間

    回傳驗證通過的訊息，失敗時拋出 AudioValidationError。
    """
    try:
        duration = get_audio_duration(audio_path)
    except Exception as e:
        raise AudioValidationError(f"無法讀取音頻檔案: {e}") from e

    if duration < MIN_AUDIO_DURATION:
        raise AudioValidationError(
            f"音頻長度 {duration:.1f} 秒太短，最少需要 {MIN_AUDIO_DURATION} 秒"
        )
    if duration > MAX_AUDIO_DURATION:
        raise AudioValidationError(
            f"音頻長度 {duration:.1f} 秒太長，最多 {MAX_AUDIO_DURATION} 秒"
        )

    return f"音頻驗證通過（長度 {duration:.1f} 秒）"


def preprocess_audio(audio_path: str, output_path: str) -> str:
    """前處理音頻：重取樣到目標取樣率、轉為單聲道

    回傳處理後的檔案路徑。
    """
    audio, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)

    # 正規化音量
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    sf.write(output_path, audio, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return output_path


def transcribe_audio(audio_path: str) -> str:
    """使用 mlx-whisper 將音頻轉錄為文字

    Args:
        audio_path: 音頻檔案路徑

    Returns:
        轉錄後的文字
    """
    import mlx_whisper

    logger.info("正在辨識語音內容...")
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_MODEL,
    )
    text = result.get("text", "").strip()
    language = result.get("language", "unknown")
    logger.info("語音辨識完成（語言：%s）", language)
    return text
