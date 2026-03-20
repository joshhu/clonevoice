"""應用程式設定"""

# 模型設定
MODEL_NAME = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

# 音頻設定
TARGET_SAMPLE_RATE = 24000
MIN_AUDIO_DURATION = 3.0  # 秒
MAX_AUDIO_DURATION = 15.0  # 秒

# 生成參數
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_TOKENS = 4096

# Whisper 語音辨識設定
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"

# 語言選項
LANGUAGE_OPTIONS = {
    "自動偵測": "auto",
    "中文": "chinese",
    "English": "english",
    "日本語": "japanese",
    "한국어": "korean",
}
