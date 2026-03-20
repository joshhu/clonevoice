"""TTS 引擎：封裝 mlx-audio Qwen3-TTS"""

import logging
import os
import tempfile

import numpy as np

from clonevoice.audio_utils import validate_audio, preprocess_audio
from clonevoice.config import (
    MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class TTSEngine:
    """Qwen3-TTS 語音克隆引擎"""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._ref_audio_path: str | None = None
        self._ref_text: str | None = None
        self._voice_loaded = False

    def ensure_model_loaded(self) -> None:
        """確保模型已載入（延遲載入）"""
        if self.model is not None:
            return
        logger.info("正在載入模型 %s ...", self.model_name)
        from mlx_audio.tts.utils import load_model
        self.model = load_model(self.model_name)
        logger.info("模型載入完成")

    def _cleanup_ref_audio(self) -> None:
        """清除舊的參考音頻暫存檔"""
        if self._ref_audio_path and os.path.exists(self._ref_audio_path):
            try:
                os.remove(self._ref_audio_path)
            except OSError:
                pass
            self._ref_audio_path = None

    def load_voice(self, audio_path: str, ref_text: str | None = None) -> str:
        """載入參考語音

        Args:
            audio_path: 參考音頻路徑
            ref_text: 參考音頻的文字稿（選填，有填品質更好）

        Returns:
            載入結果訊息
        """
        validation_msg = validate_audio(audio_path)

        # 清除舊的暫存檔
        self._cleanup_ref_audio()

        # 前處理音頻
        fd, processed_path = tempfile.mkstemp(suffix=".wav", prefix="clonevoice_ref_")
        os.close(fd)
        preprocess_audio(audio_path, processed_path)

        self._ref_audio_path = processed_path
        self._ref_text = ref_text.strip() if ref_text and ref_text.strip() else None
        self._voice_loaded = True

        mode = "ICL 模式（含文字稿）" if self._ref_text else "Speaker Embedding 模式"
        return f"{validation_msg}，使用 {mode}"

    def generate(self, text: str, lang_code: str = "auto") -> tuple[np.ndarray, int]:
        """產生克隆語音

        Args:
            text: 要轉換的文字
            lang_code: 語言代碼

        Returns:
            (audio_array, sample_rate) 元組
        """
        if not self._voice_loaded:
            raise RuntimeError("請先載入參考語音")

        self.ensure_model_loaded()

        results = list(self.model.generate(
            text=text,
            ref_audio=self._ref_audio_path,
            ref_text=self._ref_text,
            lang_code=lang_code,
            temperature=DEFAULT_TEMPERATURE,
            top_k=DEFAULT_TOP_K,
            top_p=DEFAULT_TOP_P,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            max_tokens=DEFAULT_MAX_TOKENS,
            verbose=True,
        ))

        if not results:
            raise RuntimeError("語音生成失敗，沒有產生結果")

        # 合併所有片段
        audio_arrays = [np.array(r.audio) for r in results]
        combined = np.concatenate(audio_arrays)
        sample_rate = results[0].sample_rate

        logger.info(
            "生成完成：%.1f 秒，即時因子 %.2fx",
            results[0].audio_duration,
            results[0].real_time_factor,
        )

        return combined, sample_rate

    @property
    def voice_loaded(self) -> bool:
        return self._voice_loaded

    def __del__(self):
        self._cleanup_ref_audio()
