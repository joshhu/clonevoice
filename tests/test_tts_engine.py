"""tts_engine 單元測試"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from clonevoice.tts_engine import TTSEngine
from clonevoice.audio_utils import AudioValidationError


def _make_wav(duration: float = 5.0, sr: int = 24000) -> str:
    """建立測試用 WAV 檔案"""
    samples = int(duration * sr)
    data = np.random.randn(samples).astype(np.float32) * 0.5
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, data, sr, format="WAV")
    return path


class TestTTSEngineInit:
    def test_default_model_name(self):
        engine = TTSEngine()
        assert "Qwen3-TTS" in engine.model_name

    def test_custom_model_name(self):
        engine = TTSEngine(model_name="custom-model")
        assert engine.model_name == "custom-model"

    def test_voice_not_loaded_initially(self):
        engine = TTSEngine()
        assert not engine.voice_loaded

    def test_model_not_loaded_initially(self):
        engine = TTSEngine()
        assert engine.model is None


class TestLoadVoice:
    def test_load_valid_audio(self):
        engine = TTSEngine()
        path = _make_wav(5.0)
        result = engine.load_voice(path)
        assert "通過" in result
        assert engine.voice_loaded

    def test_load_with_ref_text(self):
        engine = TTSEngine()
        path = _make_wav(5.0)
        result = engine.load_voice(path, ref_text="測試文字")
        assert "ICL" in result
        assert engine.voice_loaded

    def test_load_without_ref_text(self):
        engine = TTSEngine()
        path = _make_wav(5.0)
        result = engine.load_voice(path)
        assert "Speaker Embedding" in result

    def test_load_empty_ref_text(self):
        engine = TTSEngine()
        path = _make_wav(5.0)
        result = engine.load_voice(path, ref_text="   ")
        assert "Speaker Embedding" in result

    def test_load_too_short_audio(self):
        engine = TTSEngine()
        path = _make_wav(1.0)
        with pytest.raises(AudioValidationError, match="太短"):
            engine.load_voice(path)

    def test_load_too_long_audio(self):
        engine = TTSEngine()
        path = _make_wav(20.0)
        with pytest.raises(AudioValidationError, match="太長"):
            engine.load_voice(path)


class TestGenerate:
    def test_generate_without_voice_raises(self):
        engine = TTSEngine()
        with pytest.raises(RuntimeError, match="參考語音"):
            engine.generate("test")

    @patch("clonevoice.tts_engine.TTSEngine.ensure_model_loaded")
    def test_generate_calls_model(self, mock_ensure):
        engine = TTSEngine()
        path = _make_wav(5.0)
        engine.load_voice(path, ref_text="hello")

        # 模擬 model.generate
        mock_result = MagicMock()
        mock_result.audio = np.zeros(24000, dtype=np.float32)
        mock_result.sample_rate = 24000
        mock_result.audio_duration = 1.0
        mock_result.real_time_factor = 0.5
        engine.model = MagicMock()
        engine.model.generate.return_value = [mock_result]

        audio, sr = engine.generate("測試文字", lang_code="chinese")
        assert sr == 24000
        assert len(audio) == 24000
        engine.model.generate.assert_called_once()

    @patch("clonevoice.tts_engine.TTSEngine.ensure_model_loaded")
    def test_generate_empty_result_raises(self, mock_ensure):
        engine = TTSEngine()
        path = _make_wav(5.0)
        engine.load_voice(path)

        engine.model = MagicMock()
        engine.model.generate.return_value = []

        with pytest.raises(RuntimeError, match="沒有產生結果"):
            engine.generate("test")

    @patch("clonevoice.tts_engine.TTSEngine.ensure_model_loaded")
    def test_generate_multiple_segments(self, mock_ensure):
        engine = TTSEngine()
        path = _make_wav(5.0)
        engine.load_voice(path)

        # 模擬多片段結果
        mock_result1 = MagicMock()
        mock_result1.audio = np.zeros(12000, dtype=np.float32)
        mock_result1.sample_rate = 24000
        mock_result1.audio_duration = 0.5
        mock_result1.real_time_factor = 0.5

        mock_result2 = MagicMock()
        mock_result2.audio = np.ones(12000, dtype=np.float32)
        mock_result2.sample_rate = 24000
        mock_result2.audio_duration = 0.5
        mock_result2.real_time_factor = 0.5

        engine.model = MagicMock()
        engine.model.generate.return_value = [mock_result1, mock_result2]

        audio, sr = engine.generate("測試")
        assert len(audio) == 24000  # 12000 + 12000


class TestEnsureModelLoaded:
    @patch("mlx_audio.tts.utils.load_model")
    def test_loads_model_once(self, mock_load):
        mock_load.return_value = MagicMock()
        engine = TTSEngine()
        engine.ensure_model_loaded()
        engine.ensure_model_loaded()
        mock_load.assert_called_once()

    @patch("mlx_audio.tts.utils.load_model")
    def test_uses_correct_model_name(self, mock_load):
        mock_load.return_value = MagicMock()
        engine = TTSEngine(model_name="test-model")
        engine.ensure_model_loaded()
        mock_load.assert_called_with("test-model")
