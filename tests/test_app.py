"""app 單元測試"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from clonevoice.app import load_voice, generate_speech, on_audio_change, build_ui


def _make_wav(duration: float = 5.0, sr: int = 24000) -> str:
    """建立測試用 WAV 檔案"""
    samples = int(duration * sr)
    data = np.random.randn(samples).astype(np.float32) * 0.5
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, data, sr, format="WAV")
    return path


class TestLoadVoice:
    def test_none_audio(self):
        result = load_voice(None, "")
        assert "請先上傳" in result

    @patch("clonevoice.app.engine")
    def test_filepath_input(self, mock_engine):
        mock_engine.load_voice.return_value = "音頻驗證通過（長度 5.0 秒），使用 ICL 模式（含文字稿）"
        path = _make_wav(5.0)
        result = load_voice(path, "測試")
        assert "載入成功" in result

    @patch("clonevoice.app.engine")
    def test_tuple_input(self, mock_engine):
        mock_engine.load_voice.return_value = "音頻驗證通過（長度 5.0 秒），使用 Speaker Embedding 模式"
        sr = 24000
        data = np.random.randn(sr * 5).astype(np.float32)
        result = load_voice((sr, data), "")
        assert "載入成功" in result

    @patch("clonevoice.app.engine")
    def test_load_failure(self, mock_engine):
        mock_engine.load_voice.side_effect = Exception("test error")
        path = _make_wav(5.0)
        result = load_voice(path, "")
        assert "載入失敗" in result


class TestGenerateSpeech:
    def test_empty_text(self):
        result = generate_speech("", "自動偵測")
        assert result is None

    def test_whitespace_text(self):
        result = generate_speech("   ", "自動偵測")
        assert result is None

    @patch("clonevoice.app.engine")
    def test_no_voice_loaded(self, mock_engine):
        mock_engine.voice_loaded = False
        result = generate_speech("測試文字", "自動偵測")
        assert result is None

    @patch("clonevoice.app.engine")
    def test_successful_generation(self, mock_engine):
        mock_engine.voice_loaded = True
        mock_engine.generate.return_value = (np.zeros(24000), 24000)
        result = generate_speech("測試", "中文")
        assert result is not None
        sr, audio = result
        assert sr == 24000

    @patch("clonevoice.app.engine")
    def test_generation_failure(self, mock_engine):
        mock_engine.voice_loaded = True
        mock_engine.generate.side_effect = RuntimeError("生成失敗")
        result = generate_speech("測試", "中文")
        assert result is None


class TestOnAudioChange:
    @patch("clonevoice.app.transcribe_audio")
    def test_returns_transcribed_text(self, mock_transcribe):
        mock_transcribe.return_value = "你好世界"
        path = _make_wav(5.0)
        result = on_audio_change(path)
        assert result == "你好世界"

    def test_none_audio_returns_empty(self):
        result = on_audio_change(None)
        assert result == ""

    @patch("clonevoice.app.transcribe_audio")
    def test_transcribe_failure_returns_empty(self, mock_transcribe):
        mock_transcribe.side_effect = RuntimeError("whisper error")
        path = _make_wav(5.0)
        result = on_audio_change(path)
        assert result == ""


class TestBuildUI:
    def test_ui_builds_successfully(self):
        app = build_ui()
        assert app is not None
