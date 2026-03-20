"""audio_utils 單元測試"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from clonevoice.audio_utils import (
    AudioValidationError,
    get_audio_duration,
    preprocess_audio,
    transcribe_audio,
    validate_audio,
)
from clonevoice.config import TARGET_SAMPLE_RATE


def _make_wav(duration: float, sr: int = 44100, channels: int = 1) -> str:
    """建立測試用 WAV 檔案"""
    samples = int(duration * sr)
    if channels == 1:
        data = np.random.randn(samples).astype(np.float32) * 0.5
    else:
        data = np.random.randn(samples, channels).astype(np.float32) * 0.5
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, data, sr, format="WAV")
    return path


def _make_output_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return path


class TestGetAudioDuration:
    def test_correct_duration(self):
        path = _make_wav(5.0)
        duration = get_audio_duration(path)
        assert abs(duration - 5.0) < 0.1

    def test_short_audio(self):
        path = _make_wav(1.0)
        duration = get_audio_duration(path)
        assert abs(duration - 1.0) < 0.1

    def test_nonexistent_file(self):
        with pytest.raises(Exception):
            get_audio_duration("/nonexistent/file.wav")


class TestValidateAudio:
    def test_valid_audio(self):
        path = _make_wav(5.0)
        result = validate_audio(path)
        assert "驗證通過" in result

    def test_too_short(self):
        path = _make_wav(1.0)
        with pytest.raises(AudioValidationError, match="太短"):
            validate_audio(path)

    def test_too_long(self):
        path = _make_wav(20.0)
        with pytest.raises(AudioValidationError, match="太長"):
            validate_audio(path)

    def test_min_boundary(self):
        path = _make_wav(3.0)
        result = validate_audio(path)
        assert "驗證通過" in result

    def test_max_boundary(self):
        path = _make_wav(15.0)
        result = validate_audio(path)
        assert "驗證通過" in result

    def test_invalid_file(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        Path(path).write_text("not audio")
        with pytest.raises(AudioValidationError, match="無法讀取"):
            validate_audio(path)


class TestPreprocessAudio:
    def test_resample_to_target(self):
        path = _make_wav(5.0, sr=44100)
        output = _make_output_path()
        preprocess_audio(path, output)
        info = sf.info(output)
        assert info.samplerate == TARGET_SAMPLE_RATE

    def test_stereo_to_mono(self):
        path = _make_wav(5.0, sr=44100, channels=2)
        output = _make_output_path()
        preprocess_audio(path, output)
        info = sf.info(output)
        assert info.channels == 1

    def test_output_file_exists(self):
        path = _make_wav(5.0)
        output = _make_output_path()
        result = preprocess_audio(path, output)
        assert Path(result).exists()

    def test_normalized_volume(self):
        path = _make_wav(5.0)
        output = _make_output_path()
        preprocess_audio(path, output)
        data, _ = sf.read(output)
        peak = np.max(np.abs(data))
        assert peak <= 1.0
        assert peak > 0.85

    def test_already_target_rate(self):
        path = _make_wav(5.0, sr=TARGET_SAMPLE_RATE)
        output = _make_output_path()
        preprocess_audio(path, output)
        info = sf.info(output)
        assert info.samplerate == TARGET_SAMPLE_RATE

    def test_silent_audio(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, np.zeros(TARGET_SAMPLE_RATE * 5), TARGET_SAMPLE_RATE)
        output = _make_output_path()
        preprocess_audio(path, output)
        data, _ = sf.read(output)
        assert np.max(np.abs(data)) == 0.0


class TestTranscribeAudio:
    @patch("mlx_whisper.transcribe")
    def test_transcribe_returns_text(self, mock_transcribe):
        mock_transcribe.return_value = {
            "text": " 你好世界 ",
            "language": "zh",
        }
        path = _make_wav(5.0)
        result = transcribe_audio(path)
        assert result == "你好世界"
        mock_transcribe.assert_called_once()

    @patch("mlx_whisper.transcribe")
    def test_transcribe_empty_result(self, mock_transcribe):
        mock_transcribe.return_value = {"text": "", "language": "en"}
        path = _make_wav(5.0)
        result = transcribe_audio(path)
        assert result == ""

    @patch("mlx_whisper.transcribe")
    def test_transcribe_uses_correct_model(self, mock_transcribe):
        mock_transcribe.return_value = {"text": "hello", "language": "en"}
        path = _make_wav(5.0)
        transcribe_audio(path)
        call_kwargs = mock_transcribe.call_args
        assert "whisper-large-v3-turbo" in call_kwargs.kwargs["path_or_hf_repo"]
