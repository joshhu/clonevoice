# CloneVoice 語音克隆

上傳或錄製一段語音來記住聲音，然後輸入文字，用該聲音產生語音。

使用 [Qwen3-TTS 1.7B](https://huggingface.co/csukuangfj/Qwen3-TTS-12Hz-1.7B-Base-bf16) 模型，透過 [mlx-audio](https://github.com/lucasnewman/mlx-audio) 針對 Apple Silicon 原生加速。

## 功能特色

- **語音克隆**：上傳或錄製 3-15 秒的參考語音，即可克隆該聲音
- **多語言支援**：中文、English、日本語、한국어，以及自動偵測
- **語音辨識**：內建 [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)，自動辨識參考語音的文字稿，提升克隆品質
- **Apple Silicon 加速**：透過 MLX 框架在 M 系列晶片上原生執行，無需 GPU 伺服器
- **Web 介面**：基於 Gradio 提供直覺的操作介面

## 系統需求

| 項目 | 需求 |
|------|------|
| 作業系統 | macOS（Apple Silicon） |
| Python | >= 3.13 |
| RAM | 32GB（建議） |
| 外部工具 | ffmpeg |
| 磁碟空間 | 約 3.4GB（模型） |

安裝 ffmpeg：

```bash
brew install ffmpeg
```

## 安裝

```bash
uv sync
```

## 使用方式

啟動應用：

```bash
uv run clonevoice
```

開啟瀏覽器前往 http://127.0.0.1:7860

### 操作步驟

1. **上傳或錄製參考語音**（3-15 秒），系統會自動辨識語音內容
2. **（選填）修正文字稿**，準確的文字稿能提升克隆品質（ICL 模式）
3. 點擊 **「載入聲音」**
4. 輸入想要說的 **文字**
5. 選擇 **語言**，點擊 **「產生語音」**

首次執行會自動下載模型（約 3.4GB）。

## 專案結構

```
clonevoice/
├── src/clonevoice/
│   ├── __init__.py        # 套件初始化
│   ├── config.py          # 模型與音頻參數設定
│   ├── audio_utils.py     # 音頻前處理與語音辨識
│   ├── tts_engine.py      # TTS 引擎（Qwen3-TTS）
│   └── app.py             # Gradio Web 介面
├── tests/
│   ├── test_audio_utils.py
│   ├── test_tts_engine.py
│   └── test_app.py
├── pyproject.toml
└── uv.lock
```

### 模組說明

| 模組 | 說明 |
|------|------|
| `config.py` | 模型名稱、取樣率、音頻長度限制、語言選項等設定 |
| `audio_utils.py` | 音頻驗證、重取樣（24kHz）、單聲道轉換、音量正規化、Whisper 語音辨識 |
| `tts_engine.py` | TTSEngine 類別，負責載入參考語音與產生克隆語音，支援 ICL 與 Speaker Embedding 兩種模式 |
| `app.py` | Gradio 介面建構與事件綁定 |

## 技術細節

### 語音克隆模式

- **ICL 模式**（In-Context Learning）：提供參考語音及其文字稿，模型透過上下文學習克隆聲音，品質較佳
- **Speaker Embedding 模式**：僅提供參考語音，不需文字稿，適合快速使用

### 音頻前處理

參考語音會經過以下處理：
- 重取樣至 24,000 Hz
- 轉換為單聲道
- 音量正規化（peak = 0.95）

## 測試

```bash
uv run pytest --cov=clonevoice -v
```

共 44 個單元測試，覆蓋率超過 90%。

## 依賴項

### 核心

- [mlx-audio](https://github.com/lucasnewman/mlx-audio) >= 0.2.0：Qwen3-TTS 模型推論
- [gradio](https://gradio.app/) >= 5.0：Web UI
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) >= 0.4.0：語音辨識
- [librosa](https://librosa.org/) >= 0.10：音頻處理
- [soundfile](https://github.com/bastibe/python-soundfile) >= 0.13.0：音頻檔案讀寫
- [numpy](https://numpy.org/) >= 1.26：數值計算

### 開發

- pytest >= 8.0
- pytest-cov >= 6.0

## 授權

MIT License
