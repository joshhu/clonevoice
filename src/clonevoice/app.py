"""Gradio 主介面"""

import logging
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from clonevoice.tts_engine import TTSEngine
from clonevoice.audio_utils import transcribe_audio
from clonevoice.config import LANGUAGE_OPTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# output 目錄使用專案根目錄的絕對路徑
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"

engine = TTSEngine()


def _resolve_audio_path(audio) -> str | None:
    """將 Gradio Audio 元件的值轉換為檔案路徑"""
    if audio is None:
        return None
    if isinstance(audio, str):
        return audio
    if isinstance(audio, tuple):
        sr, data = audio
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="clonevoice_upload_")
        os.close(fd)
        sf.write(path, data, sr, format="WAV")
        return path
    return None


def on_audio_change(audio) -> str:
    """當使用者上傳或錄製語音時，自動辨識文字"""
    audio_path = _resolve_audio_path(audio)
    if audio_path is None:
        return ""

    try:
        text = transcribe_audio(audio_path)
        return text
    except Exception as e:
        logger.warning("語音辨識失敗：%s", e)
        return ""


def load_voice(audio, ref_text: str) -> str:
    """載入參考語音的回呼函式"""
    if audio is None:
        return "請先上傳或錄製一段語音"

    try:
        audio_path = _resolve_audio_path(audio)
        if audio_path is None:
            return "不支援的音頻格式"

        result = engine.load_voice(audio_path, ref_text)
        return f"載入成功！{result}"
    except Exception as e:
        return f"載入失敗：{e}"


def generate_speech(text: str, language: str) -> tuple[int, np.ndarray] | None:
    """產生克隆語音的回呼函式"""
    if not text or not text.strip():
        gr.Warning("請輸入要轉換的文字")
        return None

    if not engine.voice_loaded:
        gr.Warning("請先載入參考語音")
        return None

    try:
        lang_code = LANGUAGE_OPTIONS.get(language, "auto")
        audio, sample_rate = engine.generate(text.strip(), lang_code)

        OUTPUT_DIR.mkdir(exist_ok=True)
        output_path = OUTPUT_DIR / "latest.wav"
        sf.write(str(output_path), audio, sample_rate, format="WAV")

        return (sample_rate, audio)
    except Exception as e:
        gr.Warning(f"語音生成失敗：{e}")
        logger.exception("語音生成失敗")
        return None


def build_ui() -> gr.Blocks:
    """建立 Gradio 介面"""
    with gr.Blocks(title="CloneVoice 語音克隆") as app:
        gr.Markdown("# CloneVoice 語音克隆")
        gr.Markdown(
            "上傳或錄製一段語音來記住聲音，然後輸入文字，用該聲音輸出語音。\n\n"
            "使用 Qwen3-TTS 1.7B 模型，首次執行會自動下載模型（約 3.4GB）。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 步驟一：載入聲音")
                ref_audio = gr.Audio(
                    label="參考語音（3-15 秒）",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                ref_text = gr.Textbox(
                    label="參考語音的文字稿（自動辨識，可手動修改）",
                    placeholder="上傳語音後會自動辨識文字...",
                    lines=2,
                )
                load_btn = gr.Button("載入聲音", variant="primary")
                load_status = gr.Textbox(
                    label="載入狀態", interactive=False, lines=2
                )

            with gr.Column(scale=1):
                gr.Markdown("## 步驟二：產生語音")
                input_text = gr.Textbox(
                    label="輸入要說的文字",
                    placeholder="輸入你想要用克隆語音說的話...",
                    lines=5,
                )
                language = gr.Dropdown(
                    label="語言",
                    choices=list(LANGUAGE_OPTIONS.keys()),
                    value="自動偵測",
                )
                gen_btn = gr.Button("產生語音", variant="primary")
                output_audio = gr.Audio(
                    label="產生的語音", autoplay=True, type="numpy"
                )

        # 事件綁定：上傳/錄製語音後自動辨識文字
        ref_audio.change(
            fn=on_audio_change,
            inputs=[ref_audio],
            outputs=[ref_text],
        )
        load_btn.click(
            fn=load_voice,
            inputs=[ref_audio, ref_text],
            outputs=[load_status],
        )
        gen_btn.click(
            fn=generate_speech,
            inputs=[input_text, language],
            outputs=[output_audio],
        )

    return app


def main():
    """啟動應用"""
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
