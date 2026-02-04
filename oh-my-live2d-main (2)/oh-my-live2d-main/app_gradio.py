"""
Voice Scoring GUI (Gradio) - Aligned with VAL_WIN
-------------------------------------------------
Goals:
1) GUI 不显示模型路径
2) 运行自动打开网页
3) 推理特征链路尽量与 MFCCnew + VAL_WIN 对齐：
   - 使用 gr.Audio(type="filepath")
   - 使用 librosa.load(sr=None, mono=True) 读取文件
   - 重采样到 44100
   - 提取 128 维 MFCC
   - pad/truncate 到 128 帧
   - 模型输出 50 类 -> reshape (5,10) -> 10 个技巧 1-5 分

You still need:
- Your model definition module that provides CAMPPlus (e.g., CAM_S.py)
- Weight files for 4 parts

Run:
    pip install torch librosa gradio numpy pandas matplotlib soundfile
    python app_gradio.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import librosa
import torch
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

# ---- Import your model definition ----
# If your module name differs, change this import.
from CAM_S import CAMPPlus
import sys

def resource_path(rel_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, rel_path)

# -----------------------------
# Configuration
# -----------------------------
TECH_NAMES: List[str] = [
    "vibrato", "throat", "position", "open", "clean",
    "resonate", "unify", "falsetto", "chset", "nasal"
]

NUM_CLASSES = 50  # 5 levels x 10 techniques
TARGET_SR = 44100
N_MFCC = 128
MAX_PAD_LEN = 128

# EDIT THESE TO REAL PATHS (Windows)
DEFAULT_WEIGHTS = {
    "女高音 Soprano":   r"MODEL_WEIGHT\logs_ddnet_sopran\2025-08-28_17-36-59\best_model.pth",
    "女中音 Mezzo":    r"MODEL_WEIGHT\logs_ddnet_mezzo\2025-08-28_19-06-05\best_model.pth",
    "男高音 Tenor":    r"MODEL_WEIGHT\logs_ddnet_tenor\2025-08-28_19-45-18\best_model.pth",
    "男中音 Baritone": r"MODEL_WEIGHT\logs_ddnet_baritone\2025-07-29_12-04-20\best_model.pth",
}


@dataclass
class ModelBundle:
    model: torch.nn.Module
    device: torch.device
    weight_path: str


_MODEL_CACHE: Dict[str, ModelBundle] = {}


# -----------------------------
# MFCC extraction (file-path based)
# -----------------------------
def extract_mfcc_from_filepath(
    file_path: str,
    max_pad_len: int = MAX_PAD_LEN,
    target_sr: int = TARGET_SR,
    n_mfcc: int = N_MFCC,
) -> Tuple[np.ndarray, int, int]:
    """
    Match MFCCnew-style pipeline as closely as possible:
    - librosa.load(sr=None, mono=True)
    - resample to 44100
    - MFCC 128
    - pad / truncate to 128 frames
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError("音频文件不存在或读取失败。")

    # Load with original sr and mono like MFCCnew
    audio_raw, original_sr = librosa.load(file_path, sr=None, mono=True)
    if audio_raw is None or len(audio_raw) == 0:
        raise ValueError("音频为空或无法解码。")

    original_sr = int(original_sr)

    # Resample to target
    if original_sr != target_sr:
        audio = librosa.resample(y=audio_raw, orig_sr=original_sr, target_sr=target_sr)
    else:
        audio = audio_raw

    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc)

    # Pad / truncate frames
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs.astype(np.float32), original_sr, target_sr


def mfcc_to_tensor(mfcc: np.ndarray) -> torch.Tensor:
    """
    (n_mfcc, frames) -> (1, n_mfcc, frames)
    Later we will add batch dimension -> (1,1,n_mfcc,frames)
    """
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)


# -----------------------------
# Model building / loading
# -----------------------------
def build_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    model = CAMPPlus(
        num_class=num_classes,
        input_size=1,
        embd_dim=8192,
        growth_rate=64,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
    ).to(device)
    return model


def resolve_weight_path(voice: str) -> str:
    weight_path = DEFAULT_WEIGHTS.get(voice, "")
    if not weight_path:
        raise ValueError(f"该声部尚未配置权重：{voice}")
    return weight_path


def load_model_cached(voice: str) -> ModelBundle:
    weight_path = resolve_weight_path(voice)

    # cache hit
    if voice in _MODEL_CACHE and _MODEL_CACHE[voice].weight_path == weight_path:
        return _MODEL_CACHE[voice]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(NUM_CLASSES, device)

    if not os.path.exists(weight_path):
        # Do not show full path to user
        raise FileNotFoundError(
            f"未找到 {voice} 的权重文件。"
            f"请检查脚本 DEFAULT_WEIGHTS 中该声部的路径配置。"
        )

    checkpoint = torch.load(weight_path, map_location=device)

    # Robust loading for common checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    bundle = ModelBundle(model=model, device=device, weight_path=weight_path)
    _MODEL_CACHE[voice] = bundle
    return bundle


# -----------------------------
# Inference
# -----------------------------
def predict_scores(mfcc_tensor: torch.Tensor, bundle: ModelBundle) -> np.ndarray:
    """
    Input:
        mfcc_tensor: (1, n_mfcc, frames)
    Model input expected:
        (B, 1, n_mfcc, frames)
    """
    x = mfcc_tensor.unsqueeze(0).to(bundle.device)  # -> (1,1,128,128)

    with torch.no_grad():
        output, *_ = bundle.model(x)

    # output: (B, 50)
    output = output.view(output.shape[0], 5, 10)
    preds = output.argmax(dim=1).cpu().numpy() + 1  # 1..5
    return preds[0]


def make_bar_plot(scores: np.ndarray):
    fig = plt.figure(figsize=(9, 4.2))
    x = np.arange(len(TECH_NAMES))
    plt.bar(x, scores)
    plt.xticks(x, TECH_NAMES, rotation=30, ha="right")
    plt.ylim(0, 5.5)
    plt.ylabel("Score (1-5)")
    plt.title("Technique Scores")
    plt.tight_layout()
    return fig


def score_one(
    audio_path: Optional[str],
    voice: str,
):
    """
    Gradio callback
    audio_path: file path returned by gr.Audio(type="filepath")
    """
    if not audio_path:
        return (
            "请先录音或上传音频。",
            None,
            None,
            None,
        )

    try:
        mfcc, orig_sr, tgt_sr = extract_mfcc_from_filepath(audio_path)
        mfcc_tensor = mfcc_to_tensor(mfcc)

        bundle = load_model_cached(voice)
        scores = predict_scores(mfcc_tensor, bundle)

        df = pd.DataFrame({"technique": TECH_NAMES, "score": scores})
        overall = float(np.mean(scores))
        fig = make_bar_plot(scores)

        info = (
            f"✅ 推理完成\n"
            f"- 声部: {voice}\n"
            f"- 原采样率: {orig_sr} Hz\n"
            f"- 重采样: {tgt_sr} Hz\n"
            f"- 总体均分: {overall:.2f}/5"
        )

        return info, df, fig, overall

    except Exception as e:
        return f"❌ 发生错误: {e}", None, None, None


# -----------------------------
# UI
# -----------------------------
def build_app():
    with gr.Blocks(title="四声部演唱技巧自动评分") as demo:
        gr.Markdown(
            "### 四声部演唱技巧自动评分\n"
            "选择声部 → 录音或上传音频 → 模型推理 → 可视化评分\n\n"
            "（该版本使用文件路径提特征，以尽量对齐 VAL_WIN 的推理结果）"
        )

        voice = gr.Dropdown(
            choices=list(DEFAULT_WEIGHTS.keys()),
            value="女高音 Soprano",
            label="选择声部",
        )

        # Key change: filepath to align feature pipeline
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="录音或上传 WAV/MP3",
        )

        with gr.Row():
            btn = gr.Button("开始评分", variant="primary")
            clear = gr.Button("清空")

        status = gr.Textbox(label="状态")
        df_out = gr.Dataframe(label="分项得分", interactive=False)
        plot_out = gr.Plot(label="得分可视化")
        overall_out = gr.Number(label="总体均分 (1-5)", precision=2)

        btn.click(
            fn=score_one,
            inputs=[audio, voice],
            outputs=[status, df_out, plot_out, overall_out],
        )

        clear.click(
            fn=lambda: (None, "女高音 Soprano", "", None, None, None),
            inputs=[],
            outputs=[audio, voice, status, df_out, plot_out, overall_out],
        )

        gr.Markdown(
            "#### 说明\n"
            "- 权重路径已隐藏于界面，仅按声部自动加载。\n"
            "- 若报“未找到权重文件”，请修改脚本顶部 DEFAULT_WEIGHTS。\n"
            "- 若仍与 VAL_WIN 不一致，优先检查训练/验证时 MFCC 参数是否完全相同。"
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        prevent_thread_lock=False,  # ✅ 强制锁线程，避免 exe 启动后立即退出
    )
