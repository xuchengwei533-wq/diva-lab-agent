# -*- coding: utf-8 -*-
"""
backend/asr_server.py

音频评分后端 (CAM-S)：
- 接收 base64 音频
- 提取特征
- 调用 PyTorch 模型评分
- 返回评分结果

端口：8005
"""

from __future__ import annotations

import os
import json
import base64
import tempfile
import threading
from typing import Any, Dict, Optional, Tuple, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 尽量加载 backend/.env
try:
    from dotenv import load_dotenv  # type: ignore
    _HERE = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_HERE, ".env"))
except Exception:
    pass

# 音频评分相关库
import sys
# 添加项目根目录到 Python 路径，以便导入 CAM_S 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    _numba_cache_dir = os.path.join(tempfile.gettempdir(), "numba_cache")
    os.makedirs(_numba_cache_dir, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", _numba_cache_dir)

    import numba
    try:
        cache_dir = os.environ.get("NUMBA_CACHE_DIR")
        if cache_dir and getattr(numba, "config", None):
            numba.config.CACHE_DIR = cache_dir
    except Exception:
        pass

    try:
        from numba.core.dispatcher import Dispatcher

        _orig_enable_caching = Dispatcher.enable_caching

        def _safe_enable_caching(self):
            try:
                _orig_enable_caching(self)
            except Exception:
                return

        Dispatcher.enable_caching = _safe_enable_caching
    except Exception:
        try:
            from numba.core.registry import CPUDispatcher

            _orig_enable_caching2 = CPUDispatcher.enable_caching

            def _safe_enable_caching2(self):
                try:
                    _orig_enable_caching2(self)
                except Exception:
                    return

            CPUDispatcher.enable_caching = _safe_enable_caching2
        except Exception:
            pass
except Exception:
    pass

try:
    import torch
    import librosa
    import numpy as np
    import pandas as pd
    from CAM_S import CAMPPlus
except ImportError as e:
    print(f"音频评分库导入失败: {e}")
    torch = None
    librosa = None
    np = None
    pd = None
    CAM_S = None


# =========================
# 配置
# =========================
SCORING_HOST = os.getenv("SCORING_HOST") or os.getenv("AUDIO_SCORING_HOST") or os.getenv("ASR_HOST", "0.0.0.0")
SCORING_PORT = int(os.getenv("SCORING_PORT") or os.getenv("AUDIO_SCORING_PORT") or "8005")


# =========================
# 辅助函数
# =========================
def _decode_base64_to_bytes(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except Exception:
        try:
            return base64.b64decode(b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio payload: {e}")


# =========================
# 音频评分相关模型和函数
# =========================
class AudioScoringRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    voice_type: str = Field(default="女高音 Soprano", description="Voice type for scoring")
    audio_format: str = Field(default="wav", description="Audio container/extension, e.g. wav/mp3/webm")


# 预定义的权重路径
DEFAULT_WEIGHTS = {
    "女高音 Soprano": os.path.join(project_root, "dist", "VoiceScorer", "_internal", "MODEL_WEIGHT", "logs_ddnet_sopran", "2025-08-28_17-36-59", "best_model.pth"),
    "女中音 Mezzo": os.path.join(project_root, "dist", "VoiceScorer", "_internal", "MODEL_WEIGHT", "logs_ddnet_mezzo", "2025-08-28_19-06-05", "best_model.pth"),
    "男高音 Tenor": os.path.join(project_root, "dist", "VoiceScorer", "_internal", "MODEL_WEIGHT", "logs_ddnet_tenor", "2025-08-28_19-45-18", "best_model.pth"),
    "男中音 Baritone": os.path.join(project_root, "dist", "VoiceScorer", "_internal", "MODEL_WEIGHT", "logs_ddnet_baritone", "2025-07-29_12-04-20", "best_model.pth"),
}

VOICE_TYPE_ALIASES = {
    "soprano": "女高音 Soprano",
    "mezzo": "女中音 Mezzo",
    "tenor": "男高音 Tenor",
    "baritone": "男中音 Baritone",
    "alto": "女中音 Mezzo",
    "bass": "男中音 Baritone",
    "mao": "女高音 Soprano",
}


# 技巧名称
TECH_NAMES = [
    "vibrato", "throat", "position", "open", "clean",
    "resonate", "unify", "falsetto", "chest", "nasal"
]


def _normalize_audio_ext(audio_format: str) -> str:
    ext = (audio_format or "").strip().lower()
    if ext.startswith("."):
        ext = ext[1:]
    if "/" in ext:
        ext = ext.split("/")[-1]
    if ext in ("mpeg", "mpga"):
        ext = "mp3"
    if ext in ("x-m4a",):
        ext = "m4a"
    if ext in ("wave", "x-wav"):
        ext = "wav"
    allowed = {"wav", "mp3", "webm", "m4a", "ogg", "opus"}
    return ext if ext in allowed else "wav"


def extract_mfcc_features(audio_bytes, max_pad_len=128, target_sr=44100, n_mfcc=128, audio_format: str = "wav"):
    """
    从音频字节数据中提取MFCC特征
    """
    if librosa is None or np is None:
        raise HTTPException(status_code=500, detail="音频处理库未正确导入")

    try:
        ext = _normalize_audio_ext(audio_format)
        fd, temp_path = tempfile.mkstemp(suffix=f".{ext}", prefix="scoring_", text=False)
        os.close(fd)
        
        with open(temp_path, 'wb') as f:
            f.write(audio_bytes)
        
        audio_raw, original_sr = librosa.load(temp_path, sr=None, mono=True)
        duration_sec = float(len(audio_raw) / float(original_sr)) if original_sr else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio_raw)))) if len(audio_raw) else 0.0
        if duration_sec < 0.35:
            raise HTTPException(status_code=400, detail=f"音频过短（{duration_sec:.2f}s），请重新录制")
        if rms < 1e-4:
            raise HTTPException(status_code=400, detail=f"音频能量过低（rms={rms:.2e}），疑似静音/权限异常")
        
        # 重采样到目标采样率
        if original_sr != target_sr:
            audio = librosa.resample(audio_raw, orig_sr=original_sr, target_sr=target_sr)
        else:
            audio = audio_raw

        # 计算MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc)

        # pad / truncate 到固定帧数
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        # 清理临时文件
        os.remove(temp_path)

        return mfccs.astype(np.float32), original_sr, target_sr, duration_sec, rms

    except Exception as e:
        # 确保清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"音频特征提取失败: {str(e)}")


def build_model(num_classes: int, device: torch.device):
    """
    构建评分模型
    """
    if CAMPPlus is None:
        raise HTTPException(status_code=500, detail="评分模型未正确导入")
    
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


def predict_scores(mfcc_tensor, model_path):
    """
    使用模型预测评分
    """
    if torch is None or np is None:
        raise HTTPException(status_code=500, detail="PyTorch未正确导入")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建模型
    model = build_model(50, device)  # 50类输出 (5等级 x 10技巧)
    
    # 加载模型权重
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"模型权重文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载模型权重
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
    
    # 准备输入数据
    x = mfcc_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mfcc, frames)
    
    with torch.no_grad():
        output = model(x)
        # 如果模型返回多个值，取第一个
        if isinstance(output, tuple):
            output = output[0]
    
    # 处理输出: (B, 50) -> (5, 10) -> (10,) 技巧评分
    output = output.view(output.shape[0], 5, 10)  # 5个等级 x 10个技巧
    preds = output.argmax(dim=1).cpu().numpy()[0] + 1  # 转换为1-5的评分
    
    return preds


# =========================
# FastAPI
# =========================
app = FastAPI(title="DIVA Audio Scoring Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy", "service": "AudioScoring"}


@app.post("/api/audio/score")
def score_audio(req: AudioScoringRequest) -> Dict[str, Any]:
    """
    音频评分API
    """
    if torch is None or librosa is None:
        raise HTTPException(status_code=500, detail="音频评分功能不可用，缺少必要的库")
    
    try:
        # 解码base64音频数据
        audio_bytes = _decode_base64_to_bytes(req.audio_data)
        
        # 提取MFCC特征
        mfcc_features, original_sr, target_sr, duration_sec, rms = extract_mfcc_features(audio_bytes, audio_format=req.audio_format)
        
        # 转换为PyTorch张量
        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32)
        
        # 获取模型路径
        normalized_voice_type = req.voice_type
        model_path = DEFAULT_WEIGHTS.get(normalized_voice_type)
        if not model_path:
            vt = (req.voice_type or "").strip().lower()
            for token, mapped in VOICE_TYPE_ALIASES.items():
                if token in vt:
                    normalized_voice_type = mapped
                    model_path = DEFAULT_WEIGHTS.get(mapped)
                    break
        if not model_path:
            raise HTTPException(status_code=400, detail=f"不支持的声部类型: {req.voice_type}")
        
        # 预测评分
        scores = predict_scores(mfcc_tensor, model_path)
        
        # 计算总体平均分
        overall_score = float(np.mean(scores))
        
        # 构建评分结果
        score_results = []
        for i, tech_name in enumerate(TECH_NAMES):
            score_results.append({
                "technique": tech_name,
                "score": int(scores[i])
            })
        
        return {
            "success": True,
            "message": "音频评分完成",
            "overall": overall_score,
            "scores": score_results,
            "voice_type": normalized_voice_type,
            "original_sr": original_sr,
            "target_sr": target_sr,
            "duration_sec": duration_sec,
            "rms": rms
        }
        
    except HTTPException:
        # detail 已经是前端想看到的结构
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音频评分失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print(f"Starting Audio Scoring Server on port {SCORING_PORT}...")
    uvicorn.run("asr_server:app", host=SCORING_HOST, port=SCORING_PORT, reload=False)
