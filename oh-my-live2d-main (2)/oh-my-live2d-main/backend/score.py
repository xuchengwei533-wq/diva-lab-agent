# -*- coding: utf-8 -*-
"""
backend/score.py

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

try:
    from dotenv import load_dotenv  # type: ignore
    _HERE = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_HERE, ".env"))
except Exception:
    pass

import sys

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

torch = None
librosa = None
np = None
pd = None
CAMPPlus = None

try:
    import torch  # type: ignore
except Exception as e:
    print(f"音频评分库导入失败(torch): {e}")

try:
    import librosa  # type: ignore
except Exception as e:
    print(f"音频评分库导入失败(librosa): {e}")

try:
    import numpy as np  # type: ignore
except Exception as e:
    print(f"音频评分库导入失败(numpy): {e}")

try:
    import pandas as pd  # type: ignore
except Exception as e:
    print(f"音频评分库导入失败(pandas): {e}")

if torch is not None:
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    except Exception:
        pass

try:
    from CAM_S import CAMPPlus  # type: ignore
except Exception as e:
    print(f"音频评分库导入失败(CAM_S): {e}")


SCORING_HOST = os.getenv("SCORING_HOST") or os.getenv("AUDIO_SCORING_HOST") or os.getenv("ASR_HOST", "0.0.0.0")
SCORING_PORT = int(os.getenv("SCORING_PORT") or os.getenv("AUDIO_SCORING_PORT") or "8005")


def _decode_base64_to_bytes(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except Exception:
        try:
            return base64.b64decode(b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio payload: {e}")


class AudioScoringRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    voice_type: str = Field(default="女高音 Soprano", description="Voice type for scoring")
    audio_format: str = Field(default="wav", description="Audio container/extension, e.g. wav/mp3/webm")


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
    if librosa is None or np is None:
        raise HTTPException(status_code=500, detail="音频处理库未正确导入")

    try:
        ext = _normalize_audio_ext(audio_format)
        fd, temp_path = tempfile.mkstemp(suffix=f".{ext}", prefix="scoring_", text=False)
        os.close(fd)

        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        audio_raw, original_sr = librosa.load(temp_path, sr=None, mono=True)
        duration_sec = float(len(audio_raw) / float(original_sr)) if original_sr else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio_raw)))) if len(audio_raw) else 0.0
        if duration_sec < 0.35:
            raise HTTPException(status_code=400, detail=f"音频过短（{duration_sec:.2f}s），请重新录制")
        if rms < 1e-4:
            raise HTTPException(status_code=400, detail=f"音频能量过低（rms={rms:.2e}），疑似静音/权限异常")

        if original_sr != target_sr:
            audio = librosa.resample(audio_raw, orig_sr=original_sr, target_sr=target_sr)
        else:
            audio = audio_raw

        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc)

        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfccs = mfccs[:, :max_pad_len]

        os.remove(temp_path)

        return mfccs.astype(np.float32), original_sr, target_sr, duration_sec, rms

    except Exception as e:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"音频特征提取失败: {str(e)}")


def build_model(num_classes: int, device: torch.device):
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
    if torch is None or np is None:
        raise HTTPException(status_code=500, detail="PyTorch未正确导入")

    device_name = (os.getenv("SCORING_DEVICE") or "cpu").strip().lower()
    if device_name != "cpu":
        device_name = "cpu"
    device = torch.device(device_name)

    model = build_model(50, device)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"模型权重文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

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

    x = mfcc_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]

    output = output.view(output.shape[0], 5, 10)
    preds = output.argmax(dim=1).cpu().numpy()[0] + 1

    return preds


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
    torch_ver = None
    torch_cuda = None
    try:
        if torch is not None:
            torch_ver = getattr(torch, "__version__", None)
            torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "AudioScoring",
        "deps": {
            "torch": bool(torch),
            "librosa": bool(librosa),
            "numpy": bool(np),
            "pandas": bool(pd),
            "cam_s": bool(CAMPPlus),
            "torch_version": torch_ver,
            "torch_cuda": torch_cuda,
        },
        "device": (os.getenv("SCORING_DEVICE") or "cpu").strip().lower() or "cpu",
    }


@app.post("/api/audio/score")
def score_audio(req: AudioScoringRequest) -> Dict[str, Any]:
    if torch is None or librosa is None:
        raise HTTPException(status_code=500, detail="音频评分功能不可用，缺少必要的库")

    try:
        audio_bytes = _decode_base64_to_bytes(req.audio_data)

        mfcc_features, original_sr, target_sr, duration_sec, rms = extract_mfcc_features(audio_bytes, audio_format=req.audio_format)

        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32)

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

        scores = predict_scores(mfcc_tensor, model_path)

        overall_score = float(np.mean(scores))

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
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音频评分失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print(f"Starting Audio Scoring Server on port {SCORING_PORT}...")
    uvicorn.run("score:app", host=SCORING_HOST, port=SCORING_PORT, reload=False)
