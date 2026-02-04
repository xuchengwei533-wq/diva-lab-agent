# -*- coding: utf-8 -*-
"""
backend/asr_new.py

独立的语音识别服务模块：
- 接收 base64 PCM16LE 16kHz mono
- 调用 DashScope ASR (支持 qwen3 / paraformer)
- 返回文本用于 TTS 和 LLM 输入

端口：8006
"""

from __future__ import annotations

import os
import json
import base64
import wave
import tempfile
import threading
import inspect
import sys
import math
from array import array
from typing import Any, Dict, Optional, Tuple, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 尽量加载 backend/.env
try:
    from dotenv import load_dotenv
    _HERE = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_HERE, ".env"))
except Exception:
    pass

# DashScope
try:
    import dashscope
    from dashscope.audio.asr import Recognition
except Exception:
    dashscope = None
    Recognition = None

# =========================
# 配置
# =========================
ASR_HOST = os.getenv("ASR_HOST", "0.0.0.0")
ASR_PORT = 8006  # 新端口

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASH_SCOPE_API_KEY")
DASHSCOPE_ASR_MODEL = os.getenv("DASHSCOPE_ASR_MODEL", "qwen3-asr-flash")
DASHSCOPE_ASR_FALLBACK_MODEL = os.getenv("DASHSCOPE_ASR_FALLBACK_MODEL", "paraformer-realtime-v1")

PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH_BYTES = 2  # int16


# =========================
# 全局状态（用于 /api/voice/text 轮询）
# =========================
_state_lock = threading.Lock()
_last_text: str = ""
_has_result: bool = False
_last_error: str = ""


def _set_result(text: str) -> None:
    global _last_text, _has_result, _last_error
    with _state_lock:
        _last_text = text or ""
        _has_result = bool(text and text.strip())
        _last_error = ""


def _set_error(err: str) -> None:
    global _last_text, _has_result, _last_error
    with _state_lock:
        _last_text = ""
        _has_result = False
        _last_error = err or ""


def _clear_state() -> None:
    global _last_text, _has_result, _last_error
    with _state_lock:
        _last_text = ""
        _has_result = False
        _last_error = ""


# =========================
# 请求体
# =========================
class VoiceStartRequest(BaseModel):
    audio_data: Optional[str] = Field(default=None, description="Base64 of PCM bytes")
    audio_base64: Optional[str] = Field(default=None, description="Base64 of PCM bytes")
    audio_format: Optional[str] = Field(default="pcm16le_16k_mono")


# =========================
# PCM(base64) -> WAV
# =========================
def _decode_base64_to_bytes(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except Exception:
        try:
            return base64.b64decode(b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio payload: {e}")


def _pcm_bytes_to_wav_file(pcm_bytes: bytes) -> str:
    if not pcm_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    if len(pcm_bytes) % PCM_SAMPLE_WIDTH_BYTES != 0:
        raise HTTPException(
            status_code=400,
            detail=f"PCM byte length not aligned to int16: len={len(pcm_bytes)}",
        )

    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="asr_", text=False)
    os.close(fd)

    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(PCM_CHANNELS)
            wf.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
            wf.setframerate(PCM_SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
    except Exception as e:
        try:
            os.remove(wav_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to write wav: {e}")

    return wav_path


def _pcm_stats(pcm_bytes: bytes) -> Dict[str, Any]:
    byte_len = len(pcm_bytes or b"")
    if byte_len == 0:
        return {
            "byte_len": 0,
            "samples": 0,
            "duration_sec": 0.0,
            "rms": 0.0,
            "peak": 0,
            "mean_abs": 0.0,
            "zero_frac": 1.0,
            "dc": 0.0,
        }

    samples = byte_len // 2
    duration_sec = float(samples) / float(PCM_SAMPLE_RATE) if PCM_SAMPLE_RATE else 0.0

    a = array("h")
    a.frombytes(pcm_bytes[: samples * 2])
    if sys.byteorder != "little":
        a.byteswap()

    if not a:
        return {
            "byte_len": byte_len,
            "samples": 0,
            "duration_sec": 0.0,
            "rms": 0.0,
            "peak": 0,
            "mean_abs": 0.0,
            "zero_frac": 1.0,
            "dc": 0.0,
        }

    peak = 0
    sum_sq = 0.0
    sum_abs = 0.0
    sum_val = 0.0
    zero_cnt = 0
    for v in a:
        iv = int(v)
        if iv == 0:
            zero_cnt += 1
        av = -iv if iv < 0 else iv
        if av > peak:
            peak = av
        sum_sq += float(iv) * float(iv)
        sum_abs += float(av)
        sum_val += float(iv)

    n = float(len(a))
    rms = math.sqrt(sum_sq / n) if n else 0.0
    mean_abs = sum_abs / n if n else 0.0
    dc = sum_val / n if n else 0.0
    zero_frac = float(zero_cnt) / n if n else 1.0

    return {
        "byte_len": byte_len,
        "samples": int(len(a)),
        "duration_sec": float(duration_sec),
        "rms": float(rms),
        "peak": int(peak),
        "mean_abs": float(mean_abs),
        "zero_frac": float(zero_frac),
        "dc": float(dc),
    }


# =========================
# DashScope 响应解析
# =========================
def _to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    try:
        return json.loads(json.dumps(obj, default=lambda x: getattr(x, "__dict__", str(x))))
    except Exception:
        return str(obj)


def _extract_text_from_resp(resp: Any) -> str:
    data = _to_dict(resp)

    candidates = [
        ("text",),
        ("result",),
        ("output", "text"),
        ("output", "sentence", "text"),
        ("output", "sentence"),
        ("output", "transcription"),
        ("output", "transcript"),
        ("output", "transcripts"),
        ("output", "result"),
        ("results",),
        ("output", "results"),
        ("output", "sentences"),
        ("output", "choices"),
    ]

    def get_path(d: Any, path: Tuple[str, ...]) -> Any:
        cur = d
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur

    for p in candidates:
        v = get_path(data, p)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for lst in (
        get_path(data, ("output", "sentence")),
        get_path(data, ("output", "results")),
        get_path(data, ("results",)),
        get_path(data, ("output", "sentences")),
        get_path(data, ("output", "choices")),
    ):
        if isinstance(lst, list) and lst:
            first = lst[0]
            if isinstance(first, dict):
                for key in ("text", "sentence", "transcription", "result"):
                    if key in first and isinstance(first[key], str) and first[key].strip():
                        return first[key].strip()

    return ""


# =========================
# DashScope ASR
# =========================
def _dashscope_asr(wav_path: str) -> Tuple[str, Dict[str, Any]]:
    if Recognition is None or dashscope is None:
        raise HTTPException(status_code=500, detail="dashscope not installed. pip install dashscope")

    if not DASHSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY not set in environment")

    dashscope.api_key = DASHSCOPE_API_KEY

    if not os.path.exists(wav_path):
        raise HTTPException(status_code=500, detail=f"WAV file not found: {wav_path}")

    last_err: Optional[Exception] = None
    had_successful_call = False
    last_meta: Dict[str, Any] = {}

    model_candidates: list[str] = []
    for m in ("paraformer-realtime-v1", DASHSCOPE_ASR_MODEL, DASHSCOPE_ASR_FALLBACK_MODEL):
        m = (m or "").strip()
        if not m:
            continue
        if m not in model_candidates:
            model_candidates.append(m)

    print(f"[ASR] Candidates: {model_candidates}")

    for model_name in model_candidates:
        print(f"[ASR] Trying model: {model_name}")
        
        if "filetrans" in (model_name or "").lower():
            print(f"[ASR] Skipping {model_name} (Transcription not supported for local file)")
            continue

        try:
            # 1. Try Instance Method (Most likely for this SDK version)
            recog = None
            try:
                recog = Recognition(
                    model=model_name,
                    callback=None,
                    format="wav",
                    sample_rate=PCM_SAMPLE_RATE,
                )
            except Exception:
                pass

            if recog:
                resp = recog.call(file=wav_path)
            else:
                # 2. Try Class Method (Newer SDK standard)
                resp = Recognition.call(model=model_name, file=wav_path, format="wav", sample_rate=PCM_SAMPLE_RATE)

            # Validate response
            d = _to_dict(resp)
            meta = {}
            if isinstance(d, dict):
                meta = {
                    "model": model_name,
                    "status_code": d.get("status_code"),
                    "code": d.get("code"),
                    "message": d.get("message"),
                    "request_id": d.get("request_id"),
                }
            else:
                meta = {"model": model_name}
            last_meta = meta

            if isinstance(d, dict):
                code = d.get("code")
                msg = d.get("message")
                if code == "ModelNotFound" or (isinstance(d.get("status_code"), int) and d.get("status_code") == 44):
                    print(f"[ASR] Model {model_name} not found (Code 44/ModelNotFound).")
                    last_err = RuntimeError(f"Model {model_name} not found: {msg}")
                    continue

            text = _extract_text_from_resp(resp)
            if text.strip():
                print(f"[ASR] Success: {text[:20]}...")
                return text, meta

            print(f"[ASR] Empty result from {model_name}. Resp: {_to_dict(resp)}")
            had_successful_call = True
            last_err = RuntimeError(f"Empty ASR result from {model_name}")
            continue

        except Exception as e:
            print(f"[ASR] Failed with {model_name}: {e}")
            last_err = e
            # Try next model

    if had_successful_call:
        return "", last_meta

    raise HTTPException(status_code=500, detail=f"All ASR models failed. Last error: {last_err}")


# =========================
# FastAPI
# =========================
app = FastAPI(title="DIVA ASR Service (New)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy", "service": "ASR-New"}


@app.post("/api/voice/clear")
def clear_voice() -> Dict[str, Any]:
    _clear_state()
    return {"success": True, "message": "语音识别结果已清除", "text": "", "has_result": False}


@app.get("/api/voice/text")
def get_voice_text() -> Dict[str, Any]:
    with _state_lock:
        if _last_error:
            return {"success": False, "message": _last_error, "text": "", "has_result": False}
        return {"success": True, "message": "ok", "text": _last_text, "has_result": _has_result}


@app.post("/api/voice/start")
def start_voice(req: VoiceStartRequest) -> Dict[str, Any]:
    b64 = req.audio_data or req.audio_base64
    if not b64:
        raise HTTPException(status_code=400, detail="Missing audio_data/audio_base64")

    fmt = (req.audio_format or "").lower()
    if fmt not in ("pcm16le_16k_mono", "pcm16le", "pcm16"):
        raise HTTPException(status_code=400, detail=f"Unsupported audio_format: {req.audio_format}")

    try:
        pcm_bytes = _decode_base64_to_bytes(b64)
        audio_info = _pcm_stats(pcm_bytes)

        if audio_info.get("duration_sec", 0.0) < 0.15:
            _set_result("")
            return {
                "success": True,
                "message": "语音识别完成，但音频过短",
                "text": "",
                "has_result": False,
                "debug": {"audio": audio_info, "asr": {}},
            }

        if audio_info.get("peak", 0) <= 80 and audio_info.get("rms", 0.0) <= 40 and audio_info.get("zero_frac", 1.0) >= 0.98:
            _set_result("")
            return {
                "success": True,
                "message": "语音识别完成，但音频能量过低（疑似未录到声音）",
                "text": "",
                "has_result": False,
                "debug": {"audio": audio_info, "asr": {}},
            }

        wav_path = _pcm_bytes_to_wav_file(pcm_bytes)

        try:
            text, asr_meta = _dashscope_asr(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

        text = (text or "").strip()
        _set_result(text)

        if text:
            return {
                "success": True,
                "message": "语音识别完成",
                "text": text,
                "has_result": True,
                "debug": {"audio": audio_info, "asr": asr_meta},
            }

        return {
            "success": True,
            "message": "语音识别完成，但未识别到文本",
            "text": "",
            "has_result": False,
            "debug": {"audio": audio_info, "asr": asr_meta},
        }

    except HTTPException:
        raise
    except Exception as e:
        _set_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print(f"Starting ASR Service on port {ASR_PORT}...")
    uvicorn.run("asr_new:app", host=ASR_HOST, port=ASR_PORT, reload=False)
