from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import random
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from array import array
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import dashscope
    from dashscope import Application
    from dashscope.audio.asr import Recognition
    from dashscope.audio.qwen_tts import SpeechSynthesizer
except Exception:  # pragma: no cover - surfaced as a health/config error at runtime
    dashscope = None
    Application = None
    Recognition = None
    SpeechSynthesizer = None


BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
WEB_DIR = BASE_DIR / "web"
OPENING_DIR = ASSETS_DIR / "opening"

load_dotenv(BASE_DIR / ".env", override=False)

SYSTEM_PROMPT = (
    "你是“肖斯塔科维奇”。你以第一人称与用户对话，开场可说“我是肖斯塔科维奇，欢迎走进我的音乐世界”。"
    "你的气质深沉、克制、严肃而温和，像一位经历过战争与时代风暴的作曲家。"
    "你擅长从音乐情绪、节奏、音色、呼吸、乐句和表现力角度指导用户。"
    "回答必须简短、具体、可执行，避免空话，每次不超过100个汉字。"
)

OPENING_ASSISTANT_TEXT = "我是肖斯塔科维奇，欢迎走进我的故事"
PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH_BYTES = 2


def env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


DASHSCOPE_API_KEY = env("DASHSCOPE_API_KEY") or env("DASH_SCOPE_API_KEY")
BAILIAN_APP_ID = env("BAILIAN_APP_ID", "4dc0700043fc46679e1568339e580678")
DASHSCOPE_ASR_MODEL = env("DASHSCOPE_ASR_MODEL", "qwen3-asr-flash")
DASHSCOPE_ASR_FALLBACK_MODEL = env("DASHSCOPE_ASR_FALLBACK_MODEL", "paraformer-realtime-v1")
DASHSCOPE_TTS_MODEL = env("DASHSCOPE_TTS_MODEL", "qwen3-tts-flash")
DASHSCOPE_TTS_VOICE = env("DASHSCOPE_TTS_VOICE", "Ethan")


class TurnRequest(BaseModel):
    audio_data: str = Field(description="Base64 encoded PCM16LE 16kHz mono audio")
    audio_format: str = Field(default="pcm16le_16k_mono")
    session_id: Optional[str] = None
    language: Optional[str] = None


class TextRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    language: Optional[str] = None


app = FastAPI(title="Xiaosita Voice Avatar", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


def require_dashscope() -> None:
    if not DASHSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY is not configured")
    if dashscope is None:
        raise HTTPException(status_code=500, detail="dashscope package is not installed")
    dashscope.api_key = DASHSCOPE_API_KEY


def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def to_dict(obj: Any) -> Any:
    if obj is None or isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    try:
        return json.loads(json.dumps(obj, default=lambda item: getattr(item, "__dict__", str(item))))
    except Exception:
        return str(obj)


def decode_audio_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        try:
            return base64.b64decode(value)
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio payload: {error}") from error


def pcm_stats(pcm_bytes: bytes) -> Dict[str, Any]:
    byte_len = len(pcm_bytes or b"")
    samples = byte_len // 2
    if samples <= 0:
        return {"byte_len": byte_len, "duration_sec": 0.0, "rms": 0.0, "peak": 0, "zero_frac": 1.0}

    values = array("h")
    values.frombytes(pcm_bytes[: samples * 2])
    if sys.byteorder != "little":
        values.byteswap()

    peak = 0
    sum_sq = 0.0
    zero_count = 0
    for sample in values:
        value = int(sample)
        abs_value = -value if value < 0 else value
        peak = max(peak, abs_value)
        sum_sq += float(value) * float(value)
        if value == 0:
            zero_count += 1

    return {
        "byte_len": byte_len,
        "samples": samples,
        "duration_sec": samples / float(PCM_SAMPLE_RATE),
        "rms": math.sqrt(sum_sq / float(samples)),
        "peak": peak,
        "zero_frac": zero_count / float(samples),
    }


def pcm_to_wav_file(pcm_bytes: bytes) -> Path:
    if not pcm_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    if len(pcm_bytes) % PCM_SAMPLE_WIDTH_BYTES != 0:
        raise HTTPException(status_code=400, detail="PCM byte length is not aligned to int16")

    fd, raw_path = tempfile.mkstemp(suffix=".wav", prefix="xiaosita_asr_", text=False)
    os.close(fd)
    wav_path = Path(raw_path)
    try:
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(PCM_CHANNELS)
            handle.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
            handle.setframerate(PCM_SAMPLE_RATE)
            handle.writeframes(pcm_bytes)
        return wav_path
    except Exception as error:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to write wav: {error}") from error


def extract_text_from_response(resp: Any) -> str:
    data = to_dict(resp)
    paths = [
        ("text",),
        ("result",),
        ("output", "text"),
        ("output", "sentence", "text"),
        ("output", "transcription"),
        ("output", "transcript"),
        ("output", "result"),
    ]
    for path in paths:
        cur = data
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                cur = None
                break
        if isinstance(cur, str) and cur.strip():
            return cur.strip()

    for value in (
        safe_get(data, "output", "sentence"),
        safe_get(data, "output", "results"),
        safe_get(data, "results"),
        safe_get(data, "output", "sentences"),
        safe_get(data, "output", "choices"),
    ):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for key in ("text", "sentence", "transcription", "result"):
                        text = item.get(key)
                        if isinstance(text, str) and text.strip():
                            return text.strip()
    return ""


def unique_items(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        item = (value or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


async def recognize_pcm(pcm_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    require_dashscope()
    if Recognition is None:
        raise HTTPException(status_code=500, detail="dashscope.audio.asr.Recognition is unavailable")

    wav_path = pcm_to_wav_file(pcm_bytes)
    models = unique_items([DASHSCOPE_ASR_MODEL, "qwen3-asr-flash", "paraformer-realtime-v1", DASHSCOPE_ASR_FALLBACK_MODEL])
    last_error: Optional[Exception] = None
    last_meta: Dict[str, Any] = {}
    try:
        for model_name in models:
            try:
                def run_call() -> Any:
                    try:
                        recognizer = Recognition(model=model_name, callback=None, format="wav", sample_rate=PCM_SAMPLE_RATE)
                        return recognizer.call(file=str(wav_path))
                    except Exception:
                        return Recognition.call(model=model_name, file=str(wav_path), format="wav", sample_rate=PCM_SAMPLE_RATE)

                resp = await asyncio.to_thread(run_call)
                status_code = safe_get(resp, "status_code")
                meta = {
                    "model": model_name,
                    "status_code": status_code,
                    "code": safe_get(resp, "code"),
                    "message": safe_get(resp, "message"),
                    "request_id": safe_get(resp, "request_id"),
                }
                last_meta = meta
                if status_code not in (None, HTTPStatus.OK, 200):
                    last_error = RuntimeError(str(meta))
                    continue
                text = extract_text_from_response(resp)
                if text:
                    return text, meta
            except Exception as error:
                last_error = error
        if last_meta:
            return "", last_meta
        raise HTTPException(status_code=500, detail=f"All ASR models failed. Last error: {last_error}")
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass


def detect_language(text: str, explicit: Optional[str] = None) -> str:
    raw = (explicit or "").strip().lower()
    if raw in ("en", "eng", "english"):
        return "en"
    if raw in ("zh", "cn", "chinese", "mandarin"):
        return "zh"
    chinese = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    english = sum(1 for char in text if "a" <= char.lower() <= "z")
    return "en" if chinese == 0 and english >= 3 else "zh"


def build_chat_prompt(user_text: str, language: str) -> str:
    if language == "en":
        language_rule = "Answer in English only. Do not switch to Chinese unless the user asks for Chinese."
    else:
        language_rule = "请只用中文回答，除非用户明确要求英文。"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"已有开场白音频内容：{OPENING_ASSISTANT_TEXT}\n"
        "现在不要重复开场白，直接回答用户的问题。\n"
        f"{language_rule}\n"
        f"用户：{user_text}"
    )


async def chat_with_bailian(user_text: str, *, session_id: Optional[str], language: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
    require_dashscope()
    if Application is None:
        raise HTTPException(status_code=500, detail="dashscope.Application is unavailable")

    kwargs: Dict[str, Any] = {
        "api_key": DASHSCOPE_API_KEY,
        "app_id": BAILIAN_APP_ID,
        "prompt": build_chat_prompt(user_text, language),
    }
    if session_id:
        kwargs["session_id"] = session_id

    resp = await asyncio.to_thread(lambda: Application.call(**kwargs))
    status_code = safe_get(resp, "status_code")
    if status_code not in (None, HTTPStatus.OK, 200):
        raise HTTPException(
            status_code=502,
            detail={
                "service": "bailian",
                "status_code": status_code,
                "code": safe_get(resp, "code"),
                "message": safe_get(resp, "message"),
                "request_id": safe_get(resp, "request_id"),
            },
        )

    reply = safe_get(resp, "output", "text") or safe_get(resp, "text") or ""
    next_session_id = safe_get(resp, "output", "session_id") or session_id
    return str(reply).strip(), next_session_id, {"request_id": safe_get(resp, "request_id")}


def extract_audio_meta(resp: Any, candidate: Dict[str, str]) -> Dict[str, Any]:
    audio = safe_get(resp, "output", "audio") or {}
    url = safe_get(audio, "url")
    data = safe_get(audio, "data")
    return {
        "provider": candidate["provider"],
        "model": candidate["model"],
        "voice": candidate["voice"],
        "language_type": candidate.get("language_type"),
        "audio_url": str(url) if url else None,
        "audio_base64": str(data) if data else None,
        "request_id": safe_get(resp, "request_id"),
    }


async def synthesize_speech(text: str, language: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    require_dashscope()
    if SpeechSynthesizer is None:
        raise HTTPException(status_code=500, detail="dashscope.audio.qwen_tts.SpeechSynthesizer is unavailable")

    candidates = [
        {"provider": "qwen_tts", "model": DASHSCOPE_TTS_MODEL, "voice": DASHSCOPE_TTS_VOICE, "language_type": "Chinese"},
        {"provider": "qwen_tts", "model": DASHSCOPE_TTS_MODEL, "voice": "Ethan", "language_type": "Chinese"},
        {"provider": "qwen_tts", "model": DASHSCOPE_TTS_MODEL, "voice": "Eldric Sage", "language_type": "Chinese"},
        {"provider": "qwen_tts", "model": DASHSCOPE_TTS_MODEL, "voice": "Ryan", "language_type": "Chinese"},
        {"provider": "multimodal", "model": DASHSCOPE_TTS_MODEL, "voice": DASHSCOPE_TTS_VOICE, "language_type": "Auto" if language == "en" else "Chinese"},
        {"provider": "multimodal", "model": DASHSCOPE_TTS_MODEL, "voice": "Ethan", "language_type": "Auto" if language == "en" else "Chinese"},
    ]
    attempts: List[Dict[str, Any]] = []

    for candidate in candidates:
        try:
            def run_call() -> Any:
                if candidate["provider"] == "qwen_tts":
                    return SpeechSynthesizer.call(
                        model=candidate["model"],
                        text=text,
                        api_key=DASHSCOPE_API_KEY,
                        voice=candidate["voice"],
                    )
                return dashscope.MultiModalConversation.call(
                    api_key=DASHSCOPE_API_KEY,
                    model=candidate["model"],
                    text=text,
                    voice=candidate["voice"],
                    language_type=candidate["language_type"],
                )

            resp = await asyncio.to_thread(run_call)
            status_code = safe_get(resp, "status_code")
            attempt = {
                **candidate,
                "status_code": status_code,
                "code": safe_get(resp, "code"),
                "message": safe_get(resp, "message"),
                "request_id": safe_get(resp, "request_id"),
            }
            attempts.append(attempt)
            if status_code not in (None, HTTPStatus.OK, 200):
                continue
            meta = extract_audio_meta(resp, candidate)
            if meta.get("audio_url") or meta.get("audio_base64"):
                return meta, attempts
        except Exception as error:
            attempts.append({**candidate, "message": str(error)})

    raise HTTPException(status_code=502, detail={"service": "tts", "message": "All TTS candidates failed", "attempts": attempts})


def proxied_audio_url(audio_url: Optional[str]) -> Optional[str]:
    if not audio_url:
        return None
    parsed = urlparse(audio_url)
    if parsed.scheme == "http":
        return "/proxy-audio?url=" + quote(audio_url, safe="")
    return audio_url


def opening_files() -> List[Dict[str, str]]:
    files = []
    for path in sorted(OPENING_DIR.glob("*.wav")):
        files.append({"name": path.name, "url": f"/assets/opening/{path.name}"})
    return files


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        WEB_DIR / "index.html",
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "dashscope_configured": bool(DASHSCOPE_API_KEY),
        "bailian_app_id": BAILIAN_APP_ID,
        "opening_audio_count": len(opening_files()),
    }


@app.get("/api/opening")
async def get_opening() -> Dict[str, Any]:
    files = opening_files()
    return {"success": True, "text": OPENING_ASSISTANT_TEXT, "files": files, "random": random.choice(files) if files else None}


@app.post("/api/asr")
async def asr(req: TurnRequest) -> Dict[str, Any]:
    fmt = (req.audio_format or "").lower()
    if fmt not in ("pcm16le_16k_mono", "pcm16le", "pcm16"):
        raise HTTPException(status_code=400, detail=f"Unsupported audio_format: {req.audio_format}")

    pcm_bytes = decode_audio_base64(req.audio_data)
    audio_info = pcm_stats(pcm_bytes)
    if audio_info["duration_sec"] < 0.15 or (audio_info["peak"] <= 80 and audio_info["rms"] <= 40 and audio_info["zero_frac"] >= 0.98):
        return {"success": True, "has_text": False, "text": "", "audio_info": audio_info}

    text, asr_meta = await recognize_pcm(pcm_bytes)
    return {
        "success": True,
        "has_text": bool(text),
        "text": text,
        "audio_info": audio_info,
        "asr": asr_meta,
    }


@app.post("/api/turn")
async def turn(req: TurnRequest) -> Dict[str, Any]:
    fmt = (req.audio_format or "").lower()
    if fmt not in ("pcm16le_16k_mono", "pcm16le", "pcm16"):
        raise HTTPException(status_code=400, detail=f"Unsupported audio_format: {req.audio_format}")

    pcm_bytes = decode_audio_base64(req.audio_data)
    audio_info = pcm_stats(pcm_bytes)
    if audio_info["duration_sec"] < 0.15 or (audio_info["peak"] <= 80 and audio_info["rms"] <= 40 and audio_info["zero_frac"] >= 0.98):
        return {"success": True, "has_text": False, "text": "", "reply": "", "audio_info": audio_info}

    text, asr_meta = await recognize_pcm(pcm_bytes)
    if not text:
        return {"success": True, "has_text": False, "text": "", "reply": "", "audio_info": audio_info, "asr": asr_meta}

    language = detect_language(text, req.language)
    reply, session_id, chat_meta = await chat_with_bailian(text, session_id=req.session_id, language=language)
    if not reply:
        reply = "我听到了，但此刻还没有形成清晰的回答。请再说一遍。"

    tts_meta, attempts = await synthesize_speech(reply, language)
    audio_url = tts_meta.get("audio_url")
    return {
        "success": True,
        "has_text": True,
        "text": text,
        "reply": reply,
        "language": language,
        "session_id": session_id,
        "audio_url": audio_url,
        "proxy_audio_url": proxied_audio_url(audio_url),
        "audio_base64": tts_meta.get("audio_base64"),
        "debug": {"audio": audio_info, "asr": asr_meta, "chat": chat_meta, "tts": {"meta": tts_meta, "attempts": attempts}},
    }


@app.post("/api/chat")
async def chat(req: TextRequest) -> Dict[str, Any]:
    language = detect_language(req.text, req.language)
    reply, session_id, meta = await chat_with_bailian(req.text, session_id=req.session_id, language=language)
    return {"success": True, "text": reply, "language": language, "session_id": session_id, "debug": meta}


@app.post("/api/tts")
async def tts(req: TextRequest) -> Dict[str, Any]:
    language = detect_language(req.text, req.language)
    meta, attempts = await synthesize_speech(req.text, language)
    audio_url = meta.get("audio_url")
    return {
        "success": True,
        "audio_url": audio_url,
        "proxy_audio_url": proxied_audio_url(audio_url),
        "audio_base64": meta.get("audio_base64"),
        "debug": {"meta": meta, "attempts": attempts},
    }


@app.get("/proxy-audio")
async def proxy_audio(url: str = Query(...)) -> Response:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not (hostname == "aliyuncs.com" or hostname.endswith(".aliyuncs.com")):
        raise HTTPException(status_code=403, detail="Audio host is not allowed")

    def fetch() -> Tuple[bytes, str]:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read(), response.headers.get("Content-Type", "audio/wav")

    try:
        data, content_type = await asyncio.to_thread(fetch)
        return Response(content=data, media_type=content_type, headers={"Cache-Control": "no-store"})
    except urllib.error.HTTPError as error:
        raise HTTPException(status_code=error.code, detail=error.reason) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Audio proxy failed: {error}") from error
