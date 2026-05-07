import os
import json
import asyncio
import threading
import base64
import urllib.request
from http import HTTPStatus
from typing import Any, Dict, Optional, List, Tuple

import dashscope
from dashscope.audio.qwen_tts import SpeechSynthesizer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def _load_env_file():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, ".env"),
        os.path.join(os.path.dirname(here), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            pass
        break


_load_env_file()

dashscope.base_http_api_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

DEFAULT_MODEL = os.getenv("DASHSCOPE_TTS_MODEL", "qwen3-tts-flash")
DEFAULT_VOICE = os.getenv("DASHSCOPE_TTS_VOICE", "Ethan")
DEFAULT_LANG = os.getenv("DASHSCOPE_TTS_LANG", "Chinese")
MIN_CHARS_PER_REQ = int(os.getenv("TTS_MIN_CHARS_PER_REQ", "24"))
MAX_CHARS_PER_REQ = int(os.getenv("TTS_MAX_CHARS_PER_REQ", "300"))
PCM_SAMPLE_RATE = int(os.getenv("TTS_PCM_SAMPLE_RATE", "24000"))
PCM_FORMAT = os.getenv("TTS_PCM_FORMAT", "pcm_s16le")
CHINESE_PROBE_TEXT = os.getenv("TTS_PROBE_TEXT_ZH", "你好，今天我们来练习唱歌。")
DEFAULT_PROBE_TEXT = os.getenv("TTS_PROBE_TEXT_DEFAULT", "Hello, today we will practice singing.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICE_TYPE_TO_VOICE = {
    "default": "Ethan",
    "male": "Ethan",
    "deep_male": "Ethan",
    "mature_male": "Andre",
    "elder_male": "Eldric Sage",
    "news_male": "Neil",
    "calm": "Ethan",
    # Backward compatibility: all legacy aliases still resolve to male voices.
    "cute": "Ethan",
    "fast": "Ethan",
    "slow": "Ethan",
    "mao": "Ethan",
    "female": "Ethan",
    "female_cute": "Ethan",
    "female_lively": "Neil",
    "female_gentle": "Ethan",
}

SUPPORTED_VOICES = [
    "Ethan",
    "Eldric Sage",
    "Ryan",
    "Aiden",
    "Neil",
    "Andre",
    "Vincent",
    "Moon",
]

_probe_cache: Dict[str, Dict[str, Any]] = {}
_probe_attempts: Dict[str, List[Dict[str, Any]]] = {}
_probe_lock = threading.Lock()


class TTSSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_type: Optional[str] = Field(default="deep_male")


def _safe_get(obj: Any, *keys: str, default=None):
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _is_chinese_text(text: str) -> bool:
    for ch in text or "":
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def _text_bucket(text: str) -> str:
    return "zh" if _is_chinese_text(text) else "default"


def _normalize_text(text: str) -> str:
    return (text or "").strip()


def normalize_voice(voice: Optional[str]) -> str:
    if not voice:
        return DEFAULT_VOICE
    value = voice.strip()
    if not value:
        return DEFAULT_VOICE
    if value in ("Cherry", "Serena", "Chelsie", "cute", "mao"):
        return DEFAULT_VOICE
    return value


def _split_text_for_tts(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    if len(text) <= MAX_CHARS_PER_REQ:
        return [text]

    seps = set("。！？!?；;\n")
    chunks: List[str] = []
    cur: List[str] = []
    for ch in text:
        cur.append(ch)
        if ch in seps and len(cur) >= MIN_CHARS_PER_REQ:
            seg = "".join(cur).strip()
            if seg:
                chunks.append(seg)
            cur = []
        if len(cur) >= MAX_CHARS_PER_REQ:
            seg = "".join(cur).strip()
            if seg:
                chunks.append(seg)
            cur = []
    tail = "".join(cur).strip()
    if tail:
        chunks.append(tail)
    return [c for c in chunks if c]


def _guess_audio_format(url: Optional[str], audio_data: Optional[str]) -> str:
    if url:
        lower = url.lower()
        if ".mp3" in lower:
            return "mp3"
        if ".wav" in lower:
            return "wav"
        if ".pcm" in lower:
            return "pcm_s16le"
    if audio_data:
        return "wav"
    return "wav"


def _download_to_base64(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as resp:
        content = resp.read()
    if not content:
        raise RuntimeError("Empty audio content downloaded")
    return base64.b64encode(content).decode("utf-8")


def _dedupe_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("provider") or "").strip(),
            str(item.get("model") or "").strip(),
            str(item.get("voice") or "").strip(),
            str(item.get("language_type") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "provider": key[0],
                "model": key[1],
                "voice": key[2],
                "language_type": key[3] or DEFAULT_LANG,
            }
        )
    return out


def _default_tts_candidates() -> List[Dict[str, Any]]:
    voices = []
    for v in [DEFAULT_VOICE, *SUPPORTED_VOICES]:
        if v and v not in voices:
            voices.append(v)

    candidates: List[Dict[str, Any]] = []
    for voice in voices:
        for lang in ("Chinese", "Auto"):
            candidates.append(
                {
                    "provider": "qwen_tts",
                    "model": DEFAULT_MODEL,
                    "voice": voice,
                    "language_type": lang,
                }
            )
    for voice in voices:
        for lang in ("Chinese", "Auto"):
            candidates.append(
                {
                    "provider": "multimodal",
                    "model": DEFAULT_MODEL,
                    "voice": voice,
                    "language_type": lang,
                }
            )
    return _dedupe_candidates(candidates)


def _load_tts_candidates() -> List[Dict[str, Any]]:
    raw = os.getenv("DASHSCOPE_TTS_CANDIDATES")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                cleaned = _dedupe_candidates([x for x in data if isinstance(x, dict)])
                if cleaned:
                    return cleaned
        except Exception:
            pass
    return _default_tts_candidates()


TTS_CANDIDATES = _load_tts_candidates()


def _candidate_matches_voice(candidate: Dict[str, Any], preferred_voice: Optional[str]) -> bool:
    if not preferred_voice:
        return False
    return str(candidate.get("voice") or "").strip().lower() == preferred_voice.strip().lower()


def _build_attempt(
    candidate: Dict[str, Any],
    *,
    status_code: Optional[int] = None,
    dashscope_code: Optional[str] = None,
    dashscope_message: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "provider": candidate.get("provider"),
        "model": candidate.get("model"),
        "voice": candidate.get("voice"),
        "language_type": candidate.get("language_type"),
        "status_code": status_code,
        "dashscope_code": dashscope_code,
        "dashscope_message": dashscope_message,
        "request_id": request_id,
    }


class TTSAllFailedError(RuntimeError):
    def __init__(self, message: str, attempts: List[Dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts

    def to_dict(self) -> Dict[str, Any]:
        message = self.attempts[-1].get("dashscope_message") if self.attempts else str(self)
        return {
            "type": "error",
            "error": str(self),
            "message": message or str(self),
            "attempts": self.attempts,
        }


async def _ws_send_json(ws: WebSocket, payload: Dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


def _extract_audio_meta(resp: Any, candidate: Dict[str, Any]) -> Dict[str, Any]:
    out = _safe_get(resp, "output", default=None)
    audio = _safe_get(out, "audio", default=None)
    url = _safe_get(audio, "url", default=None)
    data = _safe_get(audio, "data", default=None)
    expires_at = _safe_get(audio, "expires_at", default=None)
    return {
        "provider": candidate.get("provider"),
        "model": candidate.get("model"),
        "voice": candidate.get("voice"),
        "language_type": candidate.get("language_type"),
        "audio_url": str(url) if url else None,
        "audio_base64": str(data) if data else None,
        "expires_at": expires_at,
        "format": _guess_audio_format(str(url) if url else None, str(data) if data else None),
    }


async def _call_qwen_tts(text: str, candidate: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    def _run():
        return SpeechSynthesizer.call(
            model=str(candidate["model"]),
            text=text,
            api_key=DASHSCOPE_API_KEY,
            voice=str(candidate["voice"]),
        )

    try:
        resp = await asyncio.to_thread(_run)
    except Exception as e:
        attempt = _build_attempt(
            candidate,
            status_code=getattr(e, "status_code", None),
            dashscope_code=str(getattr(e, "code", "") or "") or None,
            dashscope_message=str(getattr(e, "message", "") or str(e)),
            request_id=str(getattr(e, "request_id", "") or "") or None,
        )
        return None, attempt

    status_code = _safe_get(resp, "status_code")
    code = _safe_get(resp, "code")
    message = _safe_get(resp, "message")
    request_id = _safe_get(resp, "request_id")
    meta = _extract_audio_meta(resp, candidate)

    if status_code not in (None, HTTPStatus.OK, 200):
        return None, _build_attempt(
            candidate,
            status_code=int(status_code) if status_code is not None else None,
            dashscope_code=str(code) if code else None,
            dashscope_message=str(message or "DashScope request failed"),
            request_id=str(request_id) if request_id else None,
        )

    if meta["audio_url"] or meta["audio_base64"]:
        return meta, _build_attempt(
            candidate,
            status_code=200,
            dashscope_code=str(code) if code else None,
            dashscope_message=str(message or "ok"),
            request_id=str(request_id) if request_id else None,
        )

    return None, _build_attempt(
        candidate,
        status_code=200,
        dashscope_code=str(code) if code else None,
        dashscope_message="DashScope returned success but no audio data/url",
        request_id=str(request_id) if request_id else None,
    )


async def _call_multimodal_tts(text: str, candidate: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    def _run():
        return dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=str(candidate["model"]),
            text=text,
            voice=str(candidate["voice"]),
            language_type=str(candidate["language_type"]),
        )

    try:
        resp = await asyncio.to_thread(_run)
    except Exception as e:
        attempt = _build_attempt(
            candidate,
            status_code=getattr(e, "status_code", None),
            dashscope_code=str(getattr(e, "code", "") or "") or None,
            dashscope_message=str(getattr(e, "message", "") or str(e)),
            request_id=str(getattr(e, "request_id", "") or "") or None,
        )
        return None, attempt

    status_code = _safe_get(resp, "status_code")
    code = _safe_get(resp, "code")
    message = _safe_get(resp, "message")
    request_id = _safe_get(resp, "request_id")
    meta = _extract_audio_meta(resp, candidate)

    if status_code not in (None, HTTPStatus.OK, 200):
        return None, _build_attempt(
            candidate,
            status_code=int(status_code) if status_code is not None else None,
            dashscope_code=str(code) if code else None,
            dashscope_message=str(message or "DashScope request failed"),
            request_id=str(request_id) if request_id else None,
        )

    if meta["audio_url"] or meta["audio_base64"]:
        return meta, _build_attempt(
            candidate,
            status_code=200,
            dashscope_code=str(code) if code else None,
            dashscope_message=str(message or "ok"),
            request_id=str(request_id) if request_id else None,
        )

    return None, _build_attempt(
        candidate,
        status_code=200,
        dashscope_code=str(code) if code else None,
        dashscope_message="DashScope returned success but no audio data/url",
        request_id=str(request_id) if request_id else None,
    )


async def _call_tts_candidate(text: str, candidate: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    provider = str(candidate.get("provider") or "").strip()
    if provider == "qwen_tts":
        return await _call_qwen_tts(text, candidate)
    if provider == "multimodal":
        return await _call_multimodal_tts(text, candidate)
    return None, _build_attempt(
        candidate,
        dashscope_message=f"Unsupported provider: {provider}",
    )


def _cache_candidate_success(bucket: str, candidate: Dict[str, Any], attempts: List[Dict[str, Any]]) -> None:
    with _probe_lock:
        _probe_cache[bucket] = {
            "provider": candidate["provider"],
            "model": candidate["model"],
            "voice": candidate["voice"],
            "language_type": candidate["language_type"],
        }
        _probe_attempts[bucket] = attempts[:]


def _prioritize_candidates(preferred_voice: Optional[str], bucket: str) -> List[Dict[str, Any]]:
    cached: Optional[Dict[str, Any]]
    with _probe_lock:
        cached = dict(_probe_cache[bucket]) if bucket in _probe_cache else None

    ordered: List[Dict[str, Any]] = []
    if cached:
        ordered.append(cached)

    matches = []
    others = []
    for candidate in TTS_CANDIDATES:
        if cached and all(candidate.get(k) == cached.get(k) for k in ("provider", "model", "voice", "language_type")):
            continue
        if _candidate_matches_voice(candidate, preferred_voice):
            matches.append(candidate)
        else:
            others.append(candidate)
    return _dedupe_candidates(ordered + matches + others)


async def _probe_bucket(bucket: str, preferred_voice: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    probe_text = CHINESE_PROBE_TEXT if bucket == "zh" else DEFAULT_PROBE_TEXT
    attempts: List[Dict[str, Any]] = []
    for candidate in _prioritize_candidates(preferred_voice, bucket):
        meta, attempt = await _call_tts_candidate(probe_text, candidate)
        attempts.append(attempt)
        if meta:
            _cache_candidate_success(bucket, candidate, attempts)
            return meta, attempts
    with _probe_lock:
        _probe_attempts[bucket] = attempts[:]
    return None, attempts


async def synthesize_with_candidates(
    text: str,
    *,
    preferred_voice: Optional[str] = None,
    run_probe: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bucket = _text_bucket(text)
    attempts: List[Dict[str, Any]] = []

    if run_probe:
        with _probe_lock:
            has_cache = bucket in _probe_cache
        if not has_cache:
            await _probe_bucket(bucket, preferred_voice=preferred_voice)

    for candidate in _prioritize_candidates(preferred_voice, bucket):
        meta, attempt = await _call_tts_candidate(text, candidate)
        attempts.append(attempt)
        if meta:
            _cache_candidate_success(bucket, candidate, attempts)
            return meta, attempts

    raise TTSAllFailedError("All TTS providers failed", attempts)


def _guess_stream_audio_format(resp: Any, candidate: Dict[str, Any]) -> str:
    fmt = _safe_get(resp, "output", "audio", "format")
    if fmt:
        return str(fmt)
    provider = str(candidate.get("provider") or "").strip()
    if provider in ("multimodal", "qwen_tts"):
        return PCM_FORMAT
    return "wav"


def _guess_stream_sample_rate(resp: Any) -> int:
    for key in ("sample_rate", "sampleRate", "sampling_rate", "samplingRate"):
        value = _safe_get(resp, "output", "audio", key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return PCM_SAMPLE_RATE


async def _iter_dashscope_stream(callable_obj):
    stream_or_resp = await asyncio.to_thread(callable_obj)
    if hasattr(stream_or_resp, "__iter__") and not isinstance(stream_or_resp, (dict, str, bytes)):
        iterator = iter(stream_or_resp)
        sentinel = object()

        def _next_item():
            return next(iterator, sentinel)

        while True:
            item = await asyncio.to_thread(_next_item)
            if item is sentinel:
                break
            yield item
    else:
        yield stream_or_resp


def _run_streaming_call(candidate: Dict[str, Any], text: str):
    provider = str(candidate.get("provider") or "").strip()
    model = str(candidate["model"])
    voice = str(candidate["voice"])
    if provider == "multimodal":
        return dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=model,
            text=text,
            voice=voice,
            language_type=str(candidate["language_type"]),
            stream=True,
            format=PCM_FORMAT,
            sample_rate=PCM_SAMPLE_RATE,
        )
    if provider == "qwen_tts":
        return SpeechSynthesizer.call(
            model=model,
            text=text,
            api_key=DASHSCOPE_API_KEY,
            voice=voice,
            stream=True,
            format=PCM_FORMAT,
            sample_rate=PCM_SAMPLE_RATE,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


async def stream_tts_with_candidates(
    text: str,
    preferred_voice: Optional[str],
    ws: WebSocket,
    *,
    run_probe: bool = True,
) -> Dict[str, Any]:
    bucket = _text_bucket(text)
    attempts: List[Dict[str, Any]] = []

    if run_probe:
        with _probe_lock:
            has_cache = bucket in _probe_cache
        if not has_cache:
            await _probe_bucket(bucket, preferred_voice=preferred_voice)

    for candidate in _prioritize_candidates(preferred_voice, bucket):
        got_audio = False
        attempt_recorded = False
        last_attempt = _build_attempt(candidate)
        try:
            async for resp in _iter_dashscope_stream(lambda: _run_streaming_call(candidate, text)):
                status_code = _safe_get(resp, "status_code")
                code = _safe_get(resp, "code")
                message = _safe_get(resp, "message")
                request_id = _safe_get(resp, "request_id")
                last_attempt = _build_attempt(
                    candidate,
                    status_code=int(status_code) if status_code is not None else None,
                    dashscope_code=str(code) if code else None,
                    dashscope_message=str(message or "ok"),
                    request_id=str(request_id) if request_id else None,
                )

                if status_code not in (None, HTTPStatus.OK, 200):
                    attempts.append(last_attempt)
                    attempt_recorded = True
                    if got_audio:
                        await _ws_send_json(
                            ws,
                            {
                                **_ws_error_payload(TTSAllFailedError("Streaming TTS interrupted after audio started", attempts)),
                                "provider": candidate.get("provider"),
                                "model": candidate.get("model"),
                                "voice": candidate.get("voice"),
                                "language_type": candidate.get("language_type"),
                            },
                        )
                        return {"mode": "stream", "attempts": attempts, "interrupted": True}
                    break

                audio_data = _safe_get(resp, "output", "audio", "data")
                if audio_data:
                    got_audio = True
                    await _ws_send_json(
                        ws,
                        {
                            "type": "response.audio.delta",
                            "delta": str(audio_data),
                            "format": _guess_stream_audio_format(resp, candidate),
                            "sample_rate": _guess_stream_sample_rate(resp),
                            "provider": candidate["provider"],
                            "model": candidate["model"],
                            "voice": candidate["voice"],
                            "language_type": candidate["language_type"],
                        },
                    )

            if got_audio:
                if last_attempt.get("status_code") is None:
                    last_attempt["status_code"] = 200
                if not attempt_recorded:
                    attempts.append(last_attempt)
                _cache_candidate_success(bucket, candidate, attempts)
                return {"mode": "stream", "attempts": attempts, "interrupted": False}
            if not attempt_recorded and last_attempt.get("dashscope_message") in (None, "ok"):
                last_attempt["dashscope_message"] = "DashScope streaming finished without audio chunks"
            if not attempt_recorded:
                attempts.append(last_attempt)
        except Exception as e:
            attempt = _build_attempt(
                candidate,
                status_code=getattr(e, "status_code", None),
                dashscope_code=str(getattr(e, "code", "") or "") or None,
                dashscope_message=str(getattr(e, "message", "") or str(e)),
                request_id=str(getattr(e, "request_id", "") or "") or None,
            )
            attempts.append(attempt)
            if got_audio:
                await _ws_send_json(
                    ws,
                    {
                        **_ws_error_payload(TTSAllFailedError("Streaming TTS interrupted after audio started", attempts)),
                        "provider": candidate.get("provider"),
                        "model": candidate.get("model"),
                        "voice": candidate.get("voice"),
                        "language_type": candidate.get("language_type"),
                    },
                )
                return {"mode": "stream", "attempts": attempts, "interrupted": True}

    raise TTSAllFailedError("All streaming TTS providers failed", attempts)


async def _send_ws_fallback_audio(meta: Dict[str, Any], ws: WebSocket) -> None:
    if meta.get("audio_base64"):
        await _ws_send_json(
            ws,
            {
                "type": "response.audio.base64",
                "audio_base64": meta["audio_base64"],
                "format": meta["format"],
                "provider": meta["provider"],
                "model": meta["model"],
                "voice": meta["voice"],
                "language_type": meta["language_type"],
            },
        )
        return
    if meta.get("audio_url"):
        await _ws_send_json(
            ws,
            {
                "type": "response.audio.url",
                "url": meta["audio_url"],
                "format": meta["format"],
                "provider": meta["provider"],
                "model": meta["model"],
                "voice": meta["voice"],
                "language_type": meta["language_type"],
            },
        )
        return
    raise TTSAllFailedError(
        "All TTS providers failed",
        [
            _build_attempt(
                {
                    "provider": meta.get("provider"),
                    "model": meta.get("model"),
                    "voice": meta.get("voice"),
                    "language_type": meta.get("language_type"),
                },
                dashscope_message="TTS succeeded but returned no playable audio",
            )
        ],
    )


@app.post("/api/tts/speak")
async def speak(req: TTSSpeakRequest) -> Dict[str, Any]:
    if not DASHSCOPE_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": "DASHSCOPE_API_KEY not set in environment"},
        )

    text = _normalize_text(req.text)
    if not text:
        return JSONResponse(status_code=400, content={"type": "error", "error": "Empty text"})

    voice_type = (req.voice_type or "deep_male").strip().lower()
    preferred_voice = VOICE_TYPE_TO_VOICE.get(voice_type) or normalize_voice(req.voice_type or DEFAULT_VOICE)

    try:
        meta, attempts = await synthesize_with_candidates(text, preferred_voice=preferred_voice)
    except TTSAllFailedError as e:
        return JSONResponse(status_code=500, content=e.to_dict())

    result: Dict[str, Any] = {
        "success": True,
        "provider": meta["provider"],
        "model": meta["model"],
        "voice": meta["voice"],
        "language_type": meta["language_type"],
        "format": meta["format"],
        "attempts": attempts,
    }
    if meta.get("audio_url"):
        result["audio_url"] = meta["audio_url"]
    if meta.get("audio_base64"):
        result["audio_base64"] = meta["audio_base64"]
    return result


class SessionState:
    def __init__(self):
        self.text_buf = ""
        self.closed = False


def _ws_error_payload(error: TTSAllFailedError) -> Dict[str, Any]:
    return error.to_dict()


@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    await ws.accept()

    if not DASHSCOPE_API_KEY:
        await ws.send_text(json.dumps({"type": "error", "error": "DASHSCOPE_API_KEY not set in environment"}, ensure_ascii=False))
        await ws.close()
        return

    qp = ws.query_params
    model = qp.get("model") or DEFAULT_MODEL
    voice = normalize_voice(qp.get("voice") or DEFAULT_VOICE)
    language_type = qp.get("language_type") or DEFAULT_LANG
    state = SessionState()

    await ws.send_text(
        json.dumps(
            {
                "type": "session.ready",
                "sample_rate": PCM_SAMPLE_RATE,
                "format": PCM_FORMAT,
                "voice": voice,
                "model": model,
                "language_type": language_type,
            },
            ensure_ascii=False,
        )
    )

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "input_text_buffer.append":
                state.text_buf += str(msg.get("text") or "")
                continue

            if mtype == "input_text_buffer.commit":
                text = _normalize_text(state.text_buf)
                state.text_buf = ""
                if not text:
                    await _ws_send_json(ws, {"type": "error", "error": "Empty text buffer on commit"})
                    continue

                parts = _split_text_for_tts(text)
                fatal_error = False
                for seg in parts:
                    try:
                        await stream_tts_with_candidates(
                            seg,
                            preferred_voice=voice,
                            ws=ws,
                            run_probe=False,
                        )
                    except TTSAllFailedError as stream_error:
                        try:
                            meta, fallback_attempts = await synthesize_with_candidates(
                                seg,
                                preferred_voice=voice,
                                run_probe=False,
                            )
                            await _send_ws_fallback_audio(meta, ws)
                        except TTSAllFailedError as fallback_error:
                            combined_error = TTSAllFailedError(
                                "All streaming and fallback TTS providers failed",
                                stream_error.attempts + fallback_error.attempts,
                            )
                            await _ws_send_json(ws, _ws_error_payload(combined_error))
                            fatal_error = True
                            break
                    await _ws_send_json(ws, {"type": "response.segment.done"})

                if not fatal_error:
                    await _ws_send_json(ws, {"type": "response.done"})
                continue

            if mtype == "session.finish":
                state.closed = True
                break

            if mtype == "audio.playback.ended":
                continue

            await ws.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": f"Unknown message type: {mtype}",
                    },
                    ensure_ascii=False,
                )
            )

    except WebSocketDisconnect:
        state.closed = True
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
