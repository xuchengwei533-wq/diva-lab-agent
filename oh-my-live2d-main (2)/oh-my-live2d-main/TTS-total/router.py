from __future__ import annotations

import asyncio
import base64
import binascii
import mimetypes
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tts.synthesis import synthesize_with_candidates
from tts.text_utils import _normalize_text
from tts.voices import VOICE_TYPE_TO_VOICE, normalize_voice

router = APIRouter()

STORAGE_DIR = Path(__file__).resolve().parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PCM_SAMPLE_RATE = 24000
DEFAULT_PCM_CHANNELS = 1


class TTSTotalSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_type: Optional[str] = "deep_male"
    voice: Optional[str] = None
    filename_prefix: Optional[str] = "tts_total"


TTSTotalSpeakRequest.model_rebuild()


def _json_error(status_code: int, message: str, **extra: Any) -> JSONResponse:
    payload: Dict[str, Any] = {"success": False, "error": message}
    payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def _sanitize_filename_prefix(prefix: Optional[str]) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", (prefix or "tts_total").strip())
    value = value.strip("._-")
    if not value:
        value = "tts_total"
    return value[:32]


def _build_file_stem(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def _detect_audio_format(data: bytes, hint: Optional[str] = None, source_name: Optional[str] = None) -> str:
    hint_text = (hint or "").lower()
    name_text = (source_name or "").lower()
    combined = f"{hint_text} {name_text}"
    if "mp3" in combined or "mpeg" in combined:
        return "mp3"
    if "wav" in combined or "wave" in combined:
        return "wav"
    if "pcm" in combined or "s16le" in combined:
        return "pcm"
    if data.startswith(b"ID3"):
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "wav"
    return "bin"


def _raw_extension(audio_format: str) -> str:
    fmt = (audio_format or "bin").lower()
    if fmt in ("pcm_s16le", "pcm", "s16le"):
        return "pcm"
    if fmt in ("wave", "x-wav"):
        return "wav"
    if fmt in ("mpeg",):
        return "mp3"
    return fmt or "bin"


def _decode_audio_base64(audio_base64: str) -> bytes:
    try:
        return base64.b64decode(audio_base64)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"audio_base64 解码失败: {exc}") from exc


def _download_audio_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError(f"下载音频失败：{url}")
    return data


async def _resolve_audio_bytes(meta: Dict[str, Any]) -> Tuple[bytes, str]:
    if meta.get("audio_base64"):
        data = await asyncio.to_thread(_decode_audio_base64, meta["audio_base64"])
        fmt = _detect_audio_format(data, hint=meta.get("format"))
        return data, fmt
    if meta.get("audio_url"):
        data = await asyncio.to_thread(_download_audio_bytes, meta["audio_url"])
        fmt = _detect_audio_format(data, hint=meta.get("format"), source_name=meta.get("audio_url"))
        return data, fmt
    raise RuntimeError("TTS 合成成功，但返回中没有 audio_base64 或 audio_url")


def _ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def _convert_to_mp3(raw_path: Path, mp3_path: Path, audio_format: str) -> None:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError(
            "当前音频不是 mp3，且本机未安装 ffmpeg，无法转换为 MP3。"
            f"原始调试文件已保留在: {raw_path}"
        )

    fmt = _raw_extension(audio_format)
    cmd = [ffmpeg, "-y"]
    if fmt == "pcm":
        cmd.extend(
            [
                "-f",
                "s16le",
                "-ar",
                str(DEFAULT_PCM_SAMPLE_RATE),
                "-ac",
                str(DEFAULT_PCM_CHANNELS),
            ]
        )
    cmd.extend(["-i", str(raw_path), str(mp3_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "ffmpeg 转换 MP3 失败。"
            f" 原始调试文件已保留在: {raw_path}"
            f" 详情: {detail[:1000]}"
        )


def _preferred_voice(voice_type: Optional[str], voice: Optional[str]) -> Optional[str]:
    if voice:
        return normalize_voice(voice)
    mapped = VOICE_TYPE_TO_VOICE.get((voice_type or "deep_male").strip().lower())
    if mapped:
        return mapped
    return normalize_voice(voice_type)


def _safe_storage_file(filename: str) -> Path:
    if not filename or filename != Path(filename).name:
        raise HTTPException(status_code=404, detail="file not found")
    path = (STORAGE_DIR / filename).resolve()
    if path.parent != STORAGE_DIR.resolve():
        raise HTTPException(status_code=404, detail="file not found")
    return path


def _audio_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".pcm":
        return "application/octet-stream"
    media_type, _ = mimetypes.guess_type(path.name)
    return media_type or "application/octet-stream"


@router.post("/api/tts-total/speak")
async def tts_total_speak(req: TTSTotalSpeakRequest, request: Request):
    text = _normalize_text(req.text)
    if not text:
        return _json_error(400, "Empty text")

    prefix = _sanitize_filename_prefix(req.filename_prefix)
    file_stem = _build_file_stem(prefix)
    preferred_voice = _preferred_voice(req.voice_type, req.voice)

    try:
        meta, attempts = await synthesize_with_candidates(text, preferred_voice=preferred_voice)
        audio_bytes, detected_format = await _resolve_audio_bytes(meta)
    except Exception as exc:
        message = str(exc)
        if hasattr(exc, "to_dict"):
            payload = exc.to_dict()
            payload["success"] = False
            return JSONResponse(status_code=500, content=payload)
        return _json_error(500, message)

    mp3_path = STORAGE_DIR / f"{file_stem}.mp3"
    raw_extension = _raw_extension(detected_format)
    raw_path = STORAGE_DIR / f"{file_stem}.{raw_extension}"

    try:
        if detected_format == "mp3":
            mp3_path.write_bytes(audio_bytes)
        else:
            raw_path.write_bytes(audio_bytes)
            await asyncio.to_thread(_convert_to_mp3, raw_path, mp3_path, detected_format)
    except Exception as exc:
        return _json_error(
            500,
            str(exc),
            format="mp3",
            file_name=mp3_path.name,
            raw_debug_file=raw_path.name if raw_path.exists() else None,
        )

    audio_url = str(request.base_url).rstrip("/") + f"/api/tts-total/files/{mp3_path.name}"
    return {
        "success": True,
        "mode": "total",
        "format": "mp3",
        "file_name": mp3_path.name,
        "file_path": f"TTS-total/storage/{mp3_path.name}",
        "audio_url": audio_url,
        "provider": meta.get("provider"),
        "model": meta.get("model"),
        "voice": meta.get("voice"),
        "language_type": meta.get("language_type"),
        "attempts": attempts,
    }


@router.get("/api/tts-total/files/{filename}")
async def tts_total_file(filename: str):
    path = _safe_storage_file(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type=_audio_media_type(path), filename=path.name)


@router.get("/api/tts-total/random-wav")
async def tts_total_random_wav(request: Request):
    wav_files = [p for p in STORAGE_DIR.glob("*.wav") if p.is_file()]
    if not wav_files:
        return _json_error(404, "No wav files found in TTS-total/storage")

    chosen = random.choice(wav_files)
    audio_url = str(request.base_url).rstrip("/") + f"/api/tts-total/files/{chosen.name}"
    return {
        "success": True,
        "mode": "local-random-wav",
        "format": "wav",
        "file_name": chosen.name,
        "file_path": f"TTS-total/storage/{chosen.name}",
        "audio_url": audio_url,
    }
