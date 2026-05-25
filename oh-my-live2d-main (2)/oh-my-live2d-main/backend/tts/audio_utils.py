import base64
import urllib.request
from typing import Any, Dict, Optional

from .config import PCM_FORMAT, PCM_SAMPLE_RATE


def _guess_audio_format(url: Optional[str], audio_data: Optional[str]) -> str:
    if url:
        lower = url.lower()
        if '.mp3' in lower:
            return 'mp3'
        if '.wav' in lower:
            return 'wav'
        if '.pcm' in lower:
            return 'pcm_s16le'
    if audio_data:
        return 'wav'
    return 'wav'


def _guess_stream_audio_format(resp: Any, candidate: Dict[str, Any]) -> str:
    fmt = None
    try:
        output = getattr(resp, 'output', None)
        if isinstance(output, dict):
            audio = output.get('audio')
            if isinstance(audio, dict):
                fmt = audio.get('format')
    except Exception:
        fmt = None
    if fmt:
        return str(fmt)
    provider = str(candidate.get('provider') or '').strip()
    if provider in ('multimodal', 'qwen_tts'):
        return PCM_FORMAT
    return 'wav'


def _guess_stream_sample_rate(resp: Any) -> int:
    output = getattr(resp, 'output', None)
    audio = output.get('audio') if isinstance(output, dict) else None
    if isinstance(audio, dict):
        for key in ('sample_rate', 'sampleRate', 'sampling_rate', 'samplingRate'):
            value = audio.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return PCM_SAMPLE_RATE


def _download_to_base64(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as resp:
        content = resp.read()
    if not content:
        raise RuntimeError('Empty audio content downloaded')
    return base64.b64encode(content).decode('utf-8')
