import asyncio
from http import HTTPStatus
from typing import Any, Dict, Optional, Tuple

import dashscope
from dashscope.audio.qwen_tts import SpeechSynthesizer

from .audio_utils import _guess_audio_format
from .candidates import _build_attempt
from .config import DASHSCOPE_API_KEY, PCM_FORMAT, PCM_SAMPLE_RATE


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


def _extract_audio_meta(resp: Any, candidate: Dict[str, Any]) -> Dict[str, Any]:
    out = _safe_get(resp, 'output', default=None)
    audio = _safe_get(out, 'audio', default=None)
    url = _safe_get(audio, 'url', default=None)
    data = _safe_get(audio, 'data', default=None)
    expires_at = _safe_get(audio, 'expires_at', default=None)
    return {
        'provider': candidate.get('provider'),
        'model': candidate.get('model'),
        'voice': candidate.get('voice'),
        'language_type': candidate.get('language_type'),
        'audio_url': str(url) if url else None,
        'audio_base64': str(data) if data else None,
        'expires_at': expires_at,
        'format': _guess_audio_format(str(url) if url else None, str(data) if data else None),
    }


async def _call_qwen_tts(text: str, candidate: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    def _run():
        return SpeechSynthesizer.call(
            model=str(candidate['model']),
            text=text,
            api_key=DASHSCOPE_API_KEY,
            voice=str(candidate['voice']),
        )

    try:
        resp = await asyncio.to_thread(_run)
    except Exception as e:
        attempt = _build_attempt(
            candidate,
            status_code=getattr(e, 'status_code', None),
            dashscope_code=str(getattr(e, 'code', '') or '') or None,
            dashscope_message=str(getattr(e, 'message', '') or str(e)),
            request_id=str(getattr(e, 'request_id', '') or '') or None,
        )
        return None, attempt

    status_code = _safe_get(resp, 'status_code')
    code = _safe_get(resp, 'code')
    message = _safe_get(resp, 'message')
    request_id = _safe_get(resp, 'request_id')
    meta = _extract_audio_meta(resp, candidate)

    if status_code not in (None, HTTPStatus.OK, 200):
        return None, _build_attempt(
            candidate,
            status_code=int(status_code) if status_code is not None else None,
            dashscope_code=str(code) if code else None,
            dashscope_message=str(message or 'DashScope request failed'),
            request_id=str(request_id) if request_id else None,
        )

    if meta['audio_url'] or meta['audio_base64']:
        return meta, _build_attempt(
            candidate,
            status_code=200,
            dashscope_code=str(code) if code else None,
            dashscope_message=str(message or 'ok'),
            request_id=str(request_id) if request_id else None,
        )

    return None, _build_attempt(
        candidate,
        status_code=200,
        dashscope_code=str(code) if code else None,
        dashscope_message='DashScope returned success but no audio data/url',
        request_id=str(request_id) if request_id else None,
    )


async def _call_multimodal_tts(text: str, candidate: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    def _run():
        return dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=str(candidate['model']),
            text=text,
            voice=str(candidate['voice']),
            language_type=str(candidate['language_type']),
        )

    try:
        resp = await asyncio.to_thread(_run)
    except Exception as e:
        attempt = _build_attempt(
            candidate,
            status_code=getattr(e, 'status_code', None),
            dashscope_code=str(getattr(e, 'code', '') or '') or None,
            dashscope_message=str(getattr(e, 'message', '') or str(e)),
            request_id=str(getattr(e, 'request_id', '') or '') or None,
        )
        return None, attempt

    status_code = _safe_get(resp, 'status_code')
    code = _safe_get(resp, 'code')
    message = _safe_get(resp, 'message')
    request_id = _safe_get(resp, 'request_id')
    meta = _extract_audio_meta(resp, candidate)

    if status_code not in (None, HTTPStatus.OK, 200):
        return None, _build_attempt(
            candidate,
            status_code=int(status_code) if status_code is not None else None,
            dashscope_code=str(code) if code else None,
            dashscope_message=str(message or 'DashScope request failed'),
            request_id=str(request_id) if request_id else None,
        )

    if meta['audio_url'] or meta['audio_base64']:
        return meta, _build_attempt(
            candidate,
            status_code=200,
            dashscope_code=str(code) if code else None,
            dashscope_message=str(message or 'ok'),
            request_id=str(request_id) if request_id else None,
        )

    return None, _build_attempt(
        candidate,
        status_code=200,
        dashscope_code=str(code) if code else None,
        dashscope_message='DashScope returned success but no audio data/url',
        request_id=str(request_id) if request_id else None,
    )


async def _call_tts_candidate(text: str, candidate: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    provider = str(candidate.get('provider') or '').strip()
    if provider == 'qwen_tts':
        return await _call_qwen_tts(text, candidate)
    if provider == 'multimodal':
        return await _call_multimodal_tts(text, candidate)
    return None, _build_attempt(
        candidate,
        dashscope_message=f'Unsupported provider: {provider}',
    )


async def _iter_dashscope_stream(callable_obj):
    stream_or_resp = await asyncio.to_thread(callable_obj)
    if hasattr(stream_or_resp, '__iter__') and not isinstance(stream_or_resp, (dict, str, bytes)):
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
    provider = str(candidate.get('provider') or '').strip()
    model = str(candidate['model'])
    voice = str(candidate['voice'])
    if provider == 'multimodal':
        return dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=model,
            text=text,
            voice=voice,
            language_type=str(candidate['language_type']),
            stream=True,
            format=PCM_FORMAT,
            sample_rate=PCM_SAMPLE_RATE,
        )
    if provider == 'qwen_tts':
        return SpeechSynthesizer.call(
            model=model,
            text=text,
            api_key=DASHSCOPE_API_KEY,
            voice=voice,
            stream=True,
            format=PCM_FORMAT,
            sample_rate=PCM_SAMPLE_RATE,
        )
    raise RuntimeError(f'Unsupported provider: {provider}')
