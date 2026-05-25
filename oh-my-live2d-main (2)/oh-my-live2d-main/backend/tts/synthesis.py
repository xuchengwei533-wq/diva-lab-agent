import json
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

from fastapi import WebSocket

from .audio_utils import _guess_stream_audio_format, _guess_stream_sample_rate
from .candidates import _build_attempt, _cache_candidate_success, _prioritize_candidates, _probe_bucket, _probe_cache, _probe_lock
from .dashscope_client import _call_tts_candidate, _iter_dashscope_stream, _run_streaming_call, _safe_get
from .errors import TTSAllFailedError
from .text_utils import _text_bucket


async def _ws_send_json(ws: WebSocket, payload: Dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


def _ws_error_payload(error: TTSAllFailedError) -> Dict[str, Any]:
    return error.to_dict()


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

    raise TTSAllFailedError('All TTS providers failed', attempts)


async def stream_tts_with_candidates(
    text: str,
    preferred_voice: Optional[str],
    ws: WebSocket,
    *,
    run_probe: bool = True,
) -> Dict[str, Any]:
    bucket = _text_bucket(text)
    attempts: List[Dict[str, Any]] = []
    print(f'[TTS_SEGMENT] len={len(text)} text={text}')

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
                status_code = _safe_get(resp, 'status_code')
                code = _safe_get(resp, 'code')
                message = _safe_get(resp, 'message')
                request_id = _safe_get(resp, 'request_id')
                last_attempt = _build_attempt(
                    candidate,
                    status_code=int(status_code) if status_code is not None else None,
                    dashscope_code=str(code) if code else None,
                    dashscope_message=str(message or 'ok'),
                    request_id=str(request_id) if request_id else None,
                )

                if status_code not in (None, HTTPStatus.OK, 200):
                    attempts.append(last_attempt)
                    attempt_recorded = True
                    if got_audio:
                        await _ws_send_json(
                            ws,
                            {
                                **_ws_error_payload(TTSAllFailedError('Streaming TTS interrupted after audio started', attempts)),
                                'provider': candidate.get('provider'),
                                'model': candidate.get('model'),
                                'voice': candidate.get('voice'),
                                'language_type': candidate.get('language_type'),
                            },
                        )
                        return {'mode': 'stream', 'attempts': attempts, 'interrupted': True}
                    break

                audio_data = _safe_get(resp, 'output', 'audio', 'data')
                if audio_data:
                    got_audio = True
                    await _ws_send_json(
                        ws,
                        {
                            'type': 'response.audio.delta',
                            'delta': str(audio_data),
                            'format': _guess_stream_audio_format(resp, candidate),
                            'sample_rate': _guess_stream_sample_rate(resp),
                            'provider': candidate['provider'],
                            'model': candidate['model'],
                            'voice': candidate['voice'],
                            'language_type': candidate['language_type'],
                        },
                    )

            if got_audio:
                if last_attempt.get('status_code') is None:
                    last_attempt['status_code'] = 200
                if not attempt_recorded:
                    attempts.append(last_attempt)
                _cache_candidate_success(bucket, candidate, attempts)
                return {'mode': 'stream', 'attempts': attempts, 'interrupted': False}
            if not attempt_recorded and last_attempt.get('dashscope_message') in (None, 'ok'):
                last_attempt['dashscope_message'] = 'DashScope streaming finished without audio chunks'
            if not attempt_recorded:
                attempts.append(last_attempt)
        except Exception as e:
            attempt = _build_attempt(
                candidate,
                status_code=getattr(e, 'status_code', None),
                dashscope_code=str(getattr(e, 'code', '') or '') or None,
                dashscope_message=str(getattr(e, 'message', '') or str(e)),
                request_id=str(getattr(e, 'request_id', '') or '') or None,
            )
            attempts.append(attempt)
            if got_audio:
                await _ws_send_json(
                    ws,
                    {
                        **_ws_error_payload(TTSAllFailedError('Streaming TTS interrupted after audio started', attempts)),
                        'provider': candidate.get('provider'),
                        'model': candidate.get('model'),
                        'voice': candidate.get('voice'),
                        'language_type': candidate.get('language_type'),
                    },
                )
                return {'mode': 'stream', 'attempts': attempts, 'interrupted': True}

    raise TTSAllFailedError('All streaming TTS providers failed', attempts)


async def _send_ws_fallback_audio(meta: Dict[str, Any], ws: WebSocket) -> None:
    if meta.get('audio_base64'):
        await _ws_send_json(
            ws,
            {
                'type': 'response.audio.base64',
                'audio_base64': meta['audio_base64'],
                'format': meta['format'],
                'provider': meta['provider'],
                'model': meta['model'],
                'voice': meta['voice'],
                'language_type': meta['language_type'],
            },
        )
        return
    if meta.get('audio_url'):
        await _ws_send_json(
            ws,
            {
                'type': 'response.audio.url',
                'url': meta['audio_url'],
                'format': meta['format'],
                'provider': meta['provider'],
                'model': meta['model'],
                'voice': meta['voice'],
                'language_type': meta['language_type'],
            },
        )
        return
    raise TTSAllFailedError(
        'All TTS providers failed',
        [
            _build_attempt(
                {
                    'provider': meta.get('provider'),
                    'model': meta.get('model'),
                    'voice': meta.get('voice'),
                    'language_type': meta.get('language_type'),
                },
                dashscope_message='TTS succeeded but returned no playable audio',
            )
        ],
    )
