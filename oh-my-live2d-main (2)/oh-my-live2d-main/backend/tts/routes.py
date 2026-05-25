import json
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .config import DASHSCOPE_API_KEY, DEFAULT_LANG, DEFAULT_MODEL, DEFAULT_VOICE, PCM_FORMAT, PCM_SAMPLE_RATE
from .errors import TTSAllFailedError
from .schemas import SessionState, TTSSpeakRequest
from .synthesis import _send_ws_fallback_audio, _ws_error_payload, _ws_send_json, stream_tts_with_candidates, synthesize_with_candidates
from .text_utils import _normalize_text, _split_text_for_tts
from .voices import VOICE_TYPE_TO_VOICE, normalize_voice

router = APIRouter()


@router.post('/api/tts/speak')
async def speak(req: TTSSpeakRequest) -> Dict[str, Any]:
    if not DASHSCOPE_API_KEY:
        return JSONResponse(
            status_code=500,
            content={'type': 'error', 'error': 'DASHSCOPE_API_KEY not set in environment'},
        )

    text = _normalize_text(req.text)
    if not text:
        return JSONResponse(status_code=400, content={'type': 'error', 'error': 'Empty text'})

    voice_type = (req.voice_type or 'deep_male').strip().lower()
    preferred_voice = VOICE_TYPE_TO_VOICE.get(voice_type) or normalize_voice(req.voice_type or DEFAULT_VOICE)

    try:
        meta, attempts = await synthesize_with_candidates(text, preferred_voice=preferred_voice)
    except TTSAllFailedError as e:
        return JSONResponse(status_code=500, content=e.to_dict())

    result: Dict[str, Any] = {
        'success': True,
        'provider': meta['provider'],
        'model': meta['model'],
        'voice': meta['voice'],
        'language_type': meta['language_type'],
        'format': meta['format'],
        'attempts': attempts,
    }
    if meta.get('audio_url'):
        result['audio_url'] = meta['audio_url']
    if meta.get('audio_base64'):
        result['audio_base64'] = meta['audio_base64']
    return result


@router.websocket('/ws/tts')
async def ws_tts(ws: WebSocket):
    await ws.accept()

    if not DASHSCOPE_API_KEY:
        await ws.send_text(json.dumps({'type': 'error', 'error': 'DASHSCOPE_API_KEY not set in environment'}, ensure_ascii=False))
        await ws.close()
        return

    qp = ws.query_params
    model = qp.get('model') or DEFAULT_MODEL
    voice = normalize_voice(qp.get('voice') or DEFAULT_VOICE)
    language_type = qp.get('language_type') or DEFAULT_LANG
    state = SessionState()

    await ws.send_text(
        json.dumps(
            {
                'type': 'session.ready',
                'sample_rate': PCM_SAMPLE_RATE,
                'format': PCM_FORMAT,
                'voice': voice,
                'model': model,
                'language_type': language_type,
            },
            ensure_ascii=False,
        )
    )

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get('type')

            if mtype == 'input_text_buffer.append':
                state.text_buf += str(msg.get('text') or '')
                continue

            if mtype == 'input_text_buffer.commit':
                text = _normalize_text(state.text_buf)
                state.text_buf = ''
                if not text:
                    await _ws_send_json(ws, {'type': 'error', 'error': 'Empty text buffer on commit'})
                    continue

                parts = _split_text_for_tts(text)
                fatal_error = False
                for seg in parts:
                    print(f'[TTS_SEGMENT] ws_commit len={len(seg)} text={seg}')
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
                                'All streaming and fallback TTS providers failed',
                                stream_error.attempts + fallback_error.attempts,
                            )
                            await _ws_send_json(ws, _ws_error_payload(combined_error))
                            fatal_error = True
                            break
                    await _ws_send_json(ws, {'type': 'response.segment.done'})

                if not fatal_error:
                    await _ws_send_json(ws, {'type': 'response.done'})
                continue

            if mtype == 'session.finish':
                state.closed = True
                break

            if mtype == 'audio.playback.ended':
                continue

            await ws.send_text(
                json.dumps(
                    {
                        'type': 'error',
                        'error': f'Unknown message type: {mtype}',
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
