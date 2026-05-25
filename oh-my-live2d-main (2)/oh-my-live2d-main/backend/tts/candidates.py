import json
import threading
from typing import Any, Dict, List, Optional, Tuple

from .config import CHINESE_PROBE_TEXT, DEFAULT_LANG, DEFAULT_MODEL, DEFAULT_PROBE_TEXT, DEFAULT_VOICE
from .voices import SUPPORTED_VOICES

_probe_cache: Dict[str, Dict[str, Any]] = {}
_probe_attempts: Dict[str, List[Dict[str, Any]]] = {}
_probe_lock = threading.Lock()


def _dedupe_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get('provider') or '').strip(),
            str(item.get('model') or '').strip(),
            str(item.get('voice') or '').strip(),
            str(item.get('language_type') or '').strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                'provider': key[0],
                'model': key[1],
                'voice': key[2],
                'language_type': key[3] or DEFAULT_LANG,
            }
        )
    return out


def _default_tts_candidates() -> List[Dict[str, Any]]:
    voices = []
    for voice in [DEFAULT_VOICE, *SUPPORTED_VOICES]:
        if voice and voice not in voices:
            voices.append(voice)

    candidates: List[Dict[str, Any]] = []
    for voice in voices:
        for lang in ('Chinese', 'Auto'):
            candidates.append(
                {
                    'provider': 'qwen_tts',
                    'model': DEFAULT_MODEL,
                    'voice': voice,
                    'language_type': lang,
                }
            )
    for voice in voices:
        for lang in ('Chinese', 'Auto'):
            candidates.append(
                {
                    'provider': 'multimodal',
                    'model': DEFAULT_MODEL,
                    'voice': voice,
                    'language_type': lang,
                }
            )
    return _dedupe_candidates(candidates)


def _load_tts_candidates() -> List[Dict[str, Any]]:
    raw = os.getenv('DASHSCOPE_TTS_CANDIDATES')
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


try:
    import os
    TTS_CANDIDATES = _load_tts_candidates()
finally:
    pass


def _candidate_matches_voice(candidate: Dict[str, Any], preferred_voice: Optional[str]) -> bool:
    if not preferred_voice:
        return False
    return str(candidate.get('voice') or '').strip().lower() == preferred_voice.strip().lower()


def _build_attempt(
    candidate: Dict[str, Any],
    *,
    status_code: Optional[int] = None,
    dashscope_code: Optional[str] = None,
    dashscope_message: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        'provider': candidate.get('provider'),
        'model': candidate.get('model'),
        'voice': candidate.get('voice'),
        'language_type': candidate.get('language_type'),
        'status_code': status_code,
        'dashscope_code': dashscope_code,
        'dashscope_message': dashscope_message,
        'request_id': request_id,
    }


def _cache_candidate_success(bucket: str, candidate: Dict[str, Any], attempts: List[Dict[str, Any]]) -> None:
    with _probe_lock:
        _probe_cache[bucket] = {
            'provider': candidate['provider'],
            'model': candidate['model'],
            'voice': candidate['voice'],
            'language_type': candidate['language_type'],
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
        if cached and all(candidate.get(k) == cached.get(k) for k in ('provider', 'model', 'voice', 'language_type')):
            continue
        if _candidate_matches_voice(candidate, preferred_voice):
            matches.append(candidate)
        else:
            others.append(candidate)
    return _dedupe_candidates(ordered + matches + others)


async def _probe_bucket(bucket: str, preferred_voice: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    from .dashscope_client import _call_tts_candidate

    probe_text = CHINESE_PROBE_TEXT if bucket == 'zh' else DEFAULT_PROBE_TEXT
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
