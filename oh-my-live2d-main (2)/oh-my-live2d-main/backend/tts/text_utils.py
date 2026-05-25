from typing import List, Optional

from .config import MAX_CHARS_PER_REQ, MIN_CHARS_PER_REQ, MIN_TTS_MERGE_CHARS


def _is_chinese_text(text: str) -> bool:
    for ch in text or '':
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def _text_bucket(text: str) -> str:
    return 'zh' if _is_chinese_text(text) else 'default'


def _normalize_text(text: Optional[str]) -> str:
    return (text or '').strip()


def _split_text_for_tts(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    strong_seps = set('。！？!?；;\n')
    soft_seps = set('，,、：:')
    chunks: List[str] = []
    cur: List[str] = []
    last_soft_idx = -1

    def flush(end_idx: Optional[int] = None):
        nonlocal cur, last_soft_idx
        if not cur:
            return
        if end_idx is None or end_idx >= len(cur):
            segment = ''.join(cur).strip()
            cur = []
        else:
            segment = ''.join(cur[:end_idx]).strip()
            cur = cur[end_idx:]
        last_soft_idx = -1
        if segment:
            chunks.append(segment)

    for ch in text:
        cur.append(ch)
        if ch in soft_seps:
            last_soft_idx = len(cur)

        if ch in strong_seps and len(cur) >= MIN_CHARS_PER_REQ:
            flush()
            continue

        if len(cur) >= MAX_CHARS_PER_REQ:
            if last_soft_idx >= MIN_CHARS_PER_REQ:
                flush(last_soft_idx)
            else:
                flush()

    tail = ''.join(cur).strip()
    if tail:
        chunks.append(tail)

    merged: List[str] = []
    for seg in chunks:
        value = seg.strip()
        if not value:
            continue
        if merged and len(value) < MIN_TTS_MERGE_CHARS:
            merged[-1] = (merged[-1] + value).strip()
            continue
        merged.append(value)

    if len(merged) >= 2 and len(merged[-1]) < MIN_TTS_MERGE_CHARS:
        merged[-2] = (merged[-2] + merged[-1]).strip()
        merged.pop()

    return merged
