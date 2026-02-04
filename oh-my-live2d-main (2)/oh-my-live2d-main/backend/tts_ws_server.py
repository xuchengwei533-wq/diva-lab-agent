# tts_ws_server.py (NON-STREAMING VERSION)
import os
import json
import asyncio
from typing import Any, Dict, Optional

import dashscope
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ============= 基础配置 =============
dashscope.base_http_api_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 必须配置

DEFAULT_MODEL = os.getenv("DASHSCOPE_TTS_MODEL", "qwen3-tts-flash")
DEFAULT_VOICE = os.getenv("DASHSCOPE_TTS_VOICE", "Cherry")
DEFAULT_LANG  = os.getenv("DASHSCOPE_TTS_LANG", "Chinese")

# 非流式合成：为了避免“提交过短导致效果差/请求频繁”，这里保留最小长度限制
MIN_CHARS_PER_REQ = int(os.getenv("TTS_MIN_CHARS_PER_REQ", "24"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 演示环境先放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_get(d: Any, *keys, default=None):
    """兼容 dict / SDK 对象两种结构的安全取值。"""
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


def _extract_audio_meta(resp: Any) -> Dict[str, Any]:
    """
    从 DashScope response 中提取音频信息：
    - url / expires_at（最关键，前端用 <audio> 播放）
    - data（可选：如果 SDK 返回 base64 音频，也一并带回去）
    """
    # 常见结构：resp.output.audio.url / resp.output.audio.expires_at
    out = _safe_get(resp, "output", default=None)
    audio = _safe_get(out, "audio", default=None)

    url = _safe_get(audio, "url", default=None)
    expires_at = _safe_get(audio, "expires_at", default=None)
    data = _safe_get(audio, "data", default=None)  # 有时会有 base64（不保证）

    # 也可能是 dict：resp["output"]["audio"]["url"]
    # _safe_get 已经兼容

    meta = {"url": url, "expires_at": expires_at}
    if data:
        meta["data"] = data  # 可选：不推荐前端用它播放（体积大），但留作排障
    return meta


def _normalize_text(text: str) -> str:
    return (text or "").strip()


def _split_text_for_tts(text: str) -> list[str]:
    """
    非流式：一次合成建议不要太长（也避免超过模型/接口限制）。
    默认尽量整段合成（更连贯）。仅在过长时才切分。
    """
    text = _normalize_text(text)
    if not text:
        return []

    # 你可按需调整
    MAX_CHARS = int(os.getenv("TTS_MAX_CHARS_PER_REQ", "300"))

    if len(text) <= MAX_CHARS:
        return [text]

    seps = set("。！？!?；;\n")
    chunks = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in seps and len(cur) >= MIN_CHARS_PER_REQ:
            seg = "".join(cur).strip()
            if seg:
                chunks.append(seg)
            cur = []
        # 过长强制切
        if len(cur) >= MAX_CHARS:
            seg = "".join(cur).strip()
            if seg:
                chunks.append(seg)
            cur = []

    tail = "".join(cur).strip()
    if tail:
        chunks.append(tail)
    return [c for c in chunks if c]


async def _dashscope_tts_nonstream(text: str, model: str, voice: str, language_type: str) -> Any:
    """
    非流式 TTS：用线程执行 SDK 同步调用，避免阻塞事件循环。
    """
    def _run():
        return dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=model,
            text=text,
            voice=voice,
            language_type=language_type,
            # 注意：不传 stream 或 stream=False 即为非流式
        )

    return await asyncio.to_thread(_run)


class SessionState:
    def __init__(self):
        self.text_buf: str = ""
        self.closed: bool = False


@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    """
    前端协议（兼容你当前写法）：
    - {type:"input_text_buffer.append", text:"..."}   # 只缓冲
    - {type:"input_text_buffer.commit"}              # 触发一次性合成（非流式）
    - {type:"session.finish"}                        # 关闭
    返回：
    - session.ready
    - response.audio.meta   (meta: {url, expires_at, data?})
    - response.segment.done (每段结束)
    - response.done         (本次 commit 全部结束)
    - error
    """
    await ws.accept()

    if not DASHSCOPE_API_KEY:
        await ws.send_text(json.dumps({
            "type": "error",
            "error": "Missing DASHSCOPE_API_KEY in server environment"
        }, ensure_ascii=False))
        await ws.close()
        return

    qp = ws.query_params
    model = qp.get("model") or DEFAULT_MODEL
    voice = qp.get("voice") or DEFAULT_VOICE
    language_type = qp.get("language_type") or DEFAULT_LANG

    state = SessionState()

    await ws.send_text(json.dumps({
        "type": "session.ready",
        "model": model,
        "voice": voice,
        "language_type": language_type,
        "mode": "non_streaming"
    }, ensure_ascii=False))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "input_text_buffer.append":
                piece = (msg.get("text") or "")
                state.text_buf += piece

            elif mtype == "input_text_buffer.commit":
                text = _normalize_text(state.text_buf)
                state.text_buf = ""

                if not text:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "error": "Empty text buffer on commit"
                    }, ensure_ascii=False))
                    continue

                # 可选：按句切段，逐段合成并逐段播放（仍然是“非流式合成”，只是多次请求）
                parts = _split_text_for_tts(text)

                for seg in parts:
                    try:
                        resp = await _dashscope_tts_nonstream(seg, model, voice, language_type)
                        meta = _extract_audio_meta(resp)

                        if not meta.get("url") and not meta.get("data"):
                            # 返回里找不到音频信息，直接把原响应结构（裁剪）回传便于排障
                            await ws.send_text(json.dumps({
                                "type": "error",
                                "error": "TTS returned no audio url/data",
                                "debug": {
                                    "has_output": bool(_safe_get(resp, "output", default=None)),
                                    "finish_reason": _safe_get(_safe_get(resp, "output", default=None), "finish_reason", default=None),
                                }
                            }, ensure_ascii=False))
                        else:
                            await ws.send_text(json.dumps({
                                "type": "response.audio.meta",
                                "meta": meta
                            }, ensure_ascii=False))

                        await ws.send_text(json.dumps({
                            "type": "response.segment.done"
                        }, ensure_ascii=False))

                    except Exception as e:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "error": f"TTS failed: {repr(e)}"
                        }, ensure_ascii=False))

                await ws.send_text(json.dumps({
                    "type": "response.done"
                }, ensure_ascii=False))

            elif mtype == "session.finish":
                state.closed = True
                break

            else:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "error": f"Unknown message type: {mtype}"
                }, ensure_ascii=False))

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
