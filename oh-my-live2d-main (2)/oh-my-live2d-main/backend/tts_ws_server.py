import os
import json
import asyncio
import threading
from typing import Any, Dict, Optional

import dashscope
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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
                    if not line:
                        continue
                    if line.startswith("#"):
                        continue
                    if "=" not in line:
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
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 必须配置

DEFAULT_MODEL = os.getenv("DASHSCOPE_TTS_MODEL", "qwen3-tts-flash")
DEFAULT_VOICE = os.getenv("DASHSCOPE_TTS_VOICE", "Cherry")
DEFAULT_LANG  = os.getenv("DASHSCOPE_TTS_LANG", "Chinese")

# 为了避免“提交过短导致效果差/请求频繁”，这里保留最小长度限制
MIN_CHARS_PER_REQ = int(os.getenv("TTS_MIN_CHARS_PER_REQ", "24"))
PLAYBACK_ACK_TIMEOUT_SEC = float(os.getenv("TTS_PLAYBACK_ACK_TIMEOUT_SEC", "20"))

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


async def _dashscope_tts_stream(text: str, model: str, voice: str, language_type: str):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)

    def _run():
        try:
            resp_iter = dashscope.MultiModalConversation.call(
                api_key=DASHSCOPE_API_KEY,
                model=model,
                text=text,
                voice=voice,
                language_type=language_type,
                stream=True,
            )

            for chunk in resp_iter:
                fut = asyncio.run_coroutine_threadsafe(q.put(chunk), loop)
                fut.result()
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(e), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

    threading.Thread(target=_run, daemon=True).start()

    while True:
        item = await q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item



class SessionState:
    def __init__(self):
        self.text_buf: str = ""
        self.closed: bool = False
        self.commit_seq: int = 0
        self.commit_q: asyncio.Queue[Any] = asyncio.Queue()
        self.segment_seq: int = 0
        self.ready_q: asyncio.Queue[Any] = asyncio.Queue()
        self.playback_ready: asyncio.Event = asyncio.Event()
        self.playing_segment_id: Optional[int] = None
        self.playback_timeout_task: Optional[asyncio.Task] = None
        self.worker_task: Optional[asyncio.Task] = None
        self.sender_task: Optional[asyncio.Task] = None


def _extract_audio_data(resp: Any) -> Optional[str]:
    out = _safe_get(resp, "output", default=None)
    audio = _safe_get(out, "audio", default=None)
    data = _safe_get(audio, "data", default=None)
    return data if isinstance(data, str) and data else None


def _extract_finish_reason(resp: Any) -> Optional[str]:
    out = _safe_get(resp, "output", default=None)
    finish_reason = _safe_get(out, "finish_reason", default=None)
    return finish_reason if isinstance(finish_reason, str) and finish_reason else None


@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    """
    前端协议（兼容你当前写法）：
    - {type:"input_text_buffer.append", text:"..."}   # 只缓冲
    - {type:"input_text_buffer.commit"}              # 触发合成
    - {type:"session.finish"}                        # 关闭
    返回：
    - session.ready
    - response.audio.delta  (delta: base64 pcm16 @ 24k)
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
        "mode": "streaming"
    }, ensure_ascii=False))

    async def _send(obj: Dict[str, Any]):
        await ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def _auto_release(segment_id: int, timeout_sec: float):
        try:
            await asyncio.sleep(timeout_sec)
            if state.playing_segment_id == segment_id and not state.playback_ready.is_set():
                state.playback_ready.set()
        except asyncio.CancelledError:
            return

    async def _sender_worker():
        try:
            while True:
                item = await state.ready_q.get()
                if item is None:
                    break

                await state.playback_ready.wait()
                if state.closed:
                    break

                state.playback_ready.clear()

                commit_id = item.get("commit_id")
                segment_id = item.get("segment_id")
                deltas = item.get("deltas") or []
                is_last_in_commit = bool(item.get("is_last_in_commit"))

                state.playing_segment_id = segment_id
                if state.playback_timeout_task:
                    try:
                        state.playback_timeout_task.cancel()
                    except Exception:
                        pass
                    state.playback_timeout_task = None

                for delta in deltas:
                    await _send({
                        "type": "response.audio.delta",
                        "delta": delta,
                        "commit_id": commit_id,
                        "segment_id": segment_id,
                    })

                await _send({
                    "type": "response.segment.done",
                    "commit_id": commit_id,
                    "segment_id": segment_id,
                })

                if is_last_in_commit:
                    await _send({"type": "response.done", "commit_id": commit_id})

                if segment_id is not None and PLAYBACK_ACK_TIMEOUT_SEC > 0:
                    state.playback_timeout_task = asyncio.create_task(
                        _auto_release(int(segment_id), PLAYBACK_ACK_TIMEOUT_SEC)
                    )

        except Exception as e:
            await _send({"type": "error", "error": f"TTS sender failed: {repr(e)}"})

    async def _commit_worker():
        try:
            while True:
                item = await state.commit_q.get()
                if item is None:
                    break

                commit_id, parts = item

                for idx, seg in enumerate(parts):
                    deltas: list[str] = []
                    state.segment_seq += 1
                    segment_id = state.segment_seq

                    async for chunk in _dashscope_tts_stream(seg, model, voice, language_type):
                        data = _extract_audio_data(chunk)
                        if data:
                            deltas.append(data)

                        finish_reason = _extract_finish_reason(chunk)
                        if finish_reason == "stop":
                            pass

                    await state.ready_q.put({
                        "commit_id": commit_id,
                        "segment_id": segment_id,
                        "deltas": deltas,
                        "is_last_in_commit": (idx == len(parts) - 1),
                    })

        except Exception as e:
            await _send({"type": "error", "error": f"TTS worker failed: {repr(e)}"})

    state.playback_ready.set()
    state.sender_task = asyncio.create_task(_sender_worker())
    state.worker_task = asyncio.create_task(_commit_worker())

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
                    await _send({"type": "error", "error": "Empty text buffer on commit"})
                    continue

                parts = _split_text_for_tts(text)
                if not parts:
                    await _send({"type": "error", "error": "Empty text after split"})
                    continue

                state.commit_seq += 1
                await state.commit_q.put((state.commit_seq, parts))

            elif mtype == "audio.playback.ended":
                sid = msg.get("segment_id", None)
                if sid is None or sid == state.playing_segment_id:
                    if state.playback_timeout_task:
                        try:
                            state.playback_timeout_task.cancel()
                        except Exception:
                            pass
                        state.playback_timeout_task = None
                    state.playing_segment_id = None
                    state.playback_ready.set()

            elif mtype == "session.finish":
                state.closed = True
                break

            else:
                await _send({"type": "error", "error": f"Unknown message type: {mtype}"})

    except WebSocketDisconnect:
        state.closed = True
    finally:
        try:
            await state.commit_q.put(None)
        except Exception:
            pass
        try:
            await state.ready_q.put(None)
        except Exception:
            pass
        try:
            if state.playback_timeout_task:
                state.playback_timeout_task.cancel()
        except Exception:
            pass
        try:
            if state.worker_task:
                await asyncio.wait_for(state.worker_task, timeout=3)
        except Exception:
            try:
                if state.worker_task:
                    state.worker_task.cancel()
            except Exception:
                pass
        try:
            if state.sender_task:
                await asyncio.wait_for(state.sender_task, timeout=3)
        except Exception:
            try:
                if state.sender_task:
                    state.sender_task.cancel()
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
