# qwen_chat_server.py
import os
import json
from http import HTTPStatus
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from dashscope import Application

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

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("Missing env DASHSCOPE_API_KEY. Please set it in .env or environment variables.")

# 你的百炼智能体应用 ID
BAILIAN_APP_ID = "4dc0700043fc46679e1568339e580678"

# ---------- app ----------
app = FastAPI(title="Local Gateway for Bailian Agent App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发期放开；生产请收紧域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatStreamRequest(BaseModel):
    # 兼容你前端现有字段（model 不再需要，但保留以免前端报错）
    model: str = Field(default="ignored")
    messages: List[Dict[str, Any]] = Field(default_factory=list)

    # 可选：如果你未来想走云端 session_id（不传 messages），可以用下面两个
    prompt: Optional[str] = None
    session_id: Optional[str] = None

    # 可选：透传给百炼智能体（按需用）
    biz_params: Optional[Dict[str, Any]] = None
    rag_options: Optional[Dict[str, Any]] = None
    memory_id: Optional[str] = None
    has_thoughts: Optional[bool] = None

    # 兼容你之前的 extra_body（如果前端有传也不报错）
    extra_body: Optional[Dict[str, Any]] = None

@app.get("/health")
def health():
    return {"ok": True}

def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    百炼智能体 messages 一般使用 {role, content}。
    这里做两件事：
    1) 过滤 system（通常智能体已在控制台配置了系统提示词；重复传 system 容易“串味”）
    2) 只保留 user/assistant
    """
    norm: List[Dict[str, str]] = []
    for m in messages or []:
        role = (m.get("role") or "").strip()
        content = m.get("content")
        if content is None:
            continue
        content = str(content)

        if role == "system":
            # 默认丢弃 system；如你确实要强行覆盖智能体提示词，可改为转成 user
            continue

        if role not in ("user", "assistant"):
            # 其他角色忽略
            continue

        norm.append({"role": role, "content": content})
    return norm

@app.post("/api/chat/stream")
def api_chat_stream(payload: ChatStreamRequest):
    """
    SSE (text/event-stream) 转发百炼智能体流式输出给前端。
    前端按 data: {...}\n\n 解析即可（与你现有 HTML 的解析逻辑匹配）。
    """

    # 组装参数：优先用 messages；若 messages 为空且给了 session_id，则用 prompt + session_id
    messages = _normalize_messages(payload.messages)
    use_session_mode = (not messages) and bool(payload.session_id)

    # 百炼 SDK 参数：尽量只传“已知字段”，避免未知字段报错
    call_kwargs: Dict[str, Any] = {
        "api_key": DASHSCOPE_API_KEY,
        "app_id": BAILIAN_APP_ID,
        "stream": True,
        "incremental_output": True,  # 让每个 chunk.output.text 都是增量，便于你前端 delta 拼接
    }

    if payload.biz_params:
        call_kwargs["biz_params"] = payload.biz_params
    if payload.rag_options:
        call_kwargs["rag_options"] = payload.rag_options
    if payload.memory_id:
        call_kwargs["memory_id"] = payload.memory_id
    if payload.has_thoughts is True:
        call_kwargs["has_thoughts"] = True

    # 兼容 extra_body：允许你从前端透传 biz_params/rag_options/memory_id 等
    if payload.extra_body:
        for k in ("biz_params", "rag_options", "memory_id", "has_thoughts"):
            if k in payload.extra_body and k not in call_kwargs:
                call_kwargs[k] = payload.extra_body[k]

    if use_session_mode:
        # 云端存储：只传 prompt + session_id（messages 为空时生效）
        prompt = payload.prompt
        if not prompt:
            prompt = "你好"
        call_kwargs["prompt"] = prompt
        call_kwargs["session_id"] = payload.session_id
    else:
        # 自行管理：传 messages（若同时传 session_id，文档里说会忽略 session_id）
        if not messages:
            # 兜底：至少给一个 prompt，避免空请求
            call_kwargs["prompt"] = payload.prompt or "你好"
        else:
            call_kwargs["messages"] = messages

    def event_stream():
        try:
            responses = Application.call(**call_kwargs)

            for resp in responses:
                if resp.status_code != HTTPStatus.OK:
                    yield f"data: {json.dumps({'type':'error','error': resp.message, 'code': int(resp.status_code), 'request_id': resp.request_id}, ensure_ascii=False)}\n\n"
                    continue

                # 流式增量文本
                out = getattr(resp, "output", None)
                if out and getattr(out, "text", None):
                    delta = str(out.text)
                    if delta:
                        yield f"data: {json.dumps({'type':'delta','delta': delta}, ensure_ascii=False)}\n\n"

                # 你若想把 session_id 返回给前端（用于下次走 session 模式），可打开下面这段
                # if out and getattr(out, "session_id", None):
                #     yield f"data: {json.dumps({'type':'meta','session_id': out.session_id}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type':'finish','finish_reason':'stop'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/chat")
def api_chat(payload: ChatStreamRequest):
    """
    非流式（调试用）
    """
    try:
        messages = _normalize_messages(payload.messages)
        call_kwargs: Dict[str, Any] = {
            "api_key": DASHSCOPE_API_KEY,
            "app_id": BAILIAN_APP_ID,
        }

        if payload.biz_params:
            call_kwargs["biz_params"] = payload.biz_params
        if payload.rag_options:
            call_kwargs["rag_options"] = payload.rag_options
        if payload.memory_id:
            call_kwargs["memory_id"] = payload.memory_id
        if payload.has_thoughts is True:
            call_kwargs["has_thoughts"] = True

        if messages:
            call_kwargs["messages"] = messages
        else:
            call_kwargs["prompt"] = payload.prompt or "你好"
            if payload.session_id:
                call_kwargs["session_id"] = payload.session_id

        resp = Application.call(**call_kwargs)
        if resp.status_code != HTTPStatus.OK:
            return JSONResponse(
                status_code=500,
                content={"error": resp.message, "code": int(resp.status_code), "request_id": resp.request_id},
            )

        # 只回传 text（你也可把 resp.output 完整 dump 回去）
        return JSONResponse(content={"text": resp.output.text, "session_id": getattr(resp.output, "session_id", None)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("qwen_chat_server:app", host="0.0.0.0", port=8003, reload=False)
