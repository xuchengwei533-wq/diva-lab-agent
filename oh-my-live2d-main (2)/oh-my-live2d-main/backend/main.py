import os
import json
import asyncio
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Oh-My-Live2D Gateway", version="1.1.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASR_BASE_URL = os.getenv("ASR_BASE_URL", "http://127.0.0.1:8006").rstrip("/")
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://127.0.0.1:8004").rstrip("/")

class ASRRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    model: Optional[str] = None

class ASRResponse(BaseModel):
    text: str
    success: bool
    error: Optional[str] = None

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_type: Optional[str] = "cute"

class TTSResponse(BaseModel):
    success: bool
    audio_base64: Optional[str] = None
    error: Optional[str] = None

class VoiceStartReq(BaseModel):
    audio_data: Optional[str] = None
    audio_base64: Optional[str] = None
    audio_format: str = "pcm16le_16k_mono"


async def _request_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout_s: int = 60) -> Any:
    def _run():
        if method.upper() == "GET":
            return requests.get(url, timeout=timeout_s)
        return requests.post(url, json=payload, timeout=timeout_s)

    try:
        resp = await asyncio.to_thread(_run)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Downstream request failed: {e}")

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=data)

    return data

@app.get("/")
async def root():
    return {"service": "gateway", "asr_base_url": ASR_BASE_URL, "tts_base_url": TTS_BASE_URL}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "gateway", "asr_base_url": ASR_BASE_URL, "tts_base_url": TTS_BASE_URL}

@app.post("/api/asr/recognize", response_model=ASRResponse)
async def recognize_audio(request: ASRRequest):
    """HTTP API端点：识别音频文件"""
    try:
        data = await _request_json("POST", f"{ASR_BASE_URL}/api/asr/recognize", request.model_dump())
        return ASRResponse(text=str(data.get("text") or ""), success=bool(data.get("success", True)), error=data.get("error"))
    except HTTPException as e:
        return ASRResponse(text="", success=False, error=str(e.detail))
    except Exception as e:
        return ASRResponse(text="", success=False, error=str(e))

@app.post("/api/tts/speak", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """HTTP API端点：文本转语音"""
    try:
        data = await _request_json("POST", f"{TTS_BASE_URL}/api/tts/speak", request.model_dump())
        return TTSResponse(success=bool(data.get("success", True)), audio_base64=data.get("audio_base64"), error=data.get("error"))
    except HTTPException as e:
        return TTSResponse(success=False, error=str(e.detail))
    except Exception as e:
        return TTSResponse(success=False, error=str(e))


@app.post("/api/voice/start")
async def voice_start(req: VoiceStartReq):
    """语音识别开始接口"""
    payload = req.model_dump()
    if not payload.get("audio_data") and payload.get("audio_base64"):
        payload["audio_data"] = payload["audio_base64"]
    return await _request_json("POST", f"{ASR_BASE_URL}/api/voice/start", payload)


@app.get("/api/voice/text")
async def voice_text():
    """获取语音识别文本结果"""
    return await _request_json("GET", f"{ASR_BASE_URL}/api/voice/text")


@app.post("/api/voice/clear")
async def voice_clear():
    """清除语音识别结果"""
    return await _request_json("POST", f"{ASR_BASE_URL}/api/voice/clear", {})

@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data.get("type") != "audio_data":
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "未知消息类型",
                    "success": False,
                }, ensure_ascii=False))
                continue

            audio_base64 = data.get("audio_data") or data.get("audio_base64") or ""
            if not audio_base64:
                await websocket.send_text(json.dumps({
                    "type": "asr_error",
                    "error": "Missing audio_data",
                    "success": False,
                }, ensure_ascii=False))
                continue

            try:
                payload = {
                    "audio_data": audio_base64,
                    "audio_format": data.get("audio_format") or "pcm16le_16k_mono",
                }
                res = await _request_json("POST", f"{ASR_BASE_URL}/api/voice/start", payload)
                await websocket.send_text(json.dumps({
                    "type": "asr_result",
                    "text": str(res.get("text") or ""),
                    "success": bool(res.get("success", True)),
                }, ensure_ascii=False))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "asr_error",
                    "error": str(e),
                    "success": False,
                }, ensure_ascii=False))
                
    except WebSocketDisconnect:
        logger.info("WebSocket连接已断开")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": str(e),
                "success": False
            }))
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"ASR_BASE_URL={ASR_BASE_URL}")
    logger.info(f"TTS_BASE_URL={TTS_BASE_URL}")
    logger.info("启动 Oh-My-Live2D Gateway...")
    logger.info("服务地址: http://localhost:8002")
    logger.info("API文档: http://localhost:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
