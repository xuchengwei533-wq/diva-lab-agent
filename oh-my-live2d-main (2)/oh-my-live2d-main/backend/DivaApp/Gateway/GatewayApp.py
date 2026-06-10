from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from DivaApp.Configuration import GatewaySettings, LoadAppSettings


Logger = logging.getLogger(__name__)


class AsrRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    model: Optional[str] = None


class AsrResponse(BaseModel):
    text: str
    success: bool
    error: Optional[str] = None


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_type: Optional[str] = "deep_male"


class TtsResponse(BaseModel):
    success: bool
    audio_base64: Optional[str] = None
    error: Optional[str] = None


class VoiceStartRequest(BaseModel):
    audio_data: Optional[str] = None
    audio_base64: Optional[str] = None
    audio_format: str = "pcm16le_16k_mono"


async def RequestJson(Method: str, Url: str, Payload: Optional[Dict[str, Any]] = None, TimeoutSeconds: int = 60) -> Any:
    def SendRequest() -> requests.Response:
        if Method.upper() == "GET":
            return requests.get(Url, timeout=TimeoutSeconds)
        return requests.post(Url, json=Payload, timeout=TimeoutSeconds)

    try:
        Response = await asyncio.to_thread(SendRequest)
    except requests.RequestException as Error:
        raise HTTPException(status_code=502, detail=f"Downstream request failed: {Error}") from Error

    try:
        Data = Response.json()
    except Exception:
        Data = Response.text

    if Response.status_code >= 400:
        raise HTTPException(status_code=Response.status_code, detail=Data)

    return Data


def CreateGatewayApp(Settings: GatewaySettings | None = None) -> FastAPI:
    Settings = Settings or LoadAppSettings().Gateway
    App = FastAPI(title="Oh-My-Live2D Gateway", version="1.1.0")

    App.add_middleware(
        CORSMiddleware,
        allow_origins=list(Settings.CorsAllowOrigins),
        allow_credentials="*" not in Settings.CorsAllowOrigins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @App.get("/")
    async def Root() -> Dict[str, str]:
        return {
            "service": "gateway",
            "asr_base_url": Settings.AsrBaseUrl,
            "tts_base_url": Settings.TtsBaseUrl,
        }

    @App.get("/health")
    async def HealthCheck() -> Dict[str, str]:
        return {
            "status": "ok",
            "service": "gateway",
            "asr_base_url": Settings.AsrBaseUrl,
            "tts_base_url": Settings.TtsBaseUrl,
        }

    @App.post("/api/asr/recognize", response_model=AsrResponse)
    async def RecognizeAudio(Request: AsrRequest) -> AsrResponse:
        try:
            Data = await RequestJson("POST", f"{Settings.AsrBaseUrl}/api/asr/recognize", Request.model_dump())
            return AsrResponse(
                text=str(Data.get("text") or ""),
                success=bool(Data.get("success", True)),
                error=Data.get("error"),
            )
        except HTTPException as Error:
            return AsrResponse(text="", success=False, error=str(Error.detail))
        except Exception as Error:
            return AsrResponse(text="", success=False, error=str(Error))

    @App.post("/api/tts/speak", response_model=TtsResponse)
    async def TextToSpeech(Request: TtsRequest) -> TtsResponse:
        try:
            Data = await RequestJson("POST", f"{Settings.TtsBaseUrl}/api/tts/speak", Request.model_dump())
            return TtsResponse(
                success=bool(Data.get("success", True)),
                audio_base64=Data.get("audio_base64"),
                error=Data.get("error"),
            )
        except HTTPException as Error:
            return TtsResponse(success=False, error=str(Error.detail))
        except Exception as Error:
            return TtsResponse(success=False, error=str(Error))

    @App.post("/api/voice/start")
    async def VoiceStart(Request: VoiceStartRequest) -> Any:
        Payload = Request.model_dump()
        if not Payload.get("audio_data") and Payload.get("audio_base64"):
            Payload["audio_data"] = Payload["audio_base64"]
        return await RequestJson("POST", f"{Settings.AsrBaseUrl}/api/voice/start", Payload)

    @App.get("/api/voice/text")
    async def VoiceText() -> Any:
        return await RequestJson("GET", f"{Settings.AsrBaseUrl}/api/voice/text")

    @App.post("/api/voice/clear")
    async def VoiceClear() -> Any:
        return await RequestJson("POST", f"{Settings.AsrBaseUrl}/api/voice/clear", {})

    @App.websocket("/ws/asr")
    async def WebSocketAsr(Socket: WebSocket) -> None:
        await Socket.accept()
        try:
            while True:
                RawMessage = await Socket.receive_text()
                Data = json.loads(RawMessage)

                if Data.get("type") != "audio_data":
                    await Socket.send_text(json.dumps({
                        "type": "error",
                        "message": "Unknown message type",
                        "success": False,
                    }, ensure_ascii=False))
                    continue

                AudioBase64 = Data.get("audio_data") or Data.get("audio_base64") or ""
                if not AudioBase64:
                    await Socket.send_text(json.dumps({
                        "type": "asr_error",
                        "error": "Missing audio_data",
                        "success": False,
                    }, ensure_ascii=False))
                    continue

                try:
                    Payload = {
                        "audio_data": AudioBase64,
                        "audio_format": Data.get("audio_format") or "pcm16le_16k_mono",
                    }
                    Result = await RequestJson("POST", f"{Settings.AsrBaseUrl}/api/voice/start", Payload)
                    await Socket.send_text(json.dumps({
                        "type": "asr_result",
                        "text": str(Result.get("text") or ""),
                        "success": bool(Result.get("success", True)),
                    }, ensure_ascii=False))
                except Exception as Error:
                    await Socket.send_text(json.dumps({
                        "type": "asr_error",
                        "error": str(Error),
                        "success": False,
                    }, ensure_ascii=False))

        except WebSocketDisconnect:
            Logger.info("Gateway ASR WebSocket disconnected")
        except Exception as Error:
            Logger.error("Gateway ASR WebSocket error: %s", Error)
            try:
                await Socket.send_text(json.dumps({
                    "type": "error",
                    "error": str(Error),
                    "success": False,
                }, ensure_ascii=False))
            except Exception:
                pass

    return App


App = CreateGatewayApp()

