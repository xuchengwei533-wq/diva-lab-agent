import os
import json
import base64
import tempfile
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import wave
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Oh-My-Live2D ASR Service", version="1.0.0")

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

# 阿里云百炼API配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# Fallback to paraformer-realtime-v1 for better compatibility
ASR_MODEL = os.getenv("DASHSCOPE_ASR_MODEL", "paraformer-realtime-v1")

class ASRRequest(BaseModel):
    audio_base64: str
    model: Optional[str] = ASR_MODEL

class ASRResponse(BaseModel):
    text: str
    success: bool
    error: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    voice_type: Optional[str] = "cute"

class TTSResponse(BaseModel):
    success: bool
    audio_base64: Optional[str] = None
    error: Optional[str] = None

class WebSocketMessage(BaseModel):
    type: str
    data: dict

class ASRService:
    def __init__(self):
        self.api_key = DASHSCOPE_API_KEY
        self.model = ASR_MODEL
        
    def _decode_base64_to_bytes(self, audio_base64: str) -> bytes:
        """将base64音频数据转换为字节"""
        try:
            # 移除data URL前缀
            if "base64," in audio_base64:
                audio_base64 = audio_base64.split("base64,")[1]
            
            return base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f"音频数据转换失败: {e}")
            raise
    
    def _pcm_bytes_to_wav_file(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        if not pcm_bytes:
            raise ValueError("Empty audio payload")

        if len(pcm_bytes) % 2 != 0:
            raise ValueError(f"PCM byte length not aligned to int16: len={len(pcm_bytes)}")

        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="asr_", text=False)
        os.close(fd)

        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)
        except Exception:
            try:
                os.remove(wav_path)
            except Exception:
                pass
            raise

        return wav_path

    def _extract_text_from_resp(self, resp: object) -> str:
        try:
            if resp is None:
                return ""
            data = resp
            if hasattr(resp, "to_dict"):
                try:
                    data = resp.to_dict()
                except Exception:
                    data = resp
            if isinstance(data, dict):
                if isinstance(data.get("text"), str) and data["text"].strip():
                    return data["text"].strip()
                output = data.get("output")
                if isinstance(output, dict):
                    for k in ("text", "sentence", "transcription", "result"):
                        v = output.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                        # Fix: Check inside sentence dict
                        if k == "sentence" and isinstance(v, dict):
                            t = v.get("text")
                            if isinstance(t, str) and t.strip():
                                return t.strip()
                    if isinstance(output.get("results"), list) and output["results"]:
                        first = output["results"][0]
                        if isinstance(first, dict):
                            for k in ("text", "sentence", "transcription", "result"):
                                v = first.get(k)
                                if isinstance(v, str) and v.strip():
                                    return v.strip()
            if hasattr(resp, "output") and hasattr(resp.output, "text"):
                v = getattr(resp.output, "text", None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        except Exception:
            return ""
        return ""
    
    def recognize(self, audio_base64: str) -> str:
        """调用阿里云百炼API进行语音识别"""
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

        try:
            import dashscope
            from dashscope.audio.asr import Recognition
        except Exception as e:
            raise ValueError(f"dashscope 未安装或不可用: {e}")

        wav_path = ""
        try:
            pcm_bytes = self._decode_base64_to_bytes(audio_base64)
            wav_path = self._pcm_bytes_to_wav_file(pcm_bytes, sample_rate=16000)

            dashscope.api_key = self.api_key
            
            # Use Instance Method for better compatibility
            try:
                recog = Recognition(
                    model=self.model,
                    format='wav',
                    sample_rate=16000,
                    callback=None
                )
                resp = recog.call(file=wav_path)
            except Exception:
                # Fallback to Class Method
                resp = Recognition.call(model=self.model, file=wav_path)

            return self._extract_text_from_resp(resp)
        finally:
            if wav_path:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

# 初始化ASR服务
asr_service = ASRService()

class TTSService:
    def __init__(self):
        self.api_key = DASHSCOPE_API_KEY
        
    async def text_to_speech(self, text: str, voice_type: str = "cute") -> str:
        """
        使用阿里云 Qwen-TTS Realtime API 进行文本转语音
        """
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")
        
        try:
            # 音色映射（阿里云Qwen-TTS支持的音色）
            voice_map = {
                "cute": "Cherry",      # 活泼灵动的女声
                "lively": "Serena",    # 优雅知性的女声
                "gentle": "Chelsie",   # 柔和亲切的女声
                "calm": "Ethan",       # 沉稳磁性的男声
                "fast": "Cherry",      # 快速播报使用Cherry
                "slow": "Chelsie"      # 慢速清晰使用Chelsie
            }
            voice = voice_map.get(voice_type, "Cherry")
            
            logger.info(f"TTS请求: {text}, 音色: {voice}")
            
            # 导入TTS客户端
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from tts_realtime_client import TTSRealtimeClient, SessionMode
            
            # 创建音频数据缓冲区
            audio_chunks = []
            
            def audio_callback(audio_data):
                audio_chunks.append(audio_data)
            
            # 创建TTS客户端
            client = TTSRealtimeClient(
                base_url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen-tts-realtime",
                api_key=self.api_key,
                voice=voice,
                mode=SessionMode.SERVER_COMMIT,
                audio_callback=audio_callback
            )
            
            # 连接并处理文本
            await client.connect()
            
            # 启动消息处理任务
            import asyncio
            message_task = asyncio.create_task(client.handle_messages())
            
            # 发送文本并提交
            await client.append_text(text)
            await client.commit_text_buffer()
            
            # 等待响应开始（最多5秒）
            import time
            start_time = time.time()
            while not client._is_responding and time.time() - start_time < 5:
                await asyncio.sleep(0.1)
            
            if not client._is_responding:
                raise ValueError("TTS响应超时，未收到响应开始信号")
            
            # 等待响应完成（最多15秒）
            start_time = time.time()
            while client._is_responding and time.time() - start_time < 15:
                await asyncio.sleep(0.1)
            
            # 额外等待2秒确保所有音频数据接收完成
            await asyncio.sleep(2)
            
            # 结束会话
            await client.finish_session()
            
            # 等待消息处理任务自然完成（最多5秒）
            try:
                await asyncio.wait_for(message_task, timeout=5)
            except asyncio.TimeoutError:
                # 如果超时，再取消任务
                message_task.cancel()
                try:
                    await message_task
                except asyncio.CancelledError:
                    pass
            
            # 关闭连接
            await client.close()
            
            # 合并音频数据并转换为base64
            if audio_chunks:
                combined_audio = b''.join(audio_chunks)
                audio_base64 = base64.b64encode(combined_audio).decode('utf-8')
                logger.info(f"TTS合成成功，音频长度: {len(combined_audio)} bytes")
                return audio_base64
            else:
                raise ValueError("未收到音频数据")
                
        except Exception as e:
            logger.error(f"TTS合成失败: {e}")
            raise

# 初始化TTS服务
tts_service = TTSService()

@app.get("/")
async def root():
    return {"message": "Oh-My-Live2D ASR/TTS Service 运行正常", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ASR/TTS"}

@app.post("/api/asr/recognize", response_model=ASRResponse)
async def recognize_audio(request: ASRRequest):
    """HTTP API端点：识别音频文件"""
    try:
        text = asr_service.recognize(request.audio_base64)
        return ASRResponse(text=text, success=True)
    except Exception as e:
        logger.error(f"ASR识别失败: {e}")
        return ASRResponse(text="", success=False, error=str(e))

@app.post("/api/tts/speak", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """HTTP API端点：文本转语音"""
    try:
        audio_base64 = await tts_service.text_to_speech(request.text, request.voice_type)
        return TTSResponse(success=True, audio_base64=audio_base64)
    except Exception as e:
        logger.error(f"TTS合成失败: {e}")
        return TTSResponse(success=False, error=str(e))


# 用于语音识别的全局变量
current_audio_data = None
current_text_result = ""


class VoiceStartReq(BaseModel):
    audio_data: str
    audio_format: str = "pcm16le_16k_mono"


@app.post("/api/voice/start")
async def voice_start(req: VoiceStartReq):
    """语音识别开始接口"""
    global current_audio_data, current_text_result
    try:
        # 保存音频数据
        current_audio_data = req.audio_data
        current_text_result = ""
        
        # 调用ASR服务进行语音识别
        text_result = asr_service.recognize(req.audio_data)
        current_text_result = text_result
        
        # 立即返回识别结果，避免前端轮询超时
        return {
            "success": True, 
            "message": "语音识别完成", 
            "text": text_result,
            "has_result": bool(text_result.strip())
        }
    except Exception as e:
        logger.error(f"语音识别开始失败: {e}")
        current_text_result = ""  # 确保出错时清空结果
        return {"success": False, "message": str(e), "text": "", "has_result": False}


@app.get("/api/voice/text")
async def voice_text():
    """获取语音识别文本结果"""
    global current_text_result
    try:
        return {"success": True, "text": current_text_result}
    except Exception as e:
        logger.error(f"获取语音识别文本失败: {e}")
        return {"success": False, "message": str(e)}


@app.post("/api/voice/clear")
async def voice_clear():
    """清除语音识别结果"""
    global current_audio_data, current_text_result
    try:
        current_audio_data = None
        current_text_result = ""
        return {"success": True, "message": "语音识别结果已清除"}
    except Exception as e:
        logger.error(f"清除语音识别结果失败: {e}")
        return {"success": False, "message": str(e)}

@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """WebSocket端点：实时语音识别"""
    await websocket.accept()
    try:
        while True:
            # 接收WebSocket消息
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "audio_data":
                audio_base64 = data.get("audio_data", "")
                if audio_base64:
                    try:
                        text = asr_service.recognize(audio_base64)
                        response = {
                            "type": "asr_result",
                            "text": text,
                            "success": True
                        }
                        await websocket.send_text(json.dumps(response))
                    except Exception as e:
                        error_response = {
                            "type": "asr_error",
                            "error": str(e),
                            "success": False
                        }
                        await websocket.send_text(json.dumps(error_response))
            else:
                # 未知消息类型
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "未知消息类型",
                    "success": False
                }))
                
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
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    
    # 检查API密钥
    if not DASHSCOPE_API_KEY:
        logger.warning("警告: DASHSCOPE_API_KEY 环境变量未设置，ASR功能将无法使用")
        logger.warning("请在 backend/.env 文件中添加 DASHSCOPE_API_KEY=your_api_key")
    else:
        logger.info("API密钥已配置，ASR服务就绪")
    
    logger.info("启动 Oh-My-Live2D ASR Service...")
    logger.info("服务地址: http://localhost:8002")
    logger.info("API文档: http://localhost:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
