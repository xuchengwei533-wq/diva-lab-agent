# API Contract

本文档记录当前项目的接口契约现状，目标是先把“现在实际怎么通信”固定下来，为后续统一配置、拆分服务和协议清理提供基线。

## Services

| 服务 | 默认端口 | 当前入口 |
| --- | --- | --- |
| 页面静态服务 | `8000` | `oh-my-live2d-main (2)/oh-my-live2d-main/mao_demo_server.py` |
| 网关服务 | `8002` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/main.py` |
| 聊天服务 | `8003` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/qwen_chat_server.py` |
| TTS 服务 | `8004` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/tts_ws_server.py` |
| 评分服务 | `8005` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/asr_server.py` |
| 新版 ASR 服务 | `8006` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/asr_new.py` |

## Frontend Endpoints In Use

当前 `chat_interface.html` 直接使用这些地址：

- `http://${HOST}:8003/api/chat/stream`
- `ws://${HOST}:8004/ws/tts?voice=Cherry&model=qwen3-tts-flash&language_type=Chinese`
- `http://${HOST}:8005/api/audio/score`
- `http://${HOST}:8006/api/voice/start`

这意味着：

- TTS WebSocket 当前通过页面部署参数中的 `apiHost` / `ttsPort` 生成
- 页面没有统一 endpoint 配置模块
- 第一阶段只记录现状，不改前端行为

## Chat Service

### `GET /health`

- 服务：`8003`
- 响应示例：

```json
{
  "ok": true
}
```

### `POST /api/chat/stream`

- 服务：`8003`
- 内容类型：`application/json`
- 前端当前请求体：

```json
{
  "model": "qwen-plus",
  "messages": [
    { "role": "user", "content": "你好" }
  ]
}
```

- 实际兼容字段：

```json
{
  "model": "ignored",
  "messages": [],
  "prompt": "你好",
  "session_id": null,
  "biz_params": null,
  "rag_options": null,
  "memory_id": null,
  "has_thoughts": null,
  "extra_body": null
}
```

- 响应类型：`text/event-stream`
- 当前事件格式：

```text
data: {"type":"delta","delta":"..."}

data: {"type":"error","error":"...","code":500}

data: {"type":"finish","finish_reason":"stop"}

data: {"type":"done"}
```

- 说明：
  - 当前前端主要消费 `delta`
  - `system` 角色会在服务端被丢弃
  - 若上游 Bailian 失败，会返回 `type=error`

### `POST /api/chat`

- 服务：`8003`
- 用途：非流式调试
- 请求体：与 `/api/chat/stream` 相同
- 成功响应示例：

```json
{
  "text": "你好，我是肖斯塔科维奇。",
  "session_id": null
}
```

- 失败响应示例：

```json
{
  "error": "upstream error message"
}
```

## TTS Service

### `POST /api/tts/speak`

- 服务：`8004`
- 内容类型：`application/json`
- 请求体：

```json
{
  "text": "你好",
  "voice_type": "cute"
}
```

- 成功响应示例：

```json
{
  "success": true,
  "audio_base64": "BASE64_AUDIO"
}
```

- 失败响应：
  - `400`: 空文本
  - `500`: 缺少 `DASHSCOPE_API_KEY` 或 TTS 未返回音频

### `WS /ws/tts`

- 服务：`8004`
- 当前 query 参数：
  - `voice`
  - `model`
  - `language_type`

- 当前客户端发送消息：

```json
{ "type": "input_text_buffer.append", "text": "你好" }
```

```json
{ "type": "input_text_buffer.commit" }
```

```json
{ "type": "session.finish" }
```

- 当前服务端返回消息：

```json
{
  "type": "session.ready",
  "model": "qwen3-tts-flash",
  "voice": "Cherry",
  "language_type": "Chinese",
  "mode": "streaming_pcm"
}
```

```json
{
  "type": "response.audio.delta",
  "delta": "BASE64_PCM_CHUNK"
}
```

```json
{
  "type": "response.segment.done"
}
```

```json
{
  "type": "response.done"
}
```

```json
{
  "type": "error",
  "error": "TTS failed: ..."
}
```

- 说明：
  - 当前协议已经以 `response.audio.delta` 为主
  - `session.ready` 还没有返回 `sample_rate`、`format`，这是第四阶段需要补的契约项

## Scoring Service

### `GET /health`

- 服务：`8005`
- 响应示例：

```json
{
  "status": "healthy",
  "service": "AudioScoring"
}
```

### `POST /api/audio/score`

- 服务：`8005`
- 内容类型：`application/json`
- 当前前端请求体：

```json
{
  "audio_data": "BASE64_AUDIO",
  "voice_type": "女高音 Soprano",
  "audio_format": "wav"
}
```

- 成功响应示例：

```json
{
  "success": true,
  "message": "音频评分完成",
  "overall": 3.7,
  "scores": [
    { "technique": "vibrato", "score": 4 },
    { "technique": "throat", "score": 3 }
  ],
  "voice_type": "女高音 Soprano",
  "original_sr": 44100,
  "target_sr": 44100,
  "duration_sec": 5.12,
  "rms": 0.024
}
```

- 约束：
  - 返回 `scores` 为 10 维技巧评分
  - `voice_type` 支持中文声部名与别名映射

- 常见错误：
  - `400`: base64 非法、音频过短、不支持的声部类型
  - `500`: 缺库、缺模型权重、推理失败

## ASR Service

### `GET /health`

- 服务：`8006`
- 响应示例：

```json
{
  "status": "healthy",
  "service": "ASR-New"
}
```

### `POST /api/voice/clear`

- 服务：`8006`
- 请求体：无
- 响应示例：

```json
{
  "success": true,
  "message": "语音识别结果已清除",
  "text": "",
  "has_result": false
}
```

### `GET /api/voice/text`

- 服务：`8006`
- 响应示例：

```json
{
  "success": true,
  "message": "ok",
  "text": "你好",
  "has_result": true
}
```

- 失败态也可能返回：

```json
{
  "success": false,
  "message": "error message",
  "text": "",
  "has_result": false
}
```

### `POST /api/voice/start`

- 服务：`8006`
- 内容类型：`application/json`
- 当前前端请求体：

```json
{
  "audio_data": "BASE64_PCM16",
  "audio_format": "pcm16le_16k_mono"
}
```

- 实际兼容字段：

```json
{
  "audio_data": "BASE64_PCM16",
  "audio_base64": null,
  "audio_format": "pcm16le_16k_mono"
}
```

- 成功响应示例：

```json
{
  "success": true,
  "message": "语音识别完成",
  "text": "你好",
  "has_result": true,
  "debug": {
    "audio": {
      "duration_sec": 1.2,
      "rms": 156.0,
      "peak": 4021
    },
    "asr": {}
  }
}
```

- 特殊成功态：
  - 音频过短
  - 音频能量过低
  - 未识别到文本

- 失败响应：
  - `400`: 缺少音频、音频格式不支持、base64 非法
  - `500`: ASR SDK 或上游失败

### `POST /api/asr/recognize`

- 服务：`8006`
- 请求体：

```json
{
  "audio_base64": "BASE64_PCM16",
  "model": "qwen3-asr-flash"
}
```

- 成功响应示例：

```json
{
  "success": true,
  "text": "你好",
  "debug": {
    "audio": {
      "duration_sec": 1.2
    }
  }
}
```

### `WS /ws/asr`

- 服务：`8006`
- 当前客户端发送：

```json
{
  "type": "audio_data",
  "audio_data": "BASE64_PCM16",
  "model": "qwen3-asr-flash"
}
```

- 成功返回：

```json
{
  "type": "asr_result",
  "text": "你好",
  "success": true
}
```

- 失败返回：

```json
{
  "type": "asr_error",
  "error": "Missing audio_data",
  "success": false
}
```

或

```json
{
  "type": "error",
  "message": "未知消息类型",
  "success": false
}
```

## Gateway Service

### `GET /`

- 服务：`8002`
- 响应示例：

```json
{
  "service": "gateway",
  "asr_base_url": "http://127.0.0.1:8006",
  "tts_base_url": "http://127.0.0.1:8004"
}
```

### `GET /health`

- 服务：`8002`
- 响应示例：

```json
{
  "status": "ok",
  "service": "gateway",
  "asr_base_url": "http://127.0.0.1:8006",
  "tts_base_url": "http://127.0.0.1:8004"
}
```

### Proxy Endpoints

- `POST /api/asr/recognize` -> 代理到 `8006`
- `POST /api/tts/speak` -> 代理到 `8004`
- `POST /api/voice/start` -> 代理到 `8006`
- `GET /api/voice/text` -> 代理到 `8006`
- `POST /api/voice/clear` -> 代理到 `8006`
- `WS /ws/asr` -> 代理到 `8006`

## Current Contract Risks

- `TTS_WS_URL` 当前通过页面部署参数中的 `apiHost` / `ttsPort` 生成
- `session.ready` 缺少 `sample_rate`、`format`
- `qwen_chat_server.py` 仍硬编码 `BAILIAN_APP_ID`
- 多个服务仍使用 `allow_origins=["*"]`
- 页面层和服务层都散落着端口定义，尚未统一到 settings
