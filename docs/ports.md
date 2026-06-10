# Ports Inventory

本文件记录 `diva-lab-agent` 当前第一阶段中实际使用到的端口、用途、代码入口与后续重构注意事项。

## Scope

本次按要求优先搜索并整理了 `8000`、`8001`、`8002`、`8003`、`8004`、`8005`、`8006`。
另外补充了项目真实运行时会配合使用的 `8010`，因为它承担 Live2D 静态资源服务。

## Port Map

| 端口 | 当前用途 | 主要入口文件 | 当前读取方式 | 说明 |
| --- | --- | --- | --- | --- |
| `8000` | 前端页面静态服务 | `oh-my-live2d-main (2)/oh-my-live2d-main/mao_demo_server.py` | `WEB_PORT`，默认 `8000` | 提供 `mao_demo.html` 与 `chat_interface.html` 页面 |
| `8001` | 备用聊天页静态服务 | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/start_all_servers.bat` | 仅启动脚本里硬编码 | 仍由 `mao_demo_server.py` 托管，但在 bat 中以 `WEB_MODE=page` 单独启动 |
| `8002` | 网关服务（Legacy ASR/TTS proxy） | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/main.py` | 硬编码 | 对下游 `8004`、`8006` 做 HTTP / WebSocket 转发 |
| `8003` | 聊天服务 | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/qwen_chat_server.py` | 硬编码 | 提供 `/health`、`/api/chat`、`/api/chat/stream` |
| `8004` | TTS 服务 | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/tts_ws_server.py` | 硬编码 | 提供 HTTP TTS 与 `/ws/tts` WebSocket |
| `8005` | 音频评分服务 | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/asr_server.py` | `SCORING_PORT`，默认 `8005` | 当前概念上是 scoring，但文件名仍叫 `asr_server.py` |
| `8006` | 新版 ASR 服务 | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/asr_new.py` | `ASR_PORT` 变量存在，但当前值仍写死为 `8006` | 提供轮询式 ASR、一次性识别与 `/ws/asr` |

## Related Port

| 端口 | 当前用途 | 主要入口文件 | 说明 |
| --- | --- | --- | --- |
| `8010` | Live2D 资源静态服务 | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/start_all_servers.bat` | 提供 `/packages/` 与 `/natori_pro_zh/` 等模型资源，前端页面通常通过 `?live2dPort=8010` 访问 |

## References

### `8000` / `8001`

- `oh-my-live2d-main (2)/oh-my-live2d-main/mao_demo_server.py`
  - `WEB_PORT` 默认值为 `8000`
  - `WEB_MODE=page` 时仅放行页面和模型资源
- `oh-my-live2d-main (2)/oh-my-live2d-main/backend/start_all_servers.bat`
  - 以不同 `WEB_PORT`、`WEB_MODE` 组合启动 `8000`、`8001`、`8010`

### `8002`

- `oh-my-live2d-main (2)/oh-my-live2d-main/backend/main.py`
  - `uvicorn.run(... port=8002)`
  - 对下游默认读取 `ASR_BASE_URL=http://127.0.0.1:8006`
  - 对下游默认读取 `TTS_BASE_URL=http://127.0.0.1:8004`

### `8003`

- `oh-my-live2d-main (2)/oh-my-live2d-main/backend/qwen_chat_server.py`
  - `uvicorn.run("qwen_chat_server:app", host="0.0.0.0", port=8003, reload=False)`

### `8004`

- `oh-my-live2d-main (2)/oh-my-live2d-main/backend/tts_ws_server.py`
  - `uvicorn.run(app, host="0.0.0.0", port=8004)`

### `8005`

- `oh-my-live2d-main (2)/oh-my-live2d-main/backend/asr_server.py`
  - `SCORING_PORT = int(os.getenv("SCORING_PORT") ... or "8005")`

### `8006`

- `oh-my-live2d-main (2)/oh-my-live2d-main/backend/asr_new.py`
  - `ASR_PORT = 8006`
  - 后续第二阶段应改为从统一 settings 读取

### 前端引用现状

- `oh-my-live2d-main (2)/oh-my-live2d-main/chat_interface.html`
  - `8003` 用于聊天 SSE
  - `8004` 用于 TTS WebSocket
  - `8005` 用于音频评分
  - `8006` 用于语音识别
  - `TTS_WS_URL` 当前通过页面部署参数中的 `apiHost` / `ttsPort` 生成

## Refactor Notes

- 第一阶段不修改行为，只把现状记录清楚。
- 第二阶段需要把所有端口、CORS、密钥与模型路径统一转移到 settings 模块。
- 第三阶段开始迁移服务目录时，必须保留这些旧入口文件的兼容 wrapper。
