# xiaosita222

瘦身版交互数字人。第一版只保留语音交互回复链路：

- 浏览器录音和 VAD 自动断句
- 前端先播放本地开场白音频，遮住后端思考耗时
- 单进程 FastAPI 后端完成 ASR -> 百炼应用回复 -> DashScope TTS
- 使用原项目的肖斯塔科维奇 system prompt
- 完整复制原项目 `TTS-total/storage` 下的开场白 wav
- 使用原项目 `待机.gif` / `说话.gif` 作为轻量数字人形象

## 启动

```powershell
cd C:\Users\diva\Documents\Codex\2026-06-08\xuchengwei533-wq-diva-lab-agent-git\xuchengwei\xiaosita222
.\scripts\start.ps1
```

浏览器打开：

```text
http://127.0.0.1:47231/
```

公网网页要使用 HTTPS，否则浏览器会禁用麦克风。

## 环境变量

本地 `.env` 已从旧项目复制过来，不提交到 GitHub。提交到仓库的是 `.env.example`。

必需：

- `DASHSCOPE_API_KEY`
- `BAILIAN_APP_ID`

默认 `BAILIAN_APP_ID` 保留旧项目值：`4dc0700043fc46679e1568339e580678`。

## 主要接口

- `GET /health`
- `GET /api/opening`
- `POST /api/turn`
- `POST /api/chat`
- `POST /api/tts`
- `GET /proxy-audio?url=...`
