# diva-lab-agent

这是当前 `diva-lab-agent` 仓库的阶段性工程化入口文档。

本仓库目前处于“原型功能堆叠期”，功能已经齐全，但目录、配置与服务边界仍在整理中。第一阶段的目标不是重写项目，而是在不改变现有行为的前提下，把端口、接口契约、环境变量和 smoke test 固定下来。

## Current Layout

- `oh-my-live2d-main (2)/oh-my-live2d-main/`
  - 当前主项目目录
  - 包含前端页面、静态资源、Live2D 集成与 `backend/`
- `natori_pro_zh/`
  - 当前接入的 Live2D 模型资源
- `docs/`
  - 第一阶段新增的工程文档
- `scripts/`
  - 第一阶段新增的 smoke test 脚本

## Runtime Services

| 服务 | 默认端口 | 当前入口 |
| --- | --- | --- |
| 页面静态服务 | `8000` | `oh-my-live2d-main (2)/oh-my-live2d-main/mao_demo_server.py` |
| 网关服务 | `8002` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/main.py` |
| 聊天服务 | `8003` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/qwen_chat_server.py` |
| TTS 服务 | `8004` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/tts_ws_server.py` |
| 评分服务 | `8005` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/asr_server.py` |
| 新版 ASR 服务 | `8006` | `oh-my-live2d-main (2)/oh-my-live2d-main/backend/asr_new.py` |

更完整的端口说明见 `docs/ports.md`。

## Install

建议使用本地 Python 3.10+ 环境。

```bash
cd "c:\Project\Mao-zhishi\oh-my-live2d-main (2)\oh-my-live2d-main\backend"
python -m pip install -r requirements.txt
```

## Configure

复制仓库根目录的环境变量样例文件：

```bash
cd "c:\Project\Mao-zhishi"
copy .env.example .env
```

至少需要关注这些变量：

- `WEB_PORT`
- `CHAT_PORT`
- `TTS_PORT`
- `SCORING_PORT`
- `ASR_PORT`
- `DASHSCOPE_API_KEY`
- `BAILIAN_APP_ID`
- `SCORING_WEIGHTS_DIR`
- `ASSET_ROOT`
- `CORS_ALLOW_ORIGINS`

## Start

### 启动页面服务

```bash
cd "c:\Project\Mao-zhishi\oh-my-live2d-main (2)\oh-my-live2d-main"
python mao_demo_server.py
```

默认页面：

- `http://localhost:8000/mao_demo.html`
- `http://localhost:8000/chat_interface.html`

### 启动后端服务

分别进入 `backend/` 后启动：

```bash
cd "c:\Project\Mao-zhishi\oh-my-live2d-main (2)\oh-my-live2d-main\backend"
python main.py
python qwen_chat_server.py
python tts_ws_server.py
python asr_server.py
python asr_new.py
```

Windows 一键启动脚本：

```bash
cd "c:\Project\Mao-zhishi\oh-my-live2d-main (2)\oh-my-live2d-main\backend"
start_all_servers.bat
```

## Smoke Test

第一阶段新增的最小冒烟测试：

```bash
cd "c:\Project\Mao-zhishi"
python scripts/smoke_test.py
```

默认检查：

- `8000` 页面可访问
- `8003` `/health`
- `8004` `/ws/tts` WebSocket 握手
- `8005` `/health`
- `8006` `/health`

可用环境变量覆盖：

- `SMOKE_TEST_HOST`
- `WEB_PORT`
- `CHAT_PORT`
- `TTS_PORT`
- `SCORING_PORT`
- `ASR_PORT`

## Phase 1 Outputs

- `docs/ports.md`
- `docs/api-contract.md`
- `.env.example`
- `scripts/smoke_test.py`

## Refactor Direction

后续渐进式重构按这些原则推进：

- 不删除现有功能
- 不一次性重写项目
- 每次移动文件都保留旧入口兼容
- 每个阶段都保持可启动与可 smoke test
- 先统一配置和接口契约，再拆服务和前端模块
