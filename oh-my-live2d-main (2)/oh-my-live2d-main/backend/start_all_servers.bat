@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ========================================
echo 启动 Oh-My-Live2D 所有服务
echo ========================================
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Python未安装或未添加到PATH
    echo 请先安装Python 3.8+并确保python命令可用
    pause
    exit /b 1
)

echo 检查依赖包...
python -c "import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo 检测到缺少依赖包，正在安装...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 依赖包安装失败，请检查网络连接和Python环境
        pause
        exit /b 1
    )
)
echo 依赖包检查完成
echo.

REM 检查环境变量
if not exist .env (
    echo 警告: 未找到 .env 文件
    echo 请复制 .env.example 为 .env 并配置 DASHSCOPE_API_KEY
    echo.
)

echo ========================================
echo 启动服务列表:
echo 0. 前端页面服务 (端口8000) - mao_demo/chat_interface 页面
echo 0b. Live2D资源服务 (端口8010) - 模型/JS静态资源
echo 0c. chat_interface 静态服务 (端口8001) - 仅提供 chat_interface.html
echo 1. ASR服务 (端口8002) - 语音识别 (Legacy)
echo 2. LLM聊天服务 (端口8003) - 对话生成
echo 3. TTS服务 (端口8004) - 文本转语音
echo 4. CAM-S评分服务 (端口8005) - 音频评分
echo 5. 新版ASR服务 (端口8006) - 语音识别
echo ========================================
echo.

echo 正在启动 前端页面服务 (端口8000)...
start "WEB-Page-8000" cmd /k "cd /d .. & set WEB_PORT=8000 & set WEB_MODE=page & python mao_demo_server.py"
timeout /t 1 /nobreak >nul

echo 正在启动 Live2D资源服务 (端口8010)...
start "WEB-Assets-8010" cmd /k "cd /d .. & set WEB_PORT=8010 & set WEB_MODE=assets & python mao_demo_server.py"
timeout /t 1 /nobreak >nul

echo 正在启动 chat_interface 静态服务 (端口8001)...
start "WEB-Chat-8001" cmd /k "cd /d .. & python -m http.server 8001 --bind 127.0.0.1"
timeout /t 1 /nobreak >nul

echo 正在启动 ASR 服务 (端口8002)...
start "ASR-Server-8002" cmd /k "python main.py"
timeout /t 2 /nobreak >nul

echo 正在启动 LLM 聊天服务 (端口8003)...
start "LLM-Chat-Server-8003" cmd /k "python qwen_chat_server.py"
timeout /t 2 /nobreak >nul

echo 正在启动 TTS 服务 (端口8004)...
start "TTS-Server-8004" cmd /k "python tts_ws_server.py"
timeout /t 2 /nobreak >nul

echo 正在启动 CAM-S 评分服务 (端口8005)...
start "CAM-S-Server-8005" cmd /k "python asr_server.py"
timeout /t 2 /nobreak >nul

echo 正在启动 新版ASR 服务 (端口8006)...
start "ASR-New-8006" cmd /k "python asr_new.py"
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo 所有服务启动完成！
echo ========================================
echo 服务地址:
echo - Demo主页: http://localhost:8000/mao_demo.html?live2dPort=8010
echo - 聊天界面(8000): http://localhost:8000/chat_interface.html?live2dPort=8010
echo - 聊天界面(8001): http://localhost:8001/chat_interface.html?live2dPort=8010
echo - Live2D资源(8010): http://localhost:8010/
echo - ASR服务(Legacy): http://localhost:8002
echo - LLM聊天: http://localhost:8003
echo - TTS服务: ws://localhost:8004/ws/tts
echo - CAM-S评分: http://localhost:8005
echo - 新版ASR: http://localhost:8006
echo.
echo API文档:
echo - ASR(Legacy): http://localhost:8002/docs
echo - LLM: http://localhost:8003/docs
echo - 新版ASR: http://localhost:8006/docs
echo ========================================
echo.
echo 注意: 请保持这些命令行窗口打开
echo 按任意键关闭此窗口...
pause >nul
