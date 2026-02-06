@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo 正在启动 Oh-My-Live2D CAM-S 音频评分服务...
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Python未安装或未添加到PATH
    echo 请先安装Python 3.8+并确保python命令可用
    pause
    exit /b 1
)

REM 检查依赖包
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

echo 依赖包检查完�?echo.

echo 正在启动CAM-S音频评分服务...
echo 服务地址: http://localhost:8005
echo API文档: http://localhost:8005/docs
echo.

python asr_server.py

if %errorlevel% neq 0 (
    echo 服务启动失败，请检查错误信�?    pause
)
