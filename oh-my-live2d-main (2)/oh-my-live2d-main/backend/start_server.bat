@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo 正在启动 Oh-My-Live2D ASR 服务...
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

REM 检查环境变�?if not exist .env (
    echo 警告: 未找�?.env 文件
    echo 请复�?.env.example �?.env 并配�?DASHSCOPE_API_KEY
    echo.
)

echo 正在启动ASR服务...
echo 服务地址: http://localhost:8002
echo API文档: http://localhost:8002/docs
echo.

python main.py

if %errorlevel% neq 0 (
    echo 服务启动失败，请检查错误信�?    pause
)
