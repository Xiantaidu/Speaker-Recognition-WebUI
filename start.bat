@echo off
setlocal

:: 设置当前目录为工作目录
cd /d "%~dp0"

:: 设置环境变量
:: 将 env\Scripts 和 env\Lib\site-packages 添加到 PATH 和 PYTHONPATH
set "PATH=%~dp0env\Scripts;%~dp0;%PATH%"
set "PYTHONPATH=%~dp0"

:: 检查环境是否配置正确
if not exist "%~dp0env\Scripts\python.exe" (
    echo Python environment not found! Please ensure the 'env' directory exists.
    pause
    exit /b 1
)

if not exist "%~dp0ffmpeg.exe" (
    echo ffmpeg not found! Please ensure 'ffmpeg.exe' is in the current directory.
    pause
    exit /b 1
)

:: 运行应用
echo Starting application...
"%~dp0env\Scripts\python.exe" app.py
pause
