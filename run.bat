@echo off
setlocal

:: 设置当前目录为工作目录
cd /d "%~dp0"

:: 设置环境变量
set "PATH=%~dp0env;%~dp0env\Scripts;%~dp0;%PATH%"
set "PYTHONPATH=%~dp0"

:: 检查环境是否配置正确
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python environment not found! Please ensure the 'env' directory exists.
    pause
    exit /b 1
)

where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo ffmpeg not found! Please ensure 'ffmpeg.exe' is in the current directory.
    pause
    exit /b 1
)

:: 运行应用
echo Starting application...
python app.py
pause

