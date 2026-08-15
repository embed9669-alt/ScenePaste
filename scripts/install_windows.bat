@echo off
chcp 65001 >nul
python -m pip install --upgrade pip
python -m pip install -e .
if errorlevel 1 (
  echo 安装失败，请检查 Python 环境。
) else (
  echo ScenePaste 安装完成。运行 scripts\launch_windows.bat 或 scenepaste gui 启动。
)
pause
