@echo off
chcp 65001 >nul
python -m scenepaste gui %*
if errorlevel 1 pause
