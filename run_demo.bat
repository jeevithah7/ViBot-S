@echo off
title ViBot-S Demo Launcher
color 0B
echo.
echo  ====================================================
echo   ViBot-S -- Self-Balancing Navigation Robot DEMO
echo  ====================================================
echo.
echo  Starting dashboard...  Browser will open automatically.
echo  Press Ctrl+C in this window to stop the server.
echo.

"%~dp0..\.venv\Scripts\python.exe" "%~dp0demo\demo_mode.py" %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Something failed. See above for details.
    pause
)
