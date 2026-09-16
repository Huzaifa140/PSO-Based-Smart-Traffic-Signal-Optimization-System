@echo off
title PSO Smart Traffic Signal Optimizer
color 0A

echo.
echo  ==========================================
echo   PSO Smart Traffic Signal Optimizer
echo   AI Course CCP Project
echo  ==========================================
echo.

:: ── Force working directory to where THIS .bat file is located ─────────
cd /d "%~dp0"

:: ── Debug: show current directory ──────────────────────────────────────
echo  [INFO] Running from: %~dp0
echo.

:: ── Check Python ────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Download from python.org
    pause
    exit /b 1
)
echo  [OK] Python found.

:: ── Check app.py exists ─────────────────────────────────────────────────
if not exist "%~dp0app.py" (
    echo.
    echo  [ERROR] app.py not found in: %~dp0
    echo.
    echo  Make sure RUN_APP.bat is in the SAME folder as app.py
    echo  Your folder should contain:
    echo    - app.py
    echo    - requirements.txt
    echo    - RUN_APP.bat
    echo.
    pause
    exit /b 1
)
echo  [OK] app.py found.

:: ── Check requirements.txt exists ───────────────────────────────────────
if not exist "%~dp0requirements.txt" (
    echo  [WARN] requirements.txt missing, installing manually...
    pip install streamlit numpy matplotlib pandas --quiet
) else (
    echo  [..] Installing packages...
    pip install -r "%~dp0requirements.txt" --quiet --disable-pip-version-check
)
echo  [OK] Packages ready.

echo.
echo  [..] Launching app - browser will open at http://localhost:8501
echo.

:: ── Launch Streamlit with full path to app.py ───────────────────────────
python -m streamlit run "%~dp0app.py"

pause
