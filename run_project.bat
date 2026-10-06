@echo off
echo ===================================================
echo   MENTAL HEALTH CHATBOT LAUNCHER
echo ===================================================
echo.

cd /d "%~dp0"

echo 1. Activating virtual environment...
if exist "mental_env\Scripts\activate.bat" (
    call "mental_env\Scripts\activate.bat"
) else (
    echo [ERROR] Virtual environment not found!
    echo Creating new environment...
    python -m venv mental_env
    call "mental_env\Scripts\activate.bat"
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo 2. Checking dependencies...
python -c "import tensorflow" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] TensorFlow might be missing or broken.
    echo Attempting to install requirements...
    pip install -r requirements.txt
)

echo.
echo 3. Starting Server...
echo    Open index.html in your browser once the server starts.
echo.
python mental_health_api.py

pause
