@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================
echo   ESET BOT - locale launch
echo ================================
echo.

if not exist ".env" (
    echo [OSHIBKA] .env ne naiden.
    echo Skopiruj .env.example v .env i vpishi TG_TOKEN i GROQ_API_KEY.
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo [OSHIBKA] Python ne naiden v PATH.
    echo Postav Python 3.11 s https://python.org i otmet
    echo "Add python.exe to PATH" pri ustanovke.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [SETUP] Sozdayu virtualnoe okruzhenie...
    python -m venv venv
    if errorlevel 1 (
        echo [OSHIBKA] Ne udalos sozdat venv.
        pause
        exit /b 1
    )
)

echo [SETUP] Ustanavlivayu zavisimosti...
venv\Scripts\python.exe -m pip install --upgrade pip >nul
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [OSHIBKA] pip install ne srabotal. Proveryaem internet.
    pause
    exit /b 1
)

echo.
echo ================================
echo   Zapusk bota. Ctrl+C - ostanovka.
echo ================================
echo.

:run
venv\Scripts\python.exe rex_voice_v4_bot.py
if errorlevel 1 (
    echo.
    echo [SBOJ] Bot zavershilsya s oshibkoj - smotri log vyshe.
    echo Nazhmi lyubuyu klavishu dlya perezapuska, ili zakroj okno.
    pause >nul
    goto run
)

pause
