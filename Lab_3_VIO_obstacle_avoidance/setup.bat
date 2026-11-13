@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    echo Python launcher "py" was not found.
    echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo and check "Add python.exe to PATH" in the installer.
    exit /b 1
)

py -3 -m venv .venv
if errorlevel 1 exit /b 1

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Ready. Stay on a network with internet until this finishes.
echo Then activate the venv and start a script from this folder:
echo   .venv\Scripts\activate.bat
echo   python opencv-viewer.py
echo   python follow_me.py MOTIVE_SERVER_IP --client THIS_PC_IP --body-id 4 --address 0xE7E7E7E7E7
endlocal
