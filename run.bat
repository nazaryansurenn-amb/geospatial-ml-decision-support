@echo off
rem Starts the Echmiadzin Irrigation Monitor on http://localhost:8502.
rem First run creates a Python 3.12 environment in .venv and installs requirements.txt.
setlocal
cd /d "%~dp0"
set "PORT=8502"
set "VENV=%~dp0.venv"
set "VPY=%VENV%\Scripts\python.exe"

netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if %errorlevel%==0 (
    echo The app is already running on port %PORT%. Opening it in the browser.
    start "" "http://localhost:%PORT%"
    exit /b 0
)

if not exist "%VENV%\installed.ok" call :setup
if not exist "%VENV%\installed.ok" (
    echo.
    echo Setup failed. See the messages above.
    pause
    exit /b 1
)

rem Open the browser once the app answers its health check.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 240; $i++) { try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:%PORT%/_stcore/health).Content -eq 'ok') { Start-Process http://localhost:%PORT%; break } } catch {}; Start-Sleep -Milliseconds 500 }"

echo Starting the Echmiadzin Irrigation Monitor on http://localhost:%PORT%
echo Close this window to stop the app.
"%VPY%" -m streamlit run app.py --server.port %PORT%
if errorlevel 1 pause
exit /b

:setup
echo First run: creating a Python environment in .venv and installing libraries.
echo This takes a few minutes and happens only once.
set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY (
    echo Python 3.12 was not found. Install it from https://www.python.org/downloads/ and run this file again.
    exit /b 1
)
if not exist "%VPY%" %PY% -m venv "%VENV%" || exit /b 1
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt || exit /b 1
echo ok> "%VENV%\installed.ok"
exit /b 0
