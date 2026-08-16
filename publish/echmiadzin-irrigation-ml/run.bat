@echo off
cd /d %~dp0

set "PROJECT_SITE_PACKAGES=%~dp0.venv\Lib\site-packages"
set "PROJECT_VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PYTHON_EXE="
set "PYTHON_ARGS="

if exist "%PROJECT_VENV_PY%" (
  "%PROJECT_VENV_PY%" -V >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=%PROJECT_VENV_PY%"
  )
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
  if not defined PYTHON_EXE set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)

if not defined PYTHON_EXE if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

if not defined PYTHON_EXE (
  for /f "delims=" %%P in ('where py 2^>nul') do (
    if not defined PYTHON_EXE (
      set "PYTHON_EXE=%%P"
      set "PYTHON_ARGS=-3.12"
    )
  )
)

if not defined PYTHON_EXE (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
  )
)

if not defined PYTHON_EXE (
  echo Python was not found on this PC.
  echo Install Python 3.12, then run this file again.
  pause
  exit /b 1
)

if not exist "%PROJECT_SITE_PACKAGES%" (
  echo Project site-packages not found at %PROJECT_SITE_PACKAGES%
  echo Recreate the environment or install dependencies first.
  pause
  exit /b 1
)

set "PYTHONPATH=%PROJECT_SITE_PACKAGES%;%PYTHONPATH%"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_SERVER_HEADLESS=true"

"%PYTHON_EXE%" %PYTHON_ARGS% -m streamlit run app.py --server.headless true --browser.gatherUsageStats false

pause
