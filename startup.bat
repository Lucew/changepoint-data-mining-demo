@echo off
setlocal ENABLEEXTENSIONS
rem Always run from the script's folder
pushd "%~dp0"
chcp 65001 >nul

rem -------- settings --------
set "VENV_DIR=.venv"
set "APP_URL=http://127.0.0.1:8050/"
rem How long to wait for the server to come up before giving up (seconds)
set "READY_TIMEOUT=180"
rem --------------------------

rem Find Python in typical layouts
if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
) else if exist "%VENV_DIR%\python.exe" (
    set "PY_EXE=%VENV_DIR%\python.exe"
)

if not defined PY_EXE (
    call :missing_env | more
    goto :END
)

if not exist "Dash_Mainpage.py" (
    echo [ERROR] "Dash_Mainpage.py" not found in:
    echo   %CD%
    echo Make sure this script sits next to Dash_Mainpage.py.
    goto :END
)

echo Using interpreter: "%PY_EXE%"
echo Starting Dash app...
echo with command: "%PY_EXE%" "Dash_Mainpage.py" --debug False
echo.

"%PY_EXE%" "Dash_Mainpage.py" --debug False
set "EXITCODE=%ERRORLEVEL%"

echo.
echo App exited with code %EXITCODE%.

:END
echo.
pause
popd
endlocal
goto :EOF

:missing_env
@echo [ERROR] Couldn't find a Python interpreter in ".venv".
@echo.
@echo I looked for:
@echo   - .venv\Scripts\python.exe   (pip venv / uv on Windows)
@echo   - .venv\python.exe           (conda env created at a path)
@echo.
@echo Quick create commands (run ONE of these in this folder):
@echo   ^> uv
@echo   uv venv --python 3.11
@echo   uv pip install -r requirements.txt
@echo.
@echo   ^> python -m venv
@echo   python -m venv .venv
@echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
@echo.
@echo   ^> conda (path-based env)
@echo   conda create -y -p .venv python=3.11
@echo   .venv\python.exe -m pip install -r requirements.txt
@echo.
