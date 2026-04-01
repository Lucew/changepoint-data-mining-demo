@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
pushd "%~dp0"

rem -------- settings --------
set "VENV_DIR=.venv"
set "DEFAULT_ARGS=--mode standalone --port 8050"
set "DEFAULT_HOST=127.0.0.1"
set "WAIT_BEFORE_BROWSER=3"
rem --------------------------

set "PY_EXE="
if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
) else if exist "%VENV_DIR%\python.exe" (
    set "PY_EXE=%VENV_DIR%\python.exe"
)

if not defined PY_EXE (
    echo [ERROR] Could not find python in .venv
    goto :END
)

if not exist "Dash_Mainpage.py" (
    echo [ERROR] Dash_Mainpage.py not found
    goto :END
)

rem Use defaults unless args were provided to the .bat
if "%~1"=="" (
    set "ARGS=%DEFAULT_ARGS%"
) else (
    set "ARGS=%*"
)

rem ---- parse ARGS for port/host (robust; no extra windows; no scoping issues) ----
set "PORT="
set "HOST="
call :PARSE_ARGS %ARGS%

if not defined PORT set "PORT=8050"
if not defined HOST set "HOST=%DEFAULT_HOST%"

rem If host is 0.0.0.0 / :: / * (bind-all), open on 127.0.0.1 for the browser
set "BROWSER_HOST=%HOST%"
if /I "%HOST%"=="0.0.0.0" set "BROWSER_HOST=127.0.0.1"
if /I "%HOST%"=="::"      set "BROWSER_HOST=127.0.0.1"
if "%HOST%"=="*"          set "BROWSER_HOST=127.0.0.1"

set "APP_URL=http://%BROWSER_HOST%:%PORT%/"

echo Using interpreter: "%PY_EXE%"
echo Starting: Dash_Mainpage.py %ARGS%
echo Will open browser at %APP_URL% after %WAIT_BEFORE_BROWSER%s
echo (Press Ctrl+C to stop the app.)
echo.

rem --- open browser after a short delay (one-shot helper) ---
start "Dash: Browser opener" cmd /c "title Dash - Browser opener & echo This window waits %WAIT_BEFORE_BROWSER% seconds, then opens: %APP_URL% & echo Press any key to cancel. & timeout /t %WAIT_BEFORE_BROWSER% && start "" "%APP_URL%""

rem --- run the app in foreground so Ctrl+C kills it ---
"%PY_EXE%" Dash_Mainpage.py %ARGS%
set "EXITCODE=%ERRORLEVEL%"

echo.
echo App exited with code %EXITCODE%.

:END
echo.
pause
popd
endlocal
goto :EOF


:: ==============================
:: Parse %* for --port/-p and --host/-h in forms:
::   --port N   --port=N   -p N   -p=N   -pN
::   --host H   --host=H   -h H   -h=H   -hH
:: Sets PORT and HOST if present.
:: ==============================
:PARSE_ARGS
:PARSE_LOOP
if "%~1"=="" goto :EOF
set "arg=%~1"

rem ---- PORT ----
if /I "%arg%"=="--port" (
    set "PORT=%~2"
    shift & shift & goto :PARSE_LOOP
)
if /I "!arg:~0,7!"=="--port=" (
    set "PORT=!arg:~7!"
    shift & goto :PARSE_LOOP
)
if /I "%arg%"=="-p" (
    set "PORT=%~2"
    shift & shift & goto :PARSE_LOOP
)
if /I "!arg:~0,3!"=="-p=" (
    set "PORT=!arg:~3!"
    shift & goto :PARSE_LOOP
)
if /I "!arg:~0,2!"=="-p" (
    rem handles -p9000 (no separator)
    set "PORT=!arg:~2!"
    shift & goto :PARSE_LOOP
)

rem ---- HOST ----
if /I "%arg%"=="--host" (
    set "HOST=%~2"
    shift & shift & goto :PARSE_LOOP
)
if /I "!arg:~0,7!"=="--host=" (
    set "HOST=!arg:~7!"
    shift & goto :PARSE_LOOP
)
if /I "%arg%"=="-h" (
    set "HOST=%~2"
    shift & shift & goto :PARSE_LOOP
)
if /I "!arg:~0,3!"=="-h=" (
    set "HOST=!arg:~3!"
    shift & goto :PARSE_LOOP
)
if /I "!arg:~0,2!"=="-h" (
    rem handles -h127.0.0.1 (no separator)
    set "HOST=!arg:~2!"
    shift & goto :PARSE_LOOP
)

shift
goto :PARSE_LOOP
