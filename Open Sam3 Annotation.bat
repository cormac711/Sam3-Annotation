@echo off
REM Double-click launcher: runs the app straight from source (dev mode),
REM no rebuild/installer needed. For the real one-click installer, see
REM installer\build-installer.ps1 -- this .bat is the fast local-iteration path.

cd /d "%~dp0app"

REM Electron's own bundled Node can get confused if this is set in the
REM shell (forces electron.exe to behave as plain node instead of launching
REM the app) -- clear it just in case it's inherited from somewhere.
set ELECTRON_RUN_AS_NODE=

if not exist "node_modules" (
    echo Installing app dependencies -- first run only, this takes a minute...
    call npm install
    if errorlevel 1 (
        echo.
        echo npm install failed. Is Node.js installed?
        pause
        exit /b 1
    )
)

echo Starting Sam3 Annotation...
call npm start

if errorlevel 1 (
    echo.
    echo The app exited with an error -- see the output above.
    pause
)
