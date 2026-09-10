@echo off
setlocal
cd /d "%~dp0"

if defined GAIA_PYTHON (
    set "PYTHON_EXE=%GAIA_PYTHON%"
) else (
    if exist "C:\Users\RAJOS\anaconda3\envs\cardiac_twin\python.exe" (
        set "PYTHON_EXE=C:\Users\RAJOS\anaconda3\envs\cardiac_twin\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
)

"%PYTHON_EXE%" gaia.py %*
if errorlevel 1 (
    echo.
    echo GAIA failed to start with: "%PYTHON_EXE%"
    echo Install dependencies with: python -m pip install -r requirements.txt
    echo Or set GAIA_PYTHON to the Python executable for your GAIA environment.
    echo.
    pause
)
