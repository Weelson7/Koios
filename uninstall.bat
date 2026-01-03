@echo off
REM ============================================
REM Koios - Windows Uninstall Script
REM ============================================

echo.
echo ========================================
echo Koios Uninstall
echo ========================================
echo.

REM Check if running from within the project directory
if not exist "app.py" (
    echo [ERROR] This script must be run from the Koios project directory
    pause
    exit /b 1
)

echo [WARNING] This will remove:
echo   - Virtual environment (venv folder)
echo   - Python cache files (__pycache__)
echo   - Temporary files
echo.
echo [INFO] The source code will be preserved
echo.

set /p CONFIRM="Are you sure you want to uninstall? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo [INFO] Uninstall cancelled
    pause
    exit /b 0
)

echo.
echo [INFO] Starting uninstall process...
echo.

REM Remove virtual environment
if exist "venv\" (
    echo [INFO] Removing virtual environment...
    rmdir /s /q venv
    if %errorlevel% equ 0 (
        echo [SUCCESS] Virtual environment removed
    ) else (
        echo [WARNING] Failed to remove virtual environment
    )
) else (
    echo [INFO] No virtual environment found
)
echo.

REM Remove Python cache files
echo [INFO] Removing Python cache files...
for /d /r %%i in (__pycache__) do @if exist "%%i" rmdir /s /q "%%i"
del /s /q *.pyc >nul 2>&1
del /s /q *.pyo >nul 2>&1
echo [SUCCESS] Python cache files removed
echo.

REM Remove .pytest_cache if exists
if exist ".pytest_cache\" (
    echo [INFO] Removing pytest cache...
    rmdir /s /q .pytest_cache
    echo [SUCCESS] Pytest cache removed
    echo.
)

REM Remove .streamlit cache if exists
if exist ".streamlit\" (
    set /p STREAMLIT="Remove Streamlit configuration (includes telemetry settings)? (y/N): "
    if /i "!STREAMLIT!"=="y" (
        rmdir /s /q .streamlit
        echo [SUCCESS] Streamlit configuration removed
        echo.
    ) else (
        echo [INFO] Streamlit configuration preserved (telemetry disabled)
        echo.
    )
)

echo.
echo ========================================
echo Uninstall Complete
echo ========================================
echo.
echo [SUCCESS] Koios has been uninstalled
echo.
echo [INFO] Source code has been preserved
echo [INFO] To completely remove Koios, delete this directory
echo.

set /p REMOVE_ALL="Do you want to remove the entire project directory? (y/N): "
if /i "%REMOVE_ALL%"=="y" (
    echo.
    echo [WARNING] This will delete ALL files including source code!
    set /p FINAL_CONFIRM="Are you absolutely sure? (y/N): "
    if /i "!FINAL_CONFIRM!"=="y" (
        cd ..
        set CURRENT_DIR=%CD%\Koios
        echo [INFO] Removing project directory...
        rmdir /s /q "%CURRENT_DIR%"
        echo [SUCCESS] Project directory removed
        exit /b 0
    )
)

echo.
echo [INFO] You can reinstall anytime by running install_and_run.bat
pause
