@echo off
echo =====================================================
echo KOIOS - Advanced Mathematical Toolset
echo Windows Deployment Script
echo =====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or later from https://python.org
    pause
    exit /b 1
)

echo Python detected successfully.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

echo pip detected successfully.

REM Install required packages
echo Installing required packages...
pip install streamlit matplotlib numpy pandas plotly scipy sympy

if %errorlevel% neq 0 (
    echo ERROR: Failed to install required packages
    pause
    exit /b 1
)

echo.
echo All packages installed successfully!
echo.
echo Starting KOIOS application...
echo.

REM Start the application
python -m streamlit run app.py --server.port 5000 --server.address 0.0.0.0

pause