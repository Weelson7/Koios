@echo off
REM ============================================
REM Koios - Windows Installation and Deployment Script
REM ============================================

REM Configuration - Edit this line with your GitHub repository URL
set REPO_URL=https://github.com/Weelson7/Koios.git
set PROJECT_DIR=Koios

echo.
echo ========================================
echo Koios Installation and Deployment
echo ========================================
echo.

REM Check if running from within the project directory
if exist "app.py" (
    echo [INFO] Running from project directory
    goto :skip_clone
)

REM Check if Git is installed
echo [INFO] Checking for Git...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Git is not installed or not in PATH
    echo [INFO] Please install Git from https://git-scm.com/
    echo [INFO] Or manually clone the repository and run this script from the project directory
    pause
    exit /b 1
)

echo [SUCCESS] Git found
echo.

REM Check if project directory exists
if exist "%PROJECT_DIR%" (
    echo [INFO] Project directory already exists
    echo [INFO] Pulling latest changes...
    cd %PROJECT_DIR%
    git pull
    if %errorlevel% neq 0 (
        echo [WARNING] Failed to pull latest changes, continuing with existing code
    )
) else (
    echo [INFO] Cloning repository from GitHub...
    echo [INFO] Repository: %REPO_URL%
    git clone %REPO_URL% %PROJECT_DIR%
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to clone repository
        echo [INFO] Please check the repository URL and your internet connection
        pause
        exit /b 1
    )
    cd %PROJECT_DIR%
)
echo.

:skip_clone
REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
) else (
    echo [INFO] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
if exist "requirements.txt" (
    echo [INFO] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo [INFO] No requirements.txt found, installing core dependencies...
    pip install streamlit numpy scipy sympy matplotlib plotly pandas
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)
echo.

echo [SUCCESS] Installation complete!
echo.
echo ========================================
echo Starting Koios Application
echo ========================================
echo.
echo [INFO] Launching Streamlit application...
echo [INFO] Server running at http://localhost:5000
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Run the application (filter out Streamlit's default browser message)
python -m streamlit run app.py --server.headless=true --server.address=localhost --server.port=5000 --logger.level=error 2>&1 | findstr /V /C:"You can now view your Streamlit app in your browser" /C:"Local URL:" /C:"Network URL:"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start application
    pause
    exit /b 1
)

pause
