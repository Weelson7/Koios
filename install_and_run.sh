#!/bin/bash
# ============================================
# Koios - Linux/Mac Installation and Deployment Script
# ============================================

# Configuration - Edit this line with your GitHub repository URL
REPO_URL="https://github.com/Weelson7/Koios.git"
PROJECT_DIR="Koios"

echo ""
echo "========================================"
echo "Koios Installation and Deployment"
echo "========================================"
echo ""

# Check if running from within the project directory
if [ -f "app.py" ]; then
    echo "[INFO] Running from project directory"
    SKIP_CLONE=true
else
    SKIP_CLONE=false
    
    # Check if Git is installed
    echo "[INFO] Checking for Git..."
    if ! command -v git &> /dev/null; then
        echo "[WARNING] Git is not installed"
        echo "[INFO] Please install Git:"
        if [[ "$(uname -s)" == "Darwin" ]]; then
            echo "  macOS: brew install git or xcode-select --install"
        elif [[ "$(uname -s)" == "Linux" ]]; then
            echo "  Ubuntu/Debian: sudo apt-get install git"
            echo "  Fedora/RHEL: sudo dnf install git"
        fi
        echo "[INFO] Or manually clone the repository and run this script from the project directory"
        exit 1
    fi
    
    echo "[SUCCESS] Git found"
    echo ""
    
    # Check if project directory exists
    if [ -d "$PROJECT_DIR" ]; then
        echo "[INFO] Project directory already exists"
        echo "[INFO] Pulling latest changes..."
        cd "$PROJECT_DIR"
        git pull || echo "[WARNING] Failed to pull latest changes, continuing with existing code"
    else
        echo "[INFO] Cloning repository from GitHub..."
        echo "[INFO] Repository: $REPO_URL"
        if ! git clone "$REPO_URL" "$PROJECT_DIR"; then
            echo "[ERROR] Failed to clone repository"
            echo "[INFO] Please check the repository URL and your internet connection"
            exit 1
        fi
        cd "$PROJECT_DIR"
    fi
    echo ""
fi

# Detect OS
OS_TYPE=$(uname -s)
echo "[INFO] Detected OS: $OS_TYPE"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        echo "  macOS: brew install python3"
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
        echo "  Fedora/RHEL: sudo dnf install python3 python3-pip"
    fi
    exit 1
fi

echo "[INFO] Python found:"
python3 --version
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "[SUCCESS] Virtual environment created"
else
    echo "[INFO] Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi
echo ""

# Upgrade pip
echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
else
    echo "[INFO] No requirements.txt found, installing core dependencies..."
    pip install streamlit numpy scipy sympy matplotlib plotly pandas
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
fi
echo ""

echo "[SUCCESS] Installation complete!"
echo ""
echo "========================================"
echo "Starting Koios Application"
echo "========================================"
echo ""
echo "[INFO] Launching Streamlit application..."
echo "[INFO] The Koios app will open in your default browser at http://localhost:5000"
echo "[INFO] Press Ctrl+C to stop the server"
echo ""
echo ""
echo "You can now view the Koios app in your browser."
echo "  Local URL: http://localhost:5000"
echo ""

# Run the application
python -m streamlit run app.py --server.headless=false --server.address=localhost --server.port=5000 --logger.level=error

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Failed to start application"
    exit 1
fi
