#!/bin/bash
# ============================================
# Koios - Linux/Mac Uninstall Script
# ============================================

echo ""
echo "========================================"
echo "Koios Uninstall"
echo "========================================"
echo ""

# Check if running from within the project directory
if [ ! -f "app.py" ]; then
    echo "[ERROR] This script must be run from the Koios project directory"
    exit 1
fi

echo "[WARNING] This will remove:"
echo "  - Virtual environment (venv folder)"
echo "  - Python cache files (__pycache__)"
echo "  - Temporary files"
echo ""
echo "[INFO] The source code will be preserved"
echo ""

read -p "Are you sure you want to uninstall? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "[INFO] Uninstall cancelled"
    exit 0
fi

echo ""
echo "[INFO] Starting uninstall process..."
echo ""

# Remove virtual environment
if [ -d "venv" ]; then
    echo "[INFO] Removing virtual environment..."
    rm -rf venv
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Virtual environment removed"
    else
        echo "[WARNING] Failed to remove virtual environment"
    fi
else
    echo "[INFO] No virtual environment found"
fi
echo ""

# Remove Python cache files
echo "[INFO] Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "[SUCCESS] Python cache files removed"
echo ""

# Remove .pytest_cache if exists
if [ -d ".pytest_cache" ]; then
    echo "[INFO] Removing pytest cache..."
    rm -rf .pytest_cache
    echo "[SUCCESS] Pytest cache removed"
    echo ""
fi

# Remove .streamlit cache if exists
if [ -d ".streamlit" ]; then
    read -p "Remove Streamlit configuration (includes telemetry settings)? (y/N): " STREAMLIT
    if [[ "$STREAMLIT" =~ ^[Yy]$ ]]; then
        rm -rf .streamlit
        echo "[SUCCESS] Streamlit configuration removed"
        echo ""
    else
        echo "[INFO] Streamlit configuration preserved (telemetry disabled)"
        echo ""
    fi
fi

echo ""
echo "========================================"
echo "Uninstall Complete"
echo "========================================"
echo ""
echo "[SUCCESS] Koios has been uninstalled"
echo ""
echo "[INFO] Source code has been preserved"
echo "[INFO] To completely remove Koios, delete this directory"
echo ""

read -p "Do you want to remove the entire project directory? (y/N): " REMOVE_ALL
if [[ "$REMOVE_ALL" =~ ^[Yy]$ ]]; then
    echo ""
    echo "[WARNING] This will delete ALL files including source code!"
    read -p "Are you absolutely sure? (y/N): " FINAL_CONFIRM
    if [[ "$FINAL_CONFIRM" =~ ^[Yy]$ ]]; then
        CURRENT_DIR=$(pwd)
        cd ..
        echo "[INFO] Removing project directory..."
        rm -rf "$CURRENT_DIR"
        echo "[SUCCESS] Project directory removed"
        exit 0
    fi
fi

echo ""
echo "[INFO] You can reinstall anytime by running ./install_and_run.sh"
