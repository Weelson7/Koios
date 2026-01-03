#!/bin/bash

echo "====================================================="
echo "KOIOS - Advanced Mathematical Toolset"
echo "Linux Deployment Script"
echo "====================================================="
echo

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python3 using your package manager:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  Arch Linux: sudo pacman -S python python-pip"
    exit 1
fi

echo "Python3 detected successfully."

# Check if pip3 is available
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not available"
    echo "Please install pip3 using your package manager"
    exit 1
fi

echo "pip3 detected successfully."

# Install required packages
echo "Installing required packages..."
pip3 install streamlit matplotlib numpy pandas plotly scipy sympy

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install required packages"
    echo "You may need to run: sudo pip3 install streamlit matplotlib numpy pandas plotly scipy sympy"
    exit 1
fi

echo
echo "All packages installed successfully!"
echo
echo "Starting KOIOS application..."
echo

# Start the application
streamlit run app.py --server.port 5000 --server.address 0.0.0.0