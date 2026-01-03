# Koios - Installation and Deployment Guide

## Quick Start (Recommended)

The installation scripts will automatically:
- Check for Git and Python
- Clone/update the repository from GitHub
- Create a virtual environment
- Install all dependencies
- Launch the application

### First-Time Setup

### Windows
```bash
install_and_run.bat
```

### Linux / macOS
```bash
chmod +x install_and_run.sh
./install_and_run.sh
```

### Running from Project Directory

If you've already cloned the repository, simply run the script from within the project directory and it will skip cloning and proceed with setup.

## Prerequisites

- **Git** (for automatic repository cloning)
- **Python 3.8 or higher**
- **pip** (Python package installer)

### Installing Git

#### Windows
Download from [git-scm.com](https://git-scm.com/download/win)

#### macOS
```bash
# Using Homebrew
brew install git

# Or use Xcode Command Line Tools
xcode-select --install
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install git
```

#### Linux (Fedora/RHEL)
```bash
sudo dnf install git
```

### Installing Python

#### Windows
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer and check "Add Python to PATH"
3. Verify: `python --version`

#### macOS
```bash
# Using Homebrew
brew install python3
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
```

#### Linux (Fedora/RHEL)
```bash
sudo dnf install python3 python3-pip
```

## Manual Installation

### 1. Clone or Download
```bash
cd path/to/Koios
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run app.py
```

## Deployment Options

### Development Mode (Default)
```bash
# Windows
install_and_run.bat

# Linux/macOS
./install_and_run.sh
```

### Production Mode
```bash
# Custom port and host
streamlit run app.py --server.port 8080 --server.address 0.0.0.0

# With specific configuration
streamlit run app.py --server.port 5000 --server.headless=true
```

### Background Service (Linux/macOS)
```bash
# Using nohup
nohup ./install_and_run.sh &

# Using screen
screen -S koios
./install_and_run.sh
# Press Ctrl+A, then D to detach
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8501 (default Streamlit port)
# Windows
netstat -ano | findstr :8501

# Linux/macOS
lsof -i :8501

# Change port
streamlit run app.py --server.port 8502
```

### Module Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Permission Denied (Linux/macOS)
```bash
# Make script executable
chmod +x install_and_run.sh
```

### Virtual Environment Issues
```bash
# Remove and recreate
# Windows
rmdir /s venv
python -m venv venv

# Linux/macOS
rm -rf venv
python3 -m venv venv
```

## Environment Configuration

Create a `.streamlit/config.toml` file for custom configuration:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = false

[theme]
primaryColor = "#0066cc"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[browser]
gatherUsageStats = false
```

## System Requirements

- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 500MB for application and dependencies
- **Network**: Internet connection for initial setup

## Features Included

- Scientific Calculator
- Matrix Operations
- Calculus Tools
- Equation Solvers
- Physics Simulations
- Engineering Analysis
- Complex Analysis
- Tensor Calculus
- Numerical Methods
- Optimization Algorithms
- Interactive Visualizations

## Support

For issues or questions, check the documentation or create an issue in the project repository.

## License

See LICENSE file for details.
