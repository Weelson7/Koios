import subprocess
import sys
import os

def main():
    """Launch Koios app"""
    print("Starting Koios Mathematical Toolset...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("ERROR: Streamlit not installed!")
        print("Please run: pip install streamlit numpy scipy sympy pandas matplotlib plotly")
        sys.exit(1)
    
    # Launch the app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py", 
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nShutting down Koios...")
    except Exception as e:
        print(f"Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()