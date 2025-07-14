#!/usr/bin/env python3
"""
Frontend runner script
"""

import subprocess
import sys
import os

def run_frontend():
    """Run the Streamlit frontend"""
    try:
        # Check for debug mode
        debug_mode = "--debug" in sys.argv
        
        if debug_mode:
            print("🎨 Starting frontend in DEBUG mode...")
            # Run with debug logging
            subprocess.run([
                "streamlit", "run", "frontend/Home.py",
                "--server.port", "8501",
                "--server.address", "localhost",
                "--logger.level", "debug"
            ], check=True)
        else:
            print("🎨 Starting frontend...")
            subprocess.run([
                "streamlit", "run", "frontend/Home.py",
                "--server.port", "8501",
                "--server.address", "localhost"
            ], check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running frontend: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    run_frontend()