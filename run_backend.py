#!/usr/bin/env python3
"""
Backend runner script
"""

import uvicorn
import sys

if __name__ == "__main__":
    # Check for debug mode
    debug_mode = "--debug" in sys.argv
    
    if debug_mode:
        print("🚀 Starting backend in DEBUG mode...")
        # Enable debug logging and reload
        uvicorn.run(
            "backend.app.main:app",
            host="localhost",
            port=8000,
            reload=True,
            log_level="debug"
        )
    else:
        print("🚀 Starting backend...")
        uvicorn.run(
            "backend.app.main:app",
            host="localhost",
            port=8000,
            reload=True
        )