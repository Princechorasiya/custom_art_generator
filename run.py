#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import threading
import argparse
import logging
from waitress import serve
from style_art_generator.api.app import app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_comfyui(port):
    """Start the ComfyUI server in a separate process"""
    comfyui_path = os.path.abspath("ComfyUI")
    if not os.path.exists(comfyui_path):
        logger.error(f"ComfyUI directory not found at {comfyui_path}")
        sys.exit(1)
    
    logger.info(f"Starting ComfyUI server on port {port}...")
    
    # Build the command
    cmd = [
        sys.executable,
        "main.py",
        "--port", str(port),
        "--listen", "0.0.0.0"
    ]
    
    # Start ComfyUI process
    process = subprocess.Popen(
        cmd,
        cwd=comfyui_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Monitor the output
    for line in process.stdout:
        line = line.strip()
        if "Starting server" in line:
            logger.info("ComfyUI server is running")
            break
    
    return process

def main():
    parser = argparse.ArgumentParser(description="Run the Style Art Generator")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Flask app")
    parser.add_argument("--comfyui-port", type=int, default=8188, help="Port for ComfyUI server")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    parser.add_argument("--no-comfyui", action="store_true", help="Don't start ComfyUI server")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    
    args = parser.parse_args()
    
    # Start ComfyUI in a separate process
    comfyui_process = None
    if not args.no_comfyui:
        comfyui_process = start_comfyui(args.comfyui_port)
        # Wait for ComfyUI to start
        time.sleep(5)
    
    # Configure app environment variables
    os.environ["COMFYUI_PORT"] = str(args.comfyui_port)
    
    # Start Flask app
    try:
        host = "0.0.0.0"  # Listen on all interfaces
        
        if args.production:
            logger.info(f"Starting production server on port {args.port}...")
            serve(app, host=host, port=args.port)
        else:
            logger.info(f"Starting development server on port {args.port}...")
            app.run(host=host, port=args.port, debug=args.debug)
    
    finally:
        # Clean up processes
        if comfyui_process:
            logger.info("Shutting down ComfyUI server...")
            comfyui_process.terminate()
            comfyui_process.wait()

if __name__ == "__main__":
    main()