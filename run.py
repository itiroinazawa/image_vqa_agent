"""
Run script for the VQA Agent application.
"""

import os
import argparse
import logging
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_app(host=None, port=None, reload=True):
    """Run the FastAPI application."""
    # Use environment variables if not specified
    host = host or os.getenv("HOST", "0.0.0.0")
    port = port or int(os.getenv("PORT", 8000))
    reload_flag = "--reload" if reload else ""

    logger.info(f"Starting VQA Agent application on http://{host}:{port}")

    # Run the application using uvicorn
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Application stopped")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the VQA Agent application")
    parser.add_argument("--host", type=str, help="Host to run the application on")
    parser.add_argument("--port", type=int, help="Port to run the application on")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")

    args = parser.parse_args()

    # Check if setup is needed
    if not os.path.exists(".env"):
        logger.info("Environment not set up. Running setup...")
        subprocess.run([sys.executable, "setup.py"])

    # Run the application
    run_app(host=args.host, port=args.port, reload=not args.no_reload)


if __name__ == "__main__":
    main()
