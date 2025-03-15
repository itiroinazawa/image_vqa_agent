"""
Setup script for the Visual Question Answering (VQA) AI Agent package.
"""

from setuptools import setup, find_packages
import os
import shutil
import argparse
import logging

# Package metadata
NAME = "image_vqa_agent"
DESCRIPTION = "AI agent capable of performing Visual Question Answering (VQA) based on uploaded images or image URLs"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your.email@example.com"
URL = "https://github.com/yourusername/image_vqa_agent"
VERSION = "0.1.0"

# Read the contents of your README file
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Get the list of requirements
with open("requirements.txt", encoding="utf-8") as f:
    REQUIRED = f.read().splitlines()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the environment for the VQA Agent."""
    # Create .env file if it doesn't exist
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        logger.info("Creating .env file from .env.example")
        shutil.copy(".env.example", ".env")

    # Create necessary directories
    dirs = ["temp_images", "app/templates", "app/static"]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            logger.info(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

    logger.info("Environment setup complete")


def install_dependencies(force=False):
    """Install the required dependencies."""
    import subprocess

    if force or input("Install dependencies? (y/n): ").lower() == "y":
        logger.info("Installing dependencies...")
        subprocess.run(["pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed")


def main():
    """Main function for setup.py when run directly."""
    parser = argparse.ArgumentParser(
        description="Setup script for the VQA Agent project"
    )
    parser.add_argument(
        "--install-deps", action="store_true", help="Install dependencies"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force installation without prompting"
    )

    args = parser.parse_args()

    setup_environment()

    if args.install_deps:
        install_dependencies(force=args.force)

    logger.info(
        "Setup complete. You can now run the application with: python app/main.py"
    )


if __name__ == "__main__":
    main()
else:
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
        include_package_data=True,
        install_requires=REQUIRED,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires=">=3.8",
        entry_points={
            "console_scripts": [
                "vqa_agent=app.main:main",
            ],
        },
    )
