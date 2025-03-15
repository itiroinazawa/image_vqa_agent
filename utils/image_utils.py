"""
Utility functions for image handling and processing.
"""

import os
import uuid
import requests
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


def download_image(url, save_dir="temp_images"):
    """
    Download an image from a URL and save it to disk.

    Args:
        url (str): The URL of the image.
        save_dir (str, optional): The directory to save the image to.

    Returns:
        str: The path to the saved image.
    """
    try:
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(save_dir, filename)

        # Download the image
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        # Save the image
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Image downloaded and saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


def save_uploaded_image(image_data, save_dir="temp_images"):
    """
    Save an uploaded image to disk.

    Args:
        image_data (bytes): The image data.
        save_dir (str, optional): The directory to save the image to.

    Returns:
        str: The path to the saved image.
    """
    try:
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(save_dir, filename)

        # Save the image
        with open(filepath, "wb") as f:
            f.write(image_data)

        logger.info(f"Uploaded image saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save uploaded image: {e}")
        raise


def validate_image(image_path):
    """
    Validate that the file at the given path is a valid image.

    Args:
        image_path (str): The path to the image.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file {image_path}: {e}")
        return False


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for model input.

    Args:
        image_path (str): The path to the image.
        target_size (tuple, optional): The target size to resize the image to.

    Returns:
        PIL.Image.Image: The preprocessed image.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize the image
            img = img.resize(target_size)

        logger.info(f"Image {image_path} preprocessed successfully")
        return img
    except Exception as e:
        logger.error(f"Failed to preprocess image {image_path}: {e}")
        raise


def cleanup_temp_images(save_dir="temp_images", max_age_hours=24):
    """
    Clean up temporary images that are older than the specified age.

    Args:
        save_dir (str, optional): The directory containing the temporary images.
        max_age_hours (int, optional): The maximum age of images to keep, in hours.

    Returns:
        int: The number of files deleted.
    """
    try:
        if not os.path.exists(save_dir):
            return 0

        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0

        for filename in os.listdir(save_dir):
            filepath = os.path.join(save_dir, filename)

            # Skip directories
            if os.path.isdir(filepath):
                continue

            # Check file age
            file_age = current_time - os.path.getmtime(filepath)

            if file_age > max_age_seconds:
                os.remove(filepath)
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} temporary images")
        return deleted_count
    except Exception as e:
        logger.error(f"Failed to clean up temporary images: {e}")
        return 0
