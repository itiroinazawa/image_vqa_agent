"""
Example usage of the VQA Agent.
"""

import os
import argparse
import logging
from PIL import Image
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vqa_agent import VQAAgent
from utils.image_utils import download_image, validate_image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_image_file(image_path, question):
    """
    Process an image file and answer a question about it.

    Args:
        image_path (str): Path to the image file.
        question (str): Question to answer about the image.
    """
    # Validate the image
    if not validate_image(image_path):
        logger.error(f"Invalid image file: {image_path}")
        return

    # Initialize the VQA Agent
    logger.info("Initializing VQA Agent...")
    vqa_agent = VQAAgent()

    # Answer the question
    logger.info(f"Processing question: {question}")
    answer = vqa_agent.answer_question(image_path, question)

    # Display the image
    try:
        image = Image.open(image_path)
        image.show()
    except Exception as e:
        logger.error(f"Failed to display image: {e}")

    # Print the answer
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")


def process_image_url(url, question):
    """
    Process an image URL and answer a question about it.

    Args:
        url (str): URL of the image.
        question (str): Question to answer about the image.
    """
    try:
        # Download the image
        logger.info(f"Downloading image from URL: {url}")
        image_path = download_image(url)

        # Process the image
        process_image_file(image_path, question)
    except Exception as e:
        logger.error(f"Failed to process image URL: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Example usage of the VQA Agent")
    parser.add_argument("--image", type=str, help="Path to an image file")
    parser.add_argument("--url", type=str, help="URL of an image")
    parser.add_argument(
        "--question", type=str, required=True, help="Question to ask about the image"
    )

    args = parser.parse_args()

    if not args.image and not args.url:
        parser.error("Either --image or --url must be provided")

    if args.image:
        process_image_file(args.image, args.question)
    elif args.url:
        process_image_url(args.url, args.question)


# NOTE: Uncomment the following lines if you want to debug the example without passing args in vscode debugger
# def main():
#     """Main function."""
#     image_url = 'https://pt.egamersworld.com/uploads/blog/1/17/1735564816859_1735564816859.webp'
#     question = 'What are the characters in this image?'

#     process_image_url(image_url, question)


if __name__ == "__main__":
    main()
