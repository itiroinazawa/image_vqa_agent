"""
BLIP-2 model implementation for image understanding and feature extraction.
"""

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import logging

logger = logging.getLogger(__name__)


class BlipModel:
    """
    A class that implements the BLIP-2 model for image understanding and feature extraction.
    """

    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None):
        """
        Initialize the BLIP-2 model.

        Args:
            model_name (str): The name of the BLIP-2 model to use.
            device (str, optional): The device to run the model on. If None, will use CUDA if available.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading BLIP-2 model {model_name} on {self.device}...")

        try:
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            logger.info("BLIP-2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {e}")
            raise

    def process_image(self, image):
        """
        Process an image and prepare it for the model.

        Args:
            image (PIL.Image.Image or str): The image to process.

        Returns:
            torch.Tensor: The processed image.
        """
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to open image from path: {e}")
                raise

        inputs = self.processor(image, return_tensors="pt").to(self.device)
        return inputs

    def generate_caption(self, image):
        """
        Generate a caption for the given image.

        Args:
            image (PIL.Image.Image or str): The image to caption.

        Returns:
            str: The generated caption.
        """
        inputs = self.process_image(image)

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
                caption = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
            return caption
        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            raise

    def answer_question(self, image, question):
        """
        Answer a question about the given image.

        Args:
            image (PIL.Image.Image or str): The image to analyze.
            question (str): The question to answer.

        Returns:
            str: The answer to the question.
        """
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to open image from path: {e}")
                raise

        try:
            inputs = self.processor(image, question, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=100)
                answer = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()

            return answer
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise
