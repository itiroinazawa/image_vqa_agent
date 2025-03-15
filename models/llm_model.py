"""
LLM model implementation for language reasoning and answer generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


class LLMModel:
    """
    A class that implements a Language Model for reasoning and answer generation.
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", device=None):
    """

    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        """
        Initialize the LLM model.

        Args:
            model_name (str): The name of the LLM model to use.
            device (str, optional): The device to run the model on. If None, will use CUDA if available.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading LLM model {model_name} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

    def generate_response(self, prompt, max_length=512):
        """
        Generate a response for the given prompt.

        Args:
            prompt (str): The prompt to generate a response for.
            max_length (int, optional): The maximum length of the generated response.

        Returns:
            str: The generated response.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    attention_mask=inputs.attention_mask,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # Remove the prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt) :].strip()

            return response
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise

    def answer_with_context(self, question, image_caption, image_description=None):
        """
        Generate an answer to a question with image context.

        Args:
            question (str): The question to answer.
            image_caption (str): The caption of the image.
            image_description (str, optional): Additional description of the image.

        Returns:
            str: The generated answer.
        """
        context = f"Image caption: {image_caption}\n"
        if image_description:
            context += f"Image description: {image_description}\n"

        prompt = f"""Based on the following image information:
                {context}

                Please answer this question: {question}

                Answer:"""

        return self.generate_response(prompt)
