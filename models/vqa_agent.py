"""
VQA Agent implementation that combines image processing and language models.
"""

import logging
from langchain.agents import Tool
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import AIMessage, HumanMessage

from models.blip_model import BlipModel
from models.llm_model import LLMModel

logger = logging.getLogger(__name__)


class VQAAgent:
    """
    Visual Question Answering Agent that combines image processing and language models.
    """

    def __init__(
        self,
        blip_model_name="Salesforce/blip2-opt-2.7b",
        #llm_model_name="meta-llama/Llama-2-7b-chat-hf",
        llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=None,
    ):
        """
        Initialize the VQA Agent.

        Args:
            blip_model_name (str): The name of the BLIP model to use.
            llm_model_name (str): The name of the LLM model to use.
            device (str, optional): The device to run the models on. If None, will use CUDA if available.
        """
        logger.info("Initializing VQA Agent...")
        self.blip_model = BlipModel(model_name=blip_model_name, device=device)
        self.llm_model = LLMModel(model_name=llm_model_name, device=device)
        logger.info("VQA Agent initialized successfully")

    def process_image(self, image_path):
        """
        Process an image and extract visual information.

        Args:
            image_path (str): The path to the image.

        Returns:
            dict: A dictionary containing the extracted visual information.
        """
        logger.info(f"Processing image: {image_path}")
        try:
            caption = self.blip_model.generate_caption(image_path)

            # Get additional information by asking specific questions about the image
            colors = self.blip_model.answer_question(
                image_path, "What are the main colors in this image?"
            )
            objects = self.blip_model.answer_question(
                image_path, "What objects can you see in this image?"
            )
            scene = self.blip_model.answer_question(
                image_path, "Describe the scene in this image."
            )

            visual_info = {
                "caption": caption,
                "colors": colors,
                "objects": objects,
                "scene": scene,
            }

            logger.info(f"Image processed successfully: {caption}")
            return visual_info
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise

    def answer_question(self, image_path, question):
        """
        Answer a question about an image.

        Args:
            image_path (str): The path to the image.
            question (str): The question to answer.

        Returns:
            str: The answer to the question.
        """
        logger.info(f"Answering question: {question}")
        try:
            # First, try to get a direct answer from the BLIP model
            direct_answer = self.blip_model.answer_question(image_path, question)

            # Extract visual information from the image
            visual_info = self.process_image(image_path)

            # Combine the direct answer with the visual information using the LLM
            context = f"""
Caption: {visual_info['caption']}
Colors: {visual_info['colors']}
Objects: {visual_info['objects']}
Scene: {visual_info['scene']}
Direct answer from image model: {direct_answer}
"""

            # Generate a more comprehensive answer using the LLM
            prompt = f"""Based on the following image information:
{context}

Please provide a detailed and accurate answer to this question: {question}

Answer:"""

            final_answer = self.llm_model.generate_response(prompt)
            logger.info(f"Question answered successfully")

            return final_answer
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise

    def create_langchain_agent(self):
        """
        Create a LangChain agent for the VQA task.

        Returns:
            AgentExecutor: The LangChain agent executor.
        """
        # Define tools for the agent
        tools = [
            Tool(
                name="GenerateImageCaption",
                func=lambda image_path: self.blip_model.generate_caption(image_path),
                description="Generates a caption for the given image. Input should be a path to an image file.",
            ),
            Tool(
                name="AnswerImageQuestion",
                func=lambda args: self.blip_model.answer_question(
                    args["image_path"], args["question"]
                ),
                description="Answers a specific question about the given image. Input should be a dictionary with 'image_path' and 'question' keys.",
            ),
            Tool(
                name="ProcessImageDetails",
                func=lambda image_path: self.process_image(image_path),
                description="Processes an image and extracts detailed visual information. Input should be a path to an image file.",
            ),
            Tool(
                name="GenerateComprehensiveAnswer",
                func=lambda args: self.answer_question(
                    args["image_path"], args["question"]
                ),
                description="Generates a comprehensive answer to a question about an image using both vision and language models. Input should be a dictionary with 'image_path' and 'question' keys.",
            ),
        ]

        # Convert tools to OpenAI functions
        functions = [format_tool_to_openai_function(t) for t in tools]

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Visual Question Answering AI assistant. You can analyze images and answer questions about them.",
                ),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create the LangChain agent
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | self.llm_model.model
            | OpenAIFunctionsAgentOutputParser()
        )

        # Create the agent executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        return agent_executor
