"""
Main application for the VQA Agent.
"""

import os
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import shutil
from pathlib import Path
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqa_agent import VQAAgent
from utils.image_utils import (
    download_image,
    save_uploaded_image,
    validate_image,
    cleanup_temp_images,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("vqa_agent.log")],
)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="Visual Question Answering Agent",
    description="An AI agent capable of answering questions about images",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

# Create the static directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Create the temp_images directory
temp_images_dir = Path(__file__).parent.parent / "temp_images"
temp_images_dir.mkdir(exist_ok=True)

# Mount the static files directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Set up the templates
templates = Jinja2Templates(directory=templates_dir)

# Initialize the VQA Agent
# Note: In production, you might want to load these from environment variables
BLIP_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
# LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Global variable for the VQA Agent
vqa_agent = None


# Request models
class ImageUrlRequest(BaseModel):
    url: str
    question: str


class AnswerResponse(BaseModel):
    answer: str
    image_path: str


@app.on_event("startup")
async def startup_event():
    """Initialize the VQA Agent on startup."""
    global vqa_agent
    try:
        logger.info("Initializing VQA Agent...")
        vqa_agent = VQAAgent(
            blip_model_name=BLIP_MODEL_NAME, llm_model_name=LLM_MODEL_NAME
        )
        logger.info("VQA Agent initialized successfully")

        # Clean up old temporary images
        cleanup_temp_images()
    except Exception as e:
        logger.error(f"Failed to initialize VQA Agent: {e}")
        # We'll initialize it lazily when needed


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Return the index page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Visual Question Answering Agent</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1, h2 {
                color: #2c3e50;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .card {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="text"], textarea {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .tabs {
                display: flex;
                margin-bottom: 20px;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border: 1px solid #ddd;
                background-color: #f9f9f9;
                border-radius: 4px 4px 0 0;
                margin-right: 5px;
            }
            .tab.active {
                background-color: white;
                border-bottom: 1px solid white;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .result {
                margin-top: 20px;
                display: none;
            }
            .result img {
                max-width: 100%;
                max-height: 400px;
                margin-bottom: 15px;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                border: 4px solid rgba(0, 0, 0, 0.1);
                width: 36px;
                height: 36px;
                border-radius: 50%;
                border-left-color: #3498db;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <h1>Visual Question Answering Agent</h1>
        <p>Upload an image or provide an image URL and ask a question about it.</p>
        
        <div class="container">
            <div class="tabs">
                <div class="tab active" onclick="openTab(event, 'upload-tab')">Upload Image</div>
                <div class="tab" onclick="openTab(event, 'url-tab')">Image URL</div>
            </div>
            
            <div id="upload-tab" class="tab-content card active">
                <form id="upload-form">
                    <div class="form-group">
                        <label for="upload-image">Upload Image:</label>
                        <input type="file" id="upload-image" name="image" accept="image/*" required>
                    </div>
                    <div class="form-group">
                        <label for="upload-question">Question:</label>
                        <input type="text" id="upload-question" name="question" placeholder="Ask a question about the image..." required>
                    </div>
                    <button type="submit">Submit</button>
                </form>
            </div>
            
            <div id="url-tab" class="tab-content card">
                <form id="url-form">
                    <div class="form-group">
                        <label for="image-url">Image URL:</label>
                        <input type="text" id="image-url" name="url" placeholder="https://example.com/image.jpg" required>
                    </div>
                    <div class="form-group">
                        <label for="url-question">Question:</label>
                        <input type="text" id="url-question" name="question" placeholder="Ask a question about the image..." required>
                    </div>
                    <button type="submit">Submit</button>
                </form>
            </div>
            
            <div class="loading">
                <div class="spinner"></div>
                <p>Processing your request... This may take a moment.</p>
            </div>
            
            <div id="result" class="result card">
                <h2>Result</h2>
                <img id="result-image" src="" alt="Uploaded image">
                <div>
                    <h3>Question:</h3>
                    <p id="result-question"></p>
                </div>
                <div>
                    <h3>Answer:</h3>
                    <p id="result-answer"></p>
                </div>
            </div>
        </div>
        
        <script>
            function openTab(evt, tabName) {
                var i, tabContent, tabLinks;
                tabContent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabContent.length; i++) {
                    tabContent[i].className = tabContent[i].className.replace(" active", "");
                }
                tabLinks = document.getElementsByClassName("tab");
                for (i = 0; i < tabLinks.length; i++) {
                    tabLinks[i].className = tabLinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).className += " active";
                evt.currentTarget.className += " active";
            }
            
            document.getElementById('upload-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const imageFile = document.getElementById('upload-image').files[0];
                const question = document.getElementById('upload-question').value;
                
                if (!imageFile) {
                    alert('Please select an image file.');
                    return;
                }
                
                formData.append('image', imageFile);
                formData.append('question', question);
                
                document.querySelector('.loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to process the request.');
                    }
                    
                    const data = await response.json();
                    displayResult(data, question);
                } catch (error) {
                    console.error('Error:', error);
                    alert('An error occurred: ' + error.message);
                } finally {
                    document.querySelector('.loading').style.display = 'none';
                }
            });
            
            document.getElementById('url-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const url = document.getElementById('image-url').value;
                const question = document.getElementById('url-question').value;
                
                document.querySelector('.loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {
                    const response = await fetch('/api/url', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            url: url,
                            question: question
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to process the request.');
                    }
                    
                    const data = await response.json();
                    displayResult(data, question);
                } catch (error) {
                    console.error('Error:', error);
                    alert('An error occurred: ' + error.message);
                } finally {
                    document.querySelector('.loading').style.display = 'none';
                }
            });
            
            function displayResult(data, question) {
                document.getElementById('result-image').src = '/images/' + data.image_path.split('/').pop();
                document.getElementById('result-question').textContent = question;
                document.getElementById('result-answer').textContent = data.answer;
                document.getElementById('result').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/upload", response_model=AnswerResponse)
async def upload_image(image: UploadFile = File(...), question: str = Form(...)):
    """
    Handle image upload and answer a question about it.
    """
    global vqa_agent

    try:
        # Initialize the VQA Agent if it's not already initialized
        if vqa_agent is None:
            logger.info("Initializing VQA Agent...")
            vqa_agent = VQAAgent(
                blip_model_name=BLIP_MODEL_NAME, llm_model_name=LLM_MODEL_NAME
            )
            logger.info("VQA Agent initialized successfully")

        # Save the uploaded image
        image_content = await image.read()
        image_path = save_uploaded_image(image_content)

        # Validate the image
        if not validate_image(image_path):
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Answer the question
        answer = vqa_agent.answer_question(image_path, question)

        return {"answer": answer, "image_path": image_path}
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/url", response_model=AnswerResponse)
async def process_image_url(request: ImageUrlRequest):
    """
    Process an image URL and answer a question about it.
    """
    global vqa_agent

    try:
        # Initialize the VQA Agent if it's not already initialized
        if vqa_agent is None:
            logger.info("Initializing VQA Agent...")
            vqa_agent = VQAAgent(
                blip_model_name=BLIP_MODEL_NAME, llm_model_name=LLM_MODEL_NAME
            )
            logger.info("VQA Agent initialized successfully")

        # Download the image
        image_path = download_image(request.url)

        # Validate the image
        if not validate_image(image_path):
            raise HTTPException(status_code=400, detail="Invalid image URL")

        # Answer the question
        answer = vqa_agent.answer_question(image_path, request.question)

        return {"answer": answer, "image_path": image_path}
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """
    Serve images from the temp_images directory.
    """
    image_path = os.path.join("temp_images", image_name)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    cleanup_temp_images()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
