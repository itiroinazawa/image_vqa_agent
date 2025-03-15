# Visual Question Answering (VQA) AI Agent

This project implements an AI agent capable of performing Visual Question Answering (VQA) based on uploaded images or image URLs. The agent processes images, extracts visual information, and generates meaningful answers to user queries related to the images.

## Features

- Accept images via file upload or URL input
- Process images and extract relevant visual information
- Generate meaningful answers to user queries related to the image
- Utilize efficient open-source models for both image processing and language reasoning
- Modular architecture for easy expansion and fine-tuning
- Simple API and web interface for interaction

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/image_vqa_agent.git
cd image_vqa_agent
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app/main.py
```

## Usage

### Web Interface

Access the web interface at `http://localhost:8000` after starting the application.

### API

The API endpoints are available at:
- `POST /api/upload`: Upload an image file for VQA
- `POST /api/url`: Provide an image URL for VQA

### Example

```bash
python example.py --image path/to/image.jpg --question "What's in this image?"
```

```bash
python example.py --url https://example.com/image.jpg --question "What's in this image?"
```


## Architecture

The project is structured as follows:
- `app/`: Contains the FastAPI application and web interface
- `models/`: Contains the model implementations for image processing and VQA
- `utils/`: Contains utility functions for image handling and processing

## Models

This project uses the following open-source models:
- BLIP-2 for image understanding and feature extraction
- Llama 2 for language reasoning and answer generation

## License

[MIT License](LICENSE)
