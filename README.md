# Deepfake Detection Backend

Backend API for detecting deepfake videos using a trained deep learning model.

## Features
- Detects deepfake videos
- Returns prediction (`isFake`) and confidence score
- FastAPI backend with Swagger UI
- Dockerized for easy deployment

## Requirements
- Python 3.10+
- FastAPI, PyTorch, OpenCV, Pillow, Uvicorn

## Installation
```bash
git clone <repo-url>
cd deepfake-back
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
