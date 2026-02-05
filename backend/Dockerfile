FROM python:3.12-slim

# Install system dependencies for Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

# Install CPU-only PyTorch first (smaller image)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models from HuggingFace (baked into image)
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('ELCA-SA/BPMN_Detection', 'model_object.pth'); \
    hf_hub_download('ELCA-SA/BPMN_Detection', 'model_arrow.pth')"

# Copy application code
COPY app/ app/

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
