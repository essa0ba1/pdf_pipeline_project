# PDF Pipeline - Hugging Face Spaces Deployment
# This Dockerfile sets up the environment for the PDF extraction pipeline

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - tesseract-ocr: For Pytesseract OCR backend
# - libgl1-mesa-glx: Required for OpenCV
# - libglib2.0-0: Required for various Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose the Streamlit port
EXPOSE 7860

# Set environment variables for Hugging Face
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers

# Create cache directory
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers

# Run the Streamlit application
CMD ["streamlit", "run", "main.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.headless", "true"]
