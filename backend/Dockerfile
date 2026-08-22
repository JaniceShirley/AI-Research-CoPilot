# Use Python 3.11 slim image for a lightweight, production-ready environment
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for FAISS (libgomp1) and build packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code into /app
COPY . /app

# Ensure runtime data directories exist
RUN mkdir -p /app/uploads /app/vectorstore

# Expose port (Railway will override via dynamic $PORT environment variable)
EXPOSE 8000

# Start FastAPI app on 0.0.0.0 with dynamic $PORT fallback to 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
