# Use an official Python runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc files and allow logs to show immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy backend and frontend
COPY backend ./backend
COPY frontend ./frontend
COPY requirements.txt ./requirements.txt

# Install system deps that might be needed by some packages (opencv, pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose port 7860 (HF Spaces uses this by default for web services)
EXPOSE 7860

# If you mount / serve frontend static files from FastAPI,
# make sure your backend.main mounts frontend (see next step).
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]



# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile


