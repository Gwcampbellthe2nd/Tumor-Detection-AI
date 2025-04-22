FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Set default PORT (Cloud Run overrides this with env variable)
ENV PORT=8080

# Start the FastAPI app with uvicorn
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
