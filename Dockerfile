FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlite3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Manually install apscheduler if not yet in requirements.txt (it is added in previous step)

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p logs data ml_models alerts

# Expose Streamlit port
EXPOSE 8501

# Default command runs the Streamlit App
CMD ["streamlit", "run", "crypto_ai/web/Home.py", "--server.address=0.0.0.0"]
