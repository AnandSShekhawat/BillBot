# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding .env)
COPY app.py .
COPY templates/ templates/

# Create non-root user and set up directories
RUN useradd -m appuser && \
    mkdir -p /app/logs /app/uploads /app/cosmos_mock && \
    chown -R appuser:appuser /app/logs /app/uploads /app/cosmos_mock

# Switch to non-root user
USER appuser

# Expose port and set environment variable
ENV PORT=8000
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app:app"]