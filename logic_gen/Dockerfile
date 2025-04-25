# Use Python 3.9 slim image as base

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential

# Install Python packages
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    uv \
    transformers \
    torch

# Clean up system dependencies
RUN apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variable for Python buffering
ENV PYTHONUNBUFFERED=1

# Start a bash shell by default
CMD ["/bin/bash"]
