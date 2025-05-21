FROM python:3.12-slim

WORKDIR /app

# Install uv (faster pip replacement)
RUN pip install --no-cache-dir uv

# Copy only requirements.txt and install dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
