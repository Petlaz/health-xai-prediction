# Dockerfile for Health XAI Prediction Project
# Optimized for Week 5-6 XAI Implementation with LIME and SHAP

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY docker/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Create project structure
RUN mkdir -p /app/data /app/results /app/notebooks /app/src

# Copy application code
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app
ENV JUPYTER_ENABLE_LAB=yes

# Expose ports for Jupyter and Gradio
EXPOSE 8888
EXPOSE 7860

# Copy entrypoint script
COPY docker/entrypoint_app.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run entrypoint
ENTRYPOINT ["/entrypoint.sh"]