# syntax=docker/dockerfile:1.4

# ---------- Base image ----------
FROM python:3.11-slim AS base

# ---------- Environment variables ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# ---------- System dependencies ----------
RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libssl-dev \
        libffi-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install heavy packages first (rarely change, better caching) ----------
RUN pip install --upgrade pip && pip install \
    torch==2.1.0 \
    xgboost==1.7.6

# ---------- Copy requirements and install remaining Python dependencies ----------
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# ---------- Copy project files ----------
COPY . /app

# ---------- (Optional) Security: Use non-root user ----------
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser

# ---------- Default command ----------
CMD ["bash"]
