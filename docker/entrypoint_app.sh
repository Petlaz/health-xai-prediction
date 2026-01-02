#!/bin/bash

# Entrypoint script for Health XAI Prediction Docker Container
# Week 7-8 Implementation: Jupyter Lab + Gradio Demo

echo "🚀 Health XAI Prediction Container Starting..."
echo "📊 Week 7-8 Implementation: Interactive Gradio Demo"
echo "================================================"

# Set up environment
export PYTHONPATH=/app

# Create necessary directories
mkdir -p /app/results/xai_analysis
mkdir -p /app/results/models
mkdir -p /app/results/plots
mkdir -p /app/app

echo "✅ Environment configured"
echo "✅ XAI analysis directories created"
echo "✅ Gradio app directory ready"

# Check for required files
if [ -f "/app/app/app_gradio.py" ]; then
    echo "✅ Gradio application found"
else
    echo "⚠️ Gradio application not found, running in Jupyter-only mode"
fi

# Start services based on mode
if [ "$1" = "gradio" ]; then
    # Gradio-only mode
    echo "🎯 Starting Gradio Demo..."
    echo "🌐 Local URL: http://localhost:7860"
    echo "🌍 Public URL: Will be provided by Gradio sharing"
    echo "================================================"
    
    cd /app
    python app/app_gradio.py
    
elif [ "$1" = "jupyter" ]; then
    # Jupyter-only mode
    echo "🔬 Starting Jupyter Lab for XAI Analysis..."
    echo "📝 Access notebooks at: http://localhost:8888"
    echo "🎯 XAI Notebook: notebooks/05_explainability_tests.ipynb"
    echo "================================================"
    
    jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --token='' \
        --password='' \
        --NotebookApp.allow_origin='*' \
        --NotebookApp.disable_check_xsrf=True
        
else
    # Default: Combined mode with background Jupyter and foreground Gradio
    echo "🔬 Starting Jupyter Lab in background..."
    nohup jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --token='' \
        --password='' \
        --NotebookApp.allow_origin='*' \
        --NotebookApp.disable_check_xsrf=True > /app/jupyter.log 2>&1 &
    
    # Wait for Jupyter to start
    sleep 5
    
    echo "🎯 Starting Gradio Demo..."
    echo "📝 Jupyter Lab: http://localhost:8888"
    echo "🌐 Gradio Demo: http://localhost:7860"
    echo "🌍 Public URL: Will be provided by Gradio sharing"
    echo "================================================"
    
    # Start Gradio in foreground
    cd /app
    if [ -f "app/app_gradio.py" ]; then
        python app/app_gradio.py
    else
        echo "⚠️ Gradio app not found, keeping Jupyter Lab running"
        tail -f /app/jupyter.log
    fi
fi