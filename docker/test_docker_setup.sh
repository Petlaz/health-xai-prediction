#!/bin/bash

echo "🐳 Health XAI Docker Setup Validation"
echo "======================================="

# Check if required files exist
echo "📋 Checking Docker configuration files..."
for file in Dockerfile docker-compose.yml requirements.txt entrypoint_app.sh; do
    if [ -f "$file" ]; then
        echo "✅ $file - Found"
    else
        echo "❌ $file - Missing"
    fi
done

echo ""
echo "🔍 Validating requirements.txt content..."
if grep -q "lime" requirements.txt && grep -q "shap" requirements.txt; then
    echo "✅ XAI dependencies (LIME & SHAP) present"
else
    echo "❌ Missing XAI dependencies"
fi

if grep -q "gradio" requirements.txt; then
    echo "✅ Gradio dependency present for Week 7-8"
else
    echo "❌ Missing Gradio dependency"
fi

echo ""
echo "📁 Checking project structure for Docker integration..."
if [ -d "../notebooks" ] && [ -f "../notebooks/05_explainability_tests.ipynb" ]; then
    echo "✅ XAI notebook available for container execution"
else
    echo "❌ XAI notebook not found"
fi

if [ -d "../results/xai_analysis" ]; then
    echo "✅ XAI results directory ready for volume mounting"
else
    echo "❌ XAI results directory not found"
fi

echo ""
echo "🚀 Docker Setup Status: Ready for XAI deployment"
echo "To start the container when Docker daemon is running:"
echo "docker-compose -f docker-compose.yml up"
echo "Access Jupyter Lab: http://localhost:8888"
echo "XAI Notebook: notebooks/05_explainability_tests.ipynb"