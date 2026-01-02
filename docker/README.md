# Docker Setup for Health XAI Prediction

**Week 7-8 Implementation: Containerized Gradio Demo with XAI Integration**

This directory contains Docker configuration for the complete Health XAI Prediction system, enabling reproducible deployment of both Jupyter Lab analysis environment and interactive Gradio demo.

---

## 📦 Container Components

### Services Available

| **Service** | **Port** | **Purpose** | **Access URL** |
|-------------|----------|-------------|----------------|
| **Jupyter Lab** | 8888 | XAI analysis notebooks | http://localhost:8888 |
| **Gradio Demo** | 7860 | Interactive health prediction | http://localhost:7860 |
| **XAI API** | 5000 | Optional API service | http://localhost:5000 |

### XAI Components Included
- ✅ **SHAP TreeExplainer** for Enhanced XGBoost model
- ✅ **LIME TabularExplainer** for individual case analysis
- ✅ **Healthcare Interpretation Framework** with clinical thresholds
- ✅ **Professional Gradio Interface** with both local and public URLs
- ✅ **Complete Model Pipeline** from Week 1-6 implementation

---

## 🚀 Quick Start

### Prerequisites
- Docker installed and running
- At least 4GB available RAM
- Ports 7860, 8888, 5000 available

### Launch Complete Environment

```bash
# Navigate to docker directory
cd /path/to/health_xai_prediction/docker

# Start all services
docker-compose up --build

# Alternative: Run in background
docker-compose up -d --build
```

### Access Applications

```bash
# Gradio Demo (Primary Interface)
open http://localhost:7860

# Jupyter Lab (Analysis Environment)
open http://localhost:8888

# Check container status
docker-compose ps
```

---

## 🏥 Gradio Demo Features

### Professional Healthcare Interface
- **Patient Assessment Form**: Comprehensive input collection
- **Real-time Prediction**: Enhanced XGBoost with 5-class health status
- **XAI Explanations**: SHAP-based feature importance visualization
- **Clinical Risk Assessment**: Automated risk factor identification
- **Healthcare Insights**: Evidence-based recommendations

### Interface Capabilities
- **Local Access**: http://localhost:7860
- **Public URL**: Automatically provided via Gradio sharing
- **Professional UI**: Healthcare-focused design with clinical terminology
- **Interactive Visualizations**: Feature importance and risk assessment plots
- **Responsive Design**: Works on desktop and tablet devices

---

## 📊 XAI Analysis Environment

### Jupyter Lab Setup
- **Pre-configured Environment**: All XAI dependencies installed
- **Volume Mounting**: Direct access to notebooks, data, and results
- **Complete Pipeline**: Week 1-6 analysis notebooks ready to run
- **XAI Notebooks**: `notebooks/05_explainability_tests.ipynb` with full implementation

### Available Analysis
- SHAP global and individual explanations
- LIME local interpretability analysis
- Healthcare risk factor assessment
- Method comparison and validation
- Clinical threshold analysis

---

## 🔧 Configuration Details

### Dockerfile Features
```dockerfile
# Base: Python 3.11 slim for optimal performance
# XAI Libraries: SHAP 0.41.0, LIME 0.2.0.1, Gradio 4.44
# Healthcare Libraries: scikit-learn, xgboost, pandas, matplotlib
# Container Optimization: Multi-stage build, dependency caching
```

### Docker Compose Services
```yaml
services:
  health-xai:     # Main application container
  xai-api:        # Optional API service (profile: api)
networks:
  health_xai_network  # Isolated network for services
```

### Volume Mounts
```yaml
volumes:
  - ../data:/app/data              # Dataset access
  - ../results:/app/results        # Model and XAI artifacts
  - ../notebooks:/app/notebooks    # Analysis notebooks
  - ../reports:/app/reports        # Documentation
  - ../app:/app/app               # Gradio application
```

---

## 🛠️ Development Commands

### Container Management
```bash
# Build containers
docker-compose build

# Start services
docker-compose up

# Stop services
docker-compose down

# View logs
docker-compose logs health-xai

# Access container shell
docker-compose exec health-xai bash
```

### Gradio Demo Management
```bash
# Run Gradio app directly
docker-compose exec health-xai python /app/app/app_gradio.py

# Check Gradio process
docker-compose exec health-xai ps aux | grep gradio

# View Gradio logs
docker-compose logs health-xai | grep gradio
```

### XAI Analysis Commands
```bash
# Run XAI notebook
docker-compose exec health-xai jupyter nbconvert \
  --to notebook --execute /app/notebooks/05_explainability_tests.ipynb

# Access XAI results
docker-compose exec health-xai ls -la /app/results/xai_analysis/

# Export XAI artifacts
docker-compose exec health-xai python -c "import joblib; print(list(joblib.load('/app/results/xai_analysis/xai_artifacts.joblib').keys()))"
```

---

## 🔍 Troubleshooting

### Common Issues

#### **Port Conflicts**
```bash
# Check port usage
lsof -i :7860
lsof -i :8888

# Modify ports in docker-compose.yml if needed
```

#### **Model Loading Issues**
```bash
# Verify model files
docker-compose exec health-xai ls -la /app/results/models/

# Check XAI artifacts
docker-compose exec health-xai ls -la /app/results/xai_analysis/

# Gradio will run in demo mode if models unavailable
```

#### **Memory Issues**
```bash
# Check container resources
docker stats health_xai_prediction

# Increase Docker memory limit if needed (Docker Desktop > Settings > Resources)
```

### Validation Script
```bash
# Run Docker setup validation
chmod +x test_docker_setup.sh
./test_docker_setup.sh
```

---

## 📋 API Service (Optional)

### Enable API Service
```bash
# Start with API profile
docker-compose --profile api up

# API will be available at http://localhost:5000
```

### API Endpoints (Future Development)
- `POST /predict` - Health prediction endpoint
- `POST /explain` - XAI explanation endpoint
- `GET /health` - Service health check
- `GET /models` - Available model information

---

## 🚀 Production Deployment

### Security Considerations
```bash
# For production, enable authentication
# Modify app_gradio.py:
# app.launch(auth=("username", "password"))

# Use environment variables for configuration
# Add .env file with production settings
```

### Performance Optimization
```bash
# Build optimized image
docker build -t health-xai-prod \
  --build-arg OPTIMIZE=true \
  -f Dockerfile .

# Use production compose file
docker-compose -f docker-compose.prod.yml up
```

---

## 📊 Week 7-8 Implementation Status

### ✅ Completed Features
- [x] Professional Gradio interface with healthcare focus
- [x] Real-time prediction with Enhanced XGBoost integration
- [x] SHAP-based feature importance visualization
- [x] Clinical risk factor assessment framework
- [x] Healthcare interpretation with evidence-based insights
- [x] Both local (localhost:7860) and public URL support
- [x] Containerized deployment with volume mounting
- [x] Professional UI design with clinical terminology

### 🔄 Ready for Week 9-10
- User experience testing and refinement
- Performance optimization for clinical workflows
- Advanced XAI visualizations (waterfall plots, interaction effects)
- Clinical validation with healthcare professionals
- Production deployment preparation

---

**Deployment Status:** Week 7-8 Gradio Demo Complete ✅  
**Container Architecture:** Multi-service XAI environment ready  
**Next Phase:** Clinical validation and user experience optimization

---

*Health XAI Prediction Project • Docker Implementation Guide*  
*Team: Health XAI Research Group • Supervisor: Prof. Dr. Beate Rhein • Industry Partner: Nightingale Heart*