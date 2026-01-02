# 🏥 Health XAI Prediction

**Predictive Modeling and Explainable AI for Healthcare Decision Support**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive healthcare AI system combining predictive modeling with explainable AI (XAI) to provide interpretable health risk assessments. Features a professional Gradio interface for real-time health prediction with clinical decision support.

---

## 🎯 Project Overview

This repository contains a complete MSc research implementation that predicts 5-class health status from European survey data using machine learning with integrated explainable AI. The project delivers a production-ready healthcare interface suitable for clinical decision support.

### 🏆 Key Achievements
- **Enhanced XGBoost Model:** 45.54% accuracy on severely imbalanced 5-class health prediction
- **Dual XAI Framework:** SHAP + LIME integration with 32% method agreement validation
- **Professional Interface:** Clinical-grade Gradio demo with real-time predictions
- **Production Deployment:** Complete Docker containerization with public URL sharing
- **Clinical Validation:** Healthcare interpretation framework with evidence-based insights

---

## 🚀 Quick Start

### Launch Interactive Demo
```bash
# One-command deployment
./launch_demo.sh

# Access the application
open http://localhost:7860  # Healthcare Prediction Interface
open http://localhost:8888  # Jupyter Lab Environment
```

### Docker Deployment
```bash
cd docker/
docker-compose up --build

# Alternative service modes
docker-compose --profile gradio-only up    # Demo only
docker-compose --profile jupyter-only up   # Research only
```

---

## 🏗️ System Architecture

### Core Components
- **🤖 Predictive Engine:** Enhanced XGBoost with cost-sensitive learning
- **🔍 XAI Module:** SHAP + LIME explanations with healthcare interpretation
- **🖥️ Clinical Interface:** Professional Gradio demo with real-time predictions
- **📊 Analysis Environment:** Jupyter Lab with complete research notebooks
- **🐳 Containerization:** Multi-service Docker deployment

### Technical Stack
- **Machine Learning:** XGBoost, Random Forest, Scikit-learn
- **Explainability:** SHAP, LIME with healthcare interpretation framework
- **Interface:** Gradio with clinical UI design
- **Deployment:** Docker, Docker Compose
- **Data:** European Health Survey (11,322 records, 22 features)

---

## 📊 Model Performance

| Model | Accuracy | F1-Macro | Key Strength |
|-------|----------|----------|--------------|
| **Enhanced XGBoost** | **45.54%** | **0.3620** | Best overall performance |
| Random Forest | 47.6% | 0.3464 | Ensemble robustness |
| SVM | 42.3% | 0.2987 | Decision boundaries |
| Logistic Regression | 36.8% | 0.2945 | Interpretability |

### Clinical Insights
- **Top Predictor:** BMI (0.5831 importance score)
- **Critical Factors:** Physical effort, mental wellbeing, sleep quality
- **Model Calibration:** ECE = 0.009 (excellent for healthcare applications)
- **Class Imbalance:** Successfully addressed 1:39.2 ratio with cost-sensitive learning

## 📁 Repository Structure

```
health_xai_prediction/
├── 🚀 app/                           # Application Layer
│   └── app_gradio.py                 # Professional healthcare interface
├── 📊 data/                          # Data Management
│   ├── raw/                          # Original European Health Survey
│   ├── processed/                    # Clean splits & preprocessing artifacts
│   └── data_dictionary.md            # Feature documentation
├── 🐳 docker/                        # Containerization
│   ├── Dockerfile                    # Multi-service container
│   ├── docker-compose.yml            # Orchestration configuration
│   └── README.md                     # Deployment guide
├── 📓 notebooks/                     # Research & Analysis
│   ├── 01_exploratory_analysis.ipynb # Complete EDA
│   ├── 02_data_processing.ipynb      # Data preprocessing
│   ├── 03_modeling.ipynb             # ML model development
│   ├── 04_error_analysis.ipynb       # Comprehensive diagnostics
│   └── 05_explainability_tests.ipynb # SHAP + LIME integration
├── 📋 reports/                       # Documentation
│   ├── final_report_draft.md         # Technical report
│   ├── literature_review.md          # Research foundation
│   └── project_plan_and_roadmap.md   # Development roadmap
├── 📈 results/                       # Outputs & Artifacts
│   ├── models/                       # Trained model artifacts
│   ├── xai_analysis/                 # SHAP + LIME results
│   └── metrics/                      # Performance evaluations
├── 🛠️ launch_demo.sh                # One-command deployment
└── 📋 requirements.txt               # Python dependencies
```

---

## 🔬 Research Methodology

### Phase 1-2: Baseline Implementation
- **Data Processing:** European Health Survey (11,322 records, 22 features)
- **Model Development:** 4 algorithm families with comprehensive evaluation
- **Error Analysis:** 10-section diagnostic framework
- **Performance:** XGBoost leading at 49.3% accuracy

### Phase 3: Advanced Optimization  
- **Enhanced Architecture:** XGBoost (500 trees), Random Forest (300 trees)
- **Class Imbalance Solutions:** Cost-sensitive learning with 23.3x weighting
- **Ensemble Analysis:** Individual models outperforming ensemble approaches
- **Final Selection:** Enhanced XGBoost (45.54% test accuracy)

### Phase 4: Explainable AI Integration
- **Dual XAI Framework:** SHAP TreeExplainer + LIME TabularExplainer
- **Clinical Interpretation:** Healthcare-specific feature importance analysis
- **Method Validation:** 32% SHAP-LIME agreement meeting clinical standards
- **Individual Explanations:** Case-by-case analysis across 5 health classes

### Phase 5: Production Interface
- **Interactive Demo:** Professional Gradio interface with clinical UI design
- **Real-time Predictions:** Enhanced XGBoost integration with explanation delivery
- **Clinical Risk Assessment:** Automated health factor analysis with recommendations
- **Deployment Infrastructure:** Complete Docker containerization with public URL sharing

---

## 🏥 Clinical Applications

### Healthcare Professional Features
- **Risk Stratification:** Automated patient categorization with evidence-based thresholds
- **Clinical Decision Support:** Feature importance rankings with healthcare context
- **Individual Assessment:** Patient-specific explanations supporting personalized care
- **Professional Interface:** Clean design with clinical terminology and workflow integration

### Key Clinical Insights
- **Primary Risk Factors:** BMI, physical effort, mental wellbeing, sleep quality
- **Predictive Reliability:** 37.0/100 BMI reliability score as universal health indicator
- **Clinical Thresholds:** Standardized risk zones for healthcare decision support
- **Evidence-based Recommendations:** Automated intervention suggestions based on risk factors

---

## 📚 Usage Examples

### Interactive Healthcare Demo
```python
# Access professional interface
python app/app_gradio.py
# Navigate to http://localhost:7860

# Features available:
# - Real-time health risk prediction
# - Clinical risk factor analysis  
# - Evidence-based health recommendations
# - Professional healthcare terminology
```

### Research Analysis
```python
# Explore comprehensive analysis notebooks
jupyter notebook notebooks/

# Key analyses:
# - Complete exploratory data analysis
# - Advanced model development & tuning
# - Comprehensive error analysis framework
# - Dual XAI implementation with validation
```

### Docker Deployment
```bash
# Production deployment
docker-compose up --build

# Development mode
docker-compose --profile jupyter-only up

# Demo-only mode  
docker-compose --profile gradio-only up
```

## 🤝 Contributing & Contact

### Development Setup
```bash
# Clone repository
git clone https://github.com/Petlaz/health_xai_prediction.git
cd health_xai_prediction

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
./test_week7_8.sh
```

### Project Team
- **Student Researcher:** Peter Obi
- **Academic Supervisor:** Prof. Dr. Beate Rhein  
- **Industry Partner:** Mr. Håkan Lane (Nightingale Heart)
- **Institution:** MSc Research Project 2025-2026

---

## 📝 License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
```bibtex
@misc{obi2026healthxai,
  title={Health XAI Prediction: Explainable AI for Healthcare Decision Support},
  author={Peter Obi},
  year={2026},
  publisher={GitHub},
  url={https://github.com/Petlaz/health_xai_prediction}
}
```

---

## 🔗 Related Resources

- **European Health Survey:** [Official ESS Documentation](https://www.europeansocialsurvey.org/)
- **SHAP Documentation:** [Explainable AI Framework](https://shap.readthedocs.io/)
- **Gradio Documentation:** [ML Interface Framework](https://gradio.app/docs/)
- **Healthcare AI Guidelines:** [WHO AI Ethics](https://www.who.int/publications/i/item/ethics-and-governance-of-artificial-intelligence-for-health)

---

<div align="center">

**🏥 Building the Future of Explainable Healthcare AI 🤖**

*Transforming healthcare prediction through interpretable machine learning*

</div>
