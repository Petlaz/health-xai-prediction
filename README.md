# Health XAI Prediction

**Predictive Modeling and Local Explainable AI (XAI) in Healthcare**

This repository hosts a comprehensive MSc research project focused on predicting 5-class health status from European survey data using machine learning with explainable AI integration. The project follows a systematic biweekly development approach with complete **Week 1-6 implementation** covering baseline models, comprehensive error analysis, advanced class imbalance solutions, and local explainable AI integration.

---

## Week 1-6 Achievements Summary

### ✅ Phase 1-2: Complete Baseline Implementation (Week 1-2)
- **Robust data preprocessing pipeline:** 11,322 European Health Survey records with 22 numerical features
- **Multi-algorithm baseline models:** XGBoost (49.3% accuracy), Random Forest (47.6%), SVM (42.3%), Logistic Regression (36.8%)
- **Comprehensive evaluation framework:** Complete metrics calculation with calibration analysis
- **Production-ready artifacts:** Serialized models, scalers, and data splits for reproducible experiments

### ✅ Phase 3: Advanced Class Imbalance Solutions (Week 3-4)
- **Enhanced model architecture:** XGBoost with 500 trees, depth=8; Random Forest with 300 trees, depth=20
- **Cost-sensitive learning:** Balanced class weights with 23.3x emphasis on Very Bad health class
- **Individual vs ensemble analysis:** Enhanced XGBoost outperformed both hard and soft voting ensembles
- **Final model selection:** Optimized Enhanced XGBoost (Test F1-Macro: 0.3620, Accuracy: 45.54%)

### ✅ Phase 4: Local Explainable AI Integration (Week 5-6)
- **Dual XAI Implementation:** Complete SHAP TreeExplainer + LIME TabularExplainer integration
- **Healthcare Interpretation Framework:** BMI (0.5831), Physical Effort (0.4756) identified as top clinical predictors
- **Method Validation:** 32% SHAP-LIME agreement meets healthcare validation standards
- **Individual Case Analysis:** 5 health class case studies with local explanations and clinical insights
- **Docker Integration:** Containerized XAI modules for reproducible deployment
- **Research Accessibility:** 7 CSV files exported for healthcare professional and researcher collaboration

### 📊 Week 1-6 Complete Technical Results
| **Phase** | **Focus** | **Key Achievement** | **Final Performance** |
|-----------|-----------|---------------------|----------------------|
| **Phase 1-2** | Baseline Implementation | XGBoost baseline established | 49.3% accuracy, 0.3641 F1-Macro |
| **Phase 3** | Class Imbalance Solutions | Enhanced models with cost-sensitive learning | 45.54% accuracy, 0.3620 F1-Macro (test) |
| **Phase 4** | XAI Integration | SHAP + LIME with healthcare framework | 32% method agreement, BMI top predictor |
| **Status** | **Week 5-6 Complete** | **Ready for Week 7-8 Gradio Demo** | Complete XAI pipeline with clinical insights |

### 🎯 Week 5-6 XAI Key Findings
- **BMI emerges as most reliable predictor:** 37.0/100 reliability score combining importance and consistency
- **Physical Effort critical for Very Bad health:** Strongest impact on poor health outcomes
- **Sleep Quality provides consensus:** 60% agreement between SHAP and LIME methods
- **Mental Wellbeing consistently important:** Across all health classes and both XAI methods
- **Clinical thresholds established:** BMI high risk >0.62, Physical Effort >0.45 standardized units

---

## Critical Implementation Details

### 🔬 Error Analysis Deep Dive
Our comprehensive error analysis revealed several critical insights:

**Class Imbalance Challenge:**
- Severe 1:39.2 ratio between smallest and largest health classes
- All models exhibit majority class bias despite balancing attempts
- Requires advanced techniques (SMOTE, cost-sensitive learning) in Week 3-4

**Model Calibration Excellence:**
- Expected Calibration Error = 0.009 (excellent for healthcare applications)
- Strong alignment between predicted probabilities and actual outcomes  
- Meets clinical deployment standards for probability reliability

**Edge Case Prevalence:**
- 87.4% of samples classified as edge cases requiring robust handling
- High model disagreement (765 cases) indicates optimization opportunities
- XGBoost-Random Forest correlation (0.85) suggests ensemble potential

### 🎯 Feature Engineering Insights
**Validated top predictors across all models:**
1. **Self-rated health** (`numeric__health`): Dominates predictions across all algorithms
2. **BMI** (`numeric__bmi`): Consistent second-tier predictor with clinical relevance
3. **Psychological wellbeing** (`numeric__happy`): Strong correlation with health outcomes
4. **Sleep quality** (`numeric__slprl`): Important lifestyle factor for health prediction
5. **Physical activity** (`numeric__dosprt`): Significant predictor of health status

---

## Repository Structure
```
health_xai_prediction/
├── app/
│   ├── __init__.py                     # Application package
│   └── app_gradio.py                   # ✅ Week 7-8: Professional Gradio healthcare interface
├── data/
│   ├── raw/                            # Original survey datasets
│   ├── processed/                      # Clean splits and preprocessing artifacts
│   └── data_dictionary.md              # Feature documentation
├── docker/
│   ├── Dockerfile                      # ✅ Week 7-8: Enhanced container with Gradio support
│   ├── docker-compose.yml              # ✅ Multi-service container (Gradio + Jupyter + API)
│   ├── entrypoint_app.sh               # ✅ Enhanced entrypoint with Gradio service
│   ├── requirements.txt                # Container dependencies
│   ├── test_docker_setup.sh            # Docker validation script
│   └── README.md                       # ✅ Week 7-8: Comprehensive Docker deployment guide
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # ✅ Complete EDA and data quality analysis
│   ├── 02_data_processing.ipynb        # ✅ Preprocessing pipeline implementation
│   ├── 03_modeling.ipynb               # ✅ Complete: Baseline models + Phase 3 enhanced models
│   ├── 04_error_analysis.ipynb         # ✅ Comprehensive error analysis framework
│   └── 05_explainability_tests.ipynb  # ✅ Week 5-6: Complete SHAP + LIME XAI implementation
├── reports/
│   ├── biweekly_meeting_2.md            # ✅ Week 3-4 Phase 3 implementation summary
│   ├── biweekly_meeting_3.md            # ✅ Week 5-6 XAI implementation summary
│   ├── biweekly_meeting_4.md            # ✅ Week 7-8 Gradio demo implementation summary
│   ├── project_plan_and_roadmap.md      # ✅ Updated: Complete project roadmap with Week 7-8 status
│   ├── literature_review.md             # ✅ Updated: Gradio and healthcare interface literature
│   └── final_report_draft.md            # ✅ Updated: Week 7-8 interactive demo documentation
├── results/
│   ├── error_analysis/                 # Error analysis outputs and edge case studies
│   ├── metrics/                        # Model evaluation results and performance summaries
│   ├── models/                         # Serialized models and preprocessing artifacts
│   │   ├── final_phase3_model.joblib    # ✅ Final enhanced XGBoost for Week 7-8 integration
│   │   └── phase3_class_imbalance_results.joblib  # Complete Phase 3 results
│   ├── plots/                          # Visualization outputs from analysis notebooks
│   └── xai_analysis/                   # ✅ Week 5-6: SHAP + LIME artifacts and CSV exports
│       ├── xai_artifacts.joblib         # XAI model artifacts for Gradio integration
│       └── *.csv                       # Research-accessible XAI analysis results
├── src/                                # Source code modules (future development)
├── .github/                            # GitHub workflows and templates
├── .venv/                              # Virtual environment (local development)
├── .vscode/                            # VS Code workspace settings
├── launch_demo.sh                      # ✅ Week 7-8: One-command demo launcher
├── test_week7_8.sh                     # ✅ Week 7-8: Comprehensive testing suite
├── requirements.txt                    # Python dependencies
├── LICENSE                             # Project license
└── README.md                           # Project overview and implementation status
```

---

## Getting Started

### 🚀 Quick Demo Launch (Week 7-8)
```bash
# Launch professional Gradio demo with one command
./launch_demo.sh

# Access interactive healthcare interface
open http://localhost:7860  # Professional Gradio Demo
open http://localhost:8888  # Jupyter Lab for analysis
```

### 🔬 Development Setup
```bash
# Manual Docker setup
cd docker/
docker-compose up --build

# Alternative: Gradio-only mode
docker-compose --profile gradio-only up

# Alternative: Jupyter-only mode  
docker-compose --profile jupyter-only up
```

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Petlaz/health_xai_prediction.git
cd health_xai_prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Reproduce Week 1-6 Results

```bash
# Start Jupyter to explore implemented notebooks
jupyter notebook

# Navigate through completed analysis:
# 1. notebooks/01_exploratory_analysis.ipynb    - Complete EDA
# 2. notebooks/02_data_processing.ipynb         - Data preprocessing 
# 3. notebooks/03_modeling.ipynb                - Baseline + enhanced models
# 4. notebooks/04_error_analysis.ipynb          - Comprehensive error analysis
# 5. notebooks/05_explainability_tests.ipynb   - Week 5-6 XAI implementation

# Alternative: Use Docker for containerized environment
docker-compose -f docker/docker-compose.yml up
# Access Jupyter Lab at http://localhost:8888
# XAI notebooks ready to run with all dependencies
```

### 3. Review Week 5-6 XAI Results

```bash
# XAI analysis results (accessible in Excel, R, Python)
open results/xai_analysis/global_feature_importance.csv    # Population-level predictors
open results/xai_analysis/healthcare_risk_factors.csv     # Clinical thresholds
open results/xai_analysis/case_studies_analysis.csv       # Individual patient analysis
open results/xai_analysis/README.md                       # XAI interpretation guide

# Project reports and findings
open reports/biweekly_meeting_3.md        # Week 5-6 XAI progress summary
open reports/final_report_draft.md        # Complete Week 1-6 technical report
open reports/literature_review.md         # XAI literature integration
```

---

## Future Development Opportunities

Based on Week 1-2 baseline findings, future optimization phases could explore:

### 🎯 Model Performance Enhancement
1. **Advanced class imbalance handling:** SMOTE, cost-sensitive learning, threshold optimization
2. **XGBoost optimization:** Focus hyperparameter tuning on best-performing model (49.3% accuracy)
3. **Feature engineering:** Interaction terms guided by validated feature importance patterns
4. **Ensemble strategies:** Leverage high XGBoost-Random Forest correlation (0.85)

### 📊 Technical Infrastructure
- **Source code modularization:** Develop production-ready `src/` modules
- **Automated evaluation:** Enhanced pipeline for systematic model comparison
- **Experiment tracking:** Version control for iterative model improvements

### 🔬 Advanced Analytics
- **Explainability integration:** SHAP/LIME frameworks for model interpretability
- **Clinical validation:** Healthcare domain expert review of feature importance
- **Uncertainty quantification:** Calibration improvement beyond current ECE=0.009

---

## Project Status Summary

### ✅ Week 7-8 Complete Implementation
- **Professional Gradio Demo:** Interactive healthcare prediction interface with clinical UI design
- **Real-time XAI Integration:** SHAP explanations with feature importance visualization
- **Clinical Risk Assessment:** Automated BMI, mental health, and lifestyle risk factor analysis  
- **Healthcare Insights:** Evidence-based recommendations and intervention suggestions
- **Dual Access Deployment:** Both local (localhost:7860) and public URL sharing capability
- **Complete Docker Integration:** One-command deployment with comprehensive documentation
- **Professional Interface:** Clean design avoiding technical jargon for healthcare professionals

### 🎯 Key Technical Achievements (Week 1-8)
- **Excellent model calibration:** ECE=0.009 meets healthcare deployment standards
- **XAI validation:** 32% SHAP-LIME agreement with clinical interpretation framework
- **Clinical insights:** BMI (37.0/100 reliability), Physical Effort, Sleep Quality as top predictors
- **Interactive explanations:** Real-time SHAP visualizations with healthcare context
- **Production-ready deployment:** Complete Docker containerization with Gradio + Jupyter Lab
- **Accessibility:** Both technical (joblib) and research (CSV) formats with professional UI

### 🚀 Week 9-10 Ready: Clinical Validation and Optimization
- **User Experience Testing:** Healthcare professional feedback collection and interface refinement
- **Performance Optimization:** Clinical workflow integration and response time improvement
- **Advanced XAI Visualizations:** Waterfall plots, interaction effects, and confidence intervals
- **Clinical Validation:** Healthcare domain expert review and validation of explanations
- **Production Deployment:** Secure authentication, monitoring, and clinical integration preparation

**Implementation Status:** Week 1-8 complete with professional interactive demo ready for clinical validation and user experience testing.

---

## Contact & Collaboration

**Student:** Peter Obi  
**Academic Supervisor:** Prof. Dr. Beate Rhein  
**Industry Partner:** Mr. Håkan Lane (Nightingale Heart)

**Project Phase:** Week 1-2 baseline implementation completed successfully with comprehensive documentation and reproducible analysis framework established.
