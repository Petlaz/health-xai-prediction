# Health XAI Prediction

**Predictive Modeling and Local Explainable AI (XAI) in Healthcare**

This repository hosts a comprehensive MSc research project focused on predicting 5-class health status from European survey data using machine learning with planned explainable AI integration. The project follows a systematic biweekly development approach. This README summarizes accomplishments through **Week 1-2**, covering complete baseline implementation and comprehensive error analysis.

---

## Week 1-2 Achievements Summary

### ✅ Complete Baseline Implementation  
- **Robust data preprocessing pipeline:** 11,322 European Health Survey records with 22 numerical features
- **Multi-algorithm baseline models:** XGBoost (49.3% accuracy), Random Forest (47.6%), SVM (42.3%), Logistic Regression (36.8%)
- **Comprehensive evaluation framework:** Complete metrics calculation with calibration analysis
- **Production-ready artifacts:** Serialized models, scalers, and data splits for reproducible experiments

### ✅ Comprehensive Error Analysis Complete
- **Detailed misclassification study:** 3,218 misclassified samples analyzed across all models
- **Class imbalance assessment:** Severe 1:39.2 ratio identified requiring specialized handling
- **Model calibration validation:** Excellent Expected Calibration Error (0.009) achieved
- **Edge case identification:** 87.4% of samples flagged as requiring robust model handling

### 📊 Week 1-2 Technical Results
| **Component** | **Status** | **Key Metrics** | **Artifacts Generated** |
|---------------|------------|-----------------|------------------------|
| **Data Pipeline** | ✅ Complete | 11,322 records, 22 features | Clean datasets, splits, feature mapping |
| **Baseline Models** | ✅ Complete | XGBoost: 49.3% accuracy (best) | 4 trained models with evaluation reports |
| **Error Analysis** | ✅ Complete | ECE: 0.009 (excellent calibration) | 10-section comprehensive analysis |
| **Feature Analysis** | ✅ Complete | Self-rated health dominates | Feature importance rankings and correlations |

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
├── data/
│   ├── raw/                     # Original survey datasets
│   ├── processed/               # Clean splits and preprocessing artifacts
│   └── data_dictionary.md       # Feature documentation
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # ✅ Complete EDA and data quality analysis
│   ├── 02_data_processing.ipynb       # ✅ Preprocessing pipeline implementation
│   ├── 03_modeling.ipynb              # ✅ Baseline model training and evaluation
│   └── 04_error_analysis.ipynb        # ✅ Comprehensive 10-section error analysis
├── reports/
│   ├── biweekly_meeting_1.md           # ✅ Week 1-2 progress documentation
│   ├── project_plan_and_roadmap.md     # Project overview and planning
│   ├── literature_review.md            # ✅ Foundational literature supporting Week 1-2
│   └── final_report_draft.md           # ✅ Complete Week 1-2 implementation report
├── results/
│   ├── metrics/                 # Model evaluation results and performance summaries
│   ├── models/                  # Serialized baseline models and preprocessing artifacts
│   └── plots/                   # Visualization outputs from analysis notebooks
├── src/                         # Source code modules (future development)
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and Week 1-2 achievements
```

---

## Getting Started

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

### 2. Reproduce Week 1-2 Results

```bash
# Start Jupyter to explore implemented notebooks
jupyter notebook

# Navigate through completed analysis:
# 1. notebooks/01_exploratory_analysis.ipynb    - Complete EDA
# 2. notebooks/02_data_processing.ipynb         - Data preprocessing 
# 3. notebooks/03_modeling.ipynb                - Baseline models
# 4. notebooks/04_error_analysis.ipynb          - Comprehensive error analysis
```

### 3. Review Week 1-2 Documentation

```bash
# Project reports and findings
open reports/final_report_draft.md        # Complete technical report
open reports/biweekly_meeting_1.md        # Week 1-2 progress summary
open reports/literature_review.md         # Supporting literature
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

### ✅ Week 1-2 Baseline Implementation Complete
- **Comprehensive data analysis:** European Health Survey with 11,322 records processed
- **Multi-algorithm baseline:** 4 models trained with XGBoost achieving best performance (49.3%)
- **Detailed error analysis:** 10-section framework identifying key optimization opportunities
- **Complete documentation:** Technical reports, literature review, and meeting summaries
- **Reproducible pipeline:** All analysis notebooks executed with serialized model artifacts

### 🎯 Key Technical Achievements
- **Excellent model calibration:** ECE=0.009 meets healthcare deployment standards
- **Class imbalance quantified:** 1:39.2 ratio requiring specialized optimization techniques
- **Feature validation:** Self-rated health confirmed as dominant predictor across all models
- **High model correlation:** XGBoost-Random Forest (0.85) suggests strong ensemble potential

**Foundation Status:** Robust baseline established with comprehensive analysis supporting future optimization phases.

---

## Contact & Collaboration

**Student:** Peter Obi  
**Academic Supervisor:** Prof. Dr. Beate Rhein  
**Industry Partner:** Mr. Håkan Lane (Nightingale Heart)

**Project Phase:** Week 1-2 baseline implementation completed successfully with comprehensive documentation and reproducible analysis framework established.
