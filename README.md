# Health XAI Prediction

**Predictive Modeling and Local Explainable AI (XAI) in Healthcare**

This repository hosts a comprehensive MSc research project focused on predicting 5-class health status from European survey data using machine learning with explainable AI integration. The project follows a systematic biweekly development approach with complete **Week 1-4 implementation** covering baseline models, comprehensive error analysis, and advanced class imbalance solutions.

---

## Week 1-4 Achievements Summary

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

### 📊 Week 1-4 Complete Technical Results
| **Phase** | **Focus** | **Key Achievement** | **Final Model Performance** |
|-----------|-----------|---------------------|----------------------------|
| **Phase 1-2** | Baseline Implementation | XGBoost baseline established | 49.3% accuracy, 0.3641 F1-Macro |
| **Phase 3** | Class Imbalance Solutions | Enhanced models with cost-sensitive learning | 45.54% accuracy, 0.3620 F1-Macro (test) |
| **Status** | **Week 3-4 Complete** | **Ready for Week 5-6 XAI** | Enhanced XGBoost saved for LIME & SHAP |

### 🎯 Phase 3 Key Findings
- **Individual models outperform ensembles:** Enhanced XGBoost > Hard voting > Soft voting
- **Cost-sensitive learning effective:** 23.3x class weighting improved minority class detection
- **Dataset limitations identified:** Modest improvements despite major enhancements suggest missing clinical features
- **Performance appropriate:** 45.54% accuracy reasonable for healthcare 5-class prediction with 1:39.2 imbalance

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
│   ├── 03_modeling.ipynb              # ✅ Complete: Baseline models + Phase 3 enhanced models
│   └── 04_error_analysis.ipynb        # ✅ Comprehensive error analysis framework
├── reports/
│   ├── biweekly_meeting_2.md           # ✅ Week 3-4 Phase 3 implementation summary
│   ├── project_plan_and_roadmap.md     # Project overview and planning
│   ├── literature_review.md            # ✅ Updated with Phase 3 findings and ensemble analysis
│   └── final_report_draft.md           # ✅ Complete Week 1-4 implementation documentation
├── results/
│   ├── models/                         # ✅ Final enhanced XGBoost model ready for XAI
│   │   ├── final_phase3_model.joblib   # Final model for Week 5-6 XAI implementation
│   │   └── phase3_class_imbalance_results.joblib  # Complete Phase 3 results
│   ├── .keep                           # Git directory preservation
│   └── metrics/                        # Comprehensive evaluation results
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
