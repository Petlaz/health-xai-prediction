# Biweekly Meeting 2 Summary
# Biweekly Meeting 2 Summary

**Project:** Prediction and Local Explainable AI (XAI) in Healthcare  
**Period:** Weeks 3–4 (03 Nov – 16 Nov)  
**Attendees:** Peter Obi, Prof. Dr. Beate Rhein, Mr. Håkan Lane

---

## 1. Focus: Class Imbalance Solutions & Enhanced Model Development
- **Phase 3 Implementation:** Advanced class imbalance handling with cost-sensitive learning
- **Enhanced Model Development:** Addressing underfitting through increased model complexity
- **Ensemble Strategy Testing:** Hard/soft voting with optimized enhanced models
- **Threshold Optimization:** Fine-tuning decision boundaries for minority class detection

## 2. Key Updates

### ✅ Phase 3: Class Imbalance Solutions Completed
- **Enhanced XGBoost:** 500 trees, depth=8, reduced regularization, cost-sensitive weights
- **Enhanced Random Forest:** 300 trees, depth=20, balanced class weights
- **Class Imbalance Ratio:** Successfully addressed 1:39.2 Very Bad health class imbalance
- **Cost-Sensitive Learning:** Implemented 23.3x weighting for minority class

### 🔍 Underfitting Resolution Achieved
- **Problem Identification:** Phase 2 models showed UNDERFITTING status
- **Solution Implementation:** Increased model complexity parameters
- **Validation Results:** XGBoost F1-Macro improvement +0.0171 (0.3641 → 0.3812)
- **Architecture Enhancement:** Reduced regularization, increased depth and estimators
