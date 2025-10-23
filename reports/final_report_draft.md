# Final Report Draft

## 🧠 2. State of the Art

### 2.1 Introduction

Artificial Intelligence (AI) and Machine Learning (ML) have become essential in healthcare for predicting diseases and supporting clinical decisions.  
Heart disease remains one of the world’s leading causes of mortality, and early detection through predictive analytics can significantly improve patient outcomes.  
However, despite high predictive performance, many models lack transparency — making **Explainable AI (XAI)** increasingly crucial for clinical trust and adoption.

This section reviews prior research on heart disease prediction using ML techniques and the development of explainable AI methods applicable to healthcare data.

---

### 2.2 Predictive Machine Learning in Heart Disease Detection

Early studies on cardiovascular risk prediction primarily used classical ML algorithms such as **Logistic Regression**, **Support Vector Machines (SVM)**, and **Random Forests**, which provided reliable but limited interpretability.  

| **Study** | **Dataset** | **Methods** | **Key Results** | **Limitations** |
|:-----------|:-------------|:-------------|:----------------|:----------------|
| Tiwari et al. (2023) | UCI Heart Disease | XGBoost + SHAP | Achieved ~89% accuracy; SHAP identified key features such as cholesterol and age | Dataset small; limited generalizability |
| Zhang et al. (2022) | Framingham Heart Study | Logistic Regression, RF, SVM | Random Forest achieved best F1 ≈ 0.84 | No interpretability provided |
| Alharbi et al. (2024) | NHS Health Survey | XGBoost + LIME | Improved clinician understanding through local explanations | Slight trade-off in accuracy |

Recent work integrates ensemble methods (Random Forest, XGBoost) and deep neural networks (DNNs) to enhance predictive performance, though interpretability remains limited.

---

### 2.3 Explainable AI (XAI) in Healthcare

With increasing emphasis on trustworthy AI, explainability has become central to healthcare ML systems.  
**Model-agnostic** methods such as **LIME** (Ribeiro et al., 2016) and **SHAP** (Lundberg & Lee, 2017) are widely used to generate local and global feature explanations.

| **Technique** | **Description** | **Strengths** | **Limitations** |
|:----------------|:----------------|:----------------|:----------------|
| **LIME** | Creates local linear approximations around individual predictions | Model-agnostic; intuitive visual explanations | Sensitive to sampling; unstable for complex models |
| **SHAP** | Based on Shapley values from cooperative game theory | Consistent additive explanations; global + local views | Computationally intensive for large datasets |
| **Counterfactual Explanations** | Generates “what-if” scenarios to show changes leading to different outcomes | Useful for causal reasoning | Requires well-calibrated models |
| **Rule-based Models (e.g., Decision Sets)** | Use transparent if-then rules | Intuitive for clinicians | Limited scalability for high-dimensional data |

Several recent studies (Holzinger et al., 2019; Caruana et al., 2015) emphasize that **clinical acceptance** depends on both model accuracy and interpretability.

---

### 2.4 Integrating Predictive ML with Local Explainability

Modern healthcare AI research increasingly combines predictive power with interpretability:

- **XGBoost + SHAP** and **Random Forest + LIME** frameworks dominate structured health data tasks.
- Neural network models are now interpreted using **DeepSHAP** and **Integrated Gradients**.
- Hybrid systems combine **rule-based reasoning** and **black-box models** to balance accuracy and transparency.

However, few studies have applied **local XAI methods to large survey-based datasets** such as the European Social Survey (ESS), which include both lifestyle and demographic attributes.

---

### 2.5 Identified Research Gap

From the literature, the following limitations remain evident:

1. **Dataset limitation** — Most studies use small or region-specific clinical datasets.  
2. **Lack of local interpretability** — Explanations often remain at a global feature level.  
3. **Limited generalization** — Few works combine structured survey data with robust XAI frameworks.  
4. **Absence of interactive tools** — Few research outputs provide user-facing explainable interfaces for healthcare professionals.

---

### 2.6 Contribution of This Project

This research aims to bridge these gaps by:

1. Developing a **predictive model** for heart disease risk using large structured survey data (ESS, ~40,000 records).  
2. Applying **Local XAI techniques (LIME, SHAP)** to generate patient-level interpretability.  
3. Building an **interactive Gradio demo** for visualization and clinician-friendly explanations.  
4. Validating the explainability-performance trade-off using early model iterations and quantitative metrics (accuracy, F1, ROC-AUC).

---

### 2.7 Summary

The reviewed literature confirms the growing importance of interpretability in medical AI.  
While ensemble and deep models yield strong predictive results, their lack of transparency limits real-world use.  
By combining optimized models with local explainability techniques, this project advances **trustworthy and interpretable AI** for healthcare.

---

### References

*(Summarized from literature_review.md; expand later as citations are finalized)*  

1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *“Why Should I Trust You?” Explaining the Predictions of Any Classifier.*  
2. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).*  
3. Holzinger, A., et al. (2019). *What Do We Need to Build Explainable AI Systems for Health?*  
4. Caruana, R., et al. (2015). *Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital Readmission.*  
5. Tiwari, A., et al. (2023). *Heart Disease Prediction Using XGBoost and SHAP Analysis.*  
6. Zhang, L., et al. (2022). *Comparative Study of ML Techniques for Cardiovascular Disease Prediction.*  
7. Alharbi, S., et al. (2024). *Local Explainability in Heart Risk Models.*  

---

✅ **Usage Tip:**  

- Keep `literature_review.md` as your **working notes** file.  
- Once you complete Weeks 3–6, summarize its content into this **State of the Art** section for the final report.  
- Add in-text citations and proper formatting (APA or IEEE) later.

---
