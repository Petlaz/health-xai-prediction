# Final Report Draft

## 1. Introduction
Cardiovascular disease remains a leading cause of mortality across Europe, yet the wealth of survey-based health and lifestyle information collected annually is rarely transformed into actionable, personalised risk insights. This project investigates how classical machine learning models and lightweight neural networks can be paired with Local Explainable AI (XAI) methods to deliver transparent risk assessments for individuals participating in a European health survey (~40,000 records). By coupling predictive performance with local explanations, the work aims to support clinicians and public-health analysts who need to understand why a model flags a respondent as high risk and what behaviour changes could alter that prediction.

Week 1–2 focused on establishing the technical foundations for the study: curating the dataset, building reproducible preprocessing routines, exploring data quality issues, and training baseline models. The remainder of the project will iterate on these artefacts through model optimisation, explainability integration, and user-facing deployment via Gradio and Docker.

---

## 2. Methods

### 2.1 Dataset Preparation
- **Source:** European Health Survey CSV supplied by Nightingale Heart (≈42k rows, 52 engineered features after preprocessing).
- **Targets:** Primary – `hltprhc` (heart condition, binary); secondary – `hltprhb` (high blood pressure) and `hltprdi` (diabetes) for future experiments.
- **Cleaning steps:** Removed unnamed index column, standardised headers, and generated a feature mapping (`data/processed/feature_names.csv`) alongside an auto-built data dictionary (`data/data_dictionary.md`).
- **Missingness:** Overall rate 0.25% (raw). Numeric attributes imputed with the median; categorical attributes imputed with the most frequent value to retain valid categories.
- **Feature scaling/encoding:** Numeric features scaled via `StandardScaler`; categorical features one-hot encoded using `OneHotEncoder(handle_unknown="ignore")`. Processed splits (train/validation/test = 70/15/15, stratified) and the combined dataset (`health_clean.csv`) are saved to `data/processed/`.

### 2.2 Exploratory Data Analysis
- `notebooks/01_exploratory_analysis.ipynb` documents summary statistics, class balance (positive class ≈11.32%), correlation heatmaps, VIF, and IQR-based outlier detection.
- Outputs (CSV summaries, plots) are stored under `results/metrics/` and `results/plots/` for reproducibility and to track data quality changes as features evolve.

### 2.3 Baseline Modelling Pipeline
- Implemented in `src/train_models.py` with shared utilities (`src/utils.py`).
- Models: Logistic Regression (class-balanced), Random Forest, XGBoost, and a two-hidden-layer PyTorch feed-forward network.
- Training artefacts (joblib models, scaler, neural network weights, cached splits) persist to `results/models/` for downstream evaluation and tuning.

### 2.4 Evaluation & Error Analysis
- `src/evaluate_models.py` generates accuracy, precision, recall, F1, ROC-AUC; confusion matrices; ROC and precision–recall curves; and scikit-learn classification reports saved to `results/metrics/`.
- Misclassified samples with model predictions and error tags are exported as `results/metrics/misclassified_samples.csv` to seed deeper Week 3–4 analysis.

### 2.5 Reproducibility & Tooling
- Environment dependencies tracked in `requirements.txt`.
- Docker image (`docker/Dockerfile`) provisions Python 3.11, system build tools, and installs project requirements for consistent execution across machines.
- Experiment notebooks (`notebooks/02_data_processing_experiments.ipynb`, `notebooks/03_modeling_experiments.ipynb`) serve as scratchpads before formalising changes in the `src/` modules.

---

## 🧠 3. State of the Art

### 3.1 Introduction

Artificial Intelligence (AI) and Machine Learning (ML) have become essential in healthcare for predicting diseases and supporting clinical decisions.  
Heart disease remains one of the world’s leading causes of mortality, and early detection through predictive analytics can significantly improve patient outcomes.  
However, despite high predictive performance, many models lack transparency — making **Explainable AI (XAI)** increasingly crucial for clinical trust and adoption.

This section reviews prior research on heart disease prediction using ML techniques and the development of explainable AI methods applicable to healthcare data.

---

### 3.2 Predictive Machine Learning in Heart Disease Detection

Early studies on cardiovascular risk prediction primarily used classical ML algorithms such as **Logistic Regression**, **Support Vector Machines (SVM)**, and **Random Forests**, which provided reliable but limited interpretability.  

| **Study** | **Dataset** | **Methods** | **Key Results** | **Limitations** |
|:-----------|:-------------|:-------------|:----------------|:----------------|
| Tiwari et al. (2023) | UCI Heart Disease | XGBoost + SHAP | Achieved ~89% accuracy; SHAP identified key features such as cholesterol and age | Dataset small; limited generalizability |
| Zhang et al. (2022) | Framingham Heart Study | Logistic Regression, RF, SVM | Random Forest achieved best F1 ≈ 0.84 | No interpretability provided |
| Alharbi et al. (2024) | NHS Health Survey | XGBoost + LIME | Improved clinician understanding through local explanations | Slight trade-off in accuracy |

Recent work integrates ensemble methods (Random Forest, XGBoost) and deep neural networks (DNNs) to enhance predictive performance, though interpretability remains limited.

---

### 3.3 Explainable AI (XAI) in Healthcare

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
