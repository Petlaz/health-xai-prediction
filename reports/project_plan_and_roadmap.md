# Updated Research Project Plan

**Title:** Prediction and Local Explainable AI (XAI) in Healthcare  
**Duration:** October 2025 – January 2026  
**Supervisor:** Prof. Dr. Beate Rhein  
**Industry Partner:** Nightingale Heart – Mr. Håkan Lane  
**Hardware Platform:** Mac M1/M2 (Apple Silicon) - optimized for efficient ML computations  

## Project Goal

The goal is to integrate **Local Explainable AI (XAI)** techniques — specifically **LIME** and **SHAP** — to interpret model decisions at the individual level.

A **Gradio interface** will provide real-time interactive predictions and explanations.  
The entire workflow will be **containerized using Docker** for reproducibility and future deployment.

## Dataset Overview

Structured CSV dataset with health, demographic, and lifestyle variables.

• **Target variable:** `health` (5-class ordinal health rating: 1=Very Good, 2=Good, 3=Fair, 4=Bad, 5=Very Bad)  
• **Alternative targets:** `hltprhc` (heart condition), `hltprhb` (blood pressure), `hltprdi` (diabetes)

## Research Objectives

1. **Develop and compare predictive models:** Logistic Regression, Random Forest, XGBoost, SVM, and PyTorch Neural Network for 5-class health prediction.
2. **Perform early error analysis** (accuracy, precision, recall, F1-macro/weighted, confusion matrix, and misclassified samples for 5-class classification).
3. **Conduct model optimization and iterative validation** on unseen data after tuning.
4. **Apply Local Explainability** (LIME and SHAP) for individual-level interpretation.
5. **Conduct a literature review** ("State of the Art") informed by model errors.
6. **Write report sections** (Methods, Results, Discussion) in parallel with experiments.
7. **Build a Gradio demo** for interpretable healthcare prediction.
8. **Containerize all experiments** using Docker for reproducibility.

## 🧩 3-Month Research Project Roadmap

*(Biweekly meetings – 6 total, ~20 hrs/week)*

### Weeks 1–2 (Oct 20 – Nov 2): Data Understanding, Baseline Modeling & Error Analysis ✅ COMPLETE

• Load and explore the dataset. ✅  
• Conduct Full EDA. ✅  
• Data preprocessing and feature engineering. ✅  
• Train baseline models using Logistic Regression, Random Forest, XGBoost, SVM, and Neural Network with PyTorch (using AdamW with patience set to 10). ✅  
• Evaluate with accuracy, precision, recall, F1, ROC curve, classification report, and confusion matrix ✅  
• Perform misclassified samples ✅  
• Perform full error analysis ✅  
• Initialize the GitHub repository, create a requirements.txt file, and create a Dockerfile. ✅  
• Begin writing the Introduction and Methods sections. ✅

**Deliverables:** Clean dataset + baseline results + error plots + Docker setup ✅  

**Key Results:**
- XGBoost achieved best performance: 49.3% accuracy, 0.3641 F1-Macro
- Comprehensive error analysis revealed 1:39.2 class imbalance
- All baseline models established with ECE=0.009 excellent calibration  
**Reading:** Interpretable ML Ch. 2–3 · Hands-On ML Ch. 2–4 · Designing ML Systems Ch. 2

### Weeks 3–4 (Nov 3 – Nov 16): Class Imbalance Solutions & Enhanced Model Development ✅ COMPLETE

**Context:** Week 1-2 established baseline performance with XGBoost leading at 49.3% accuracy. Phase 3 addressed underfitting and severe 1:39.2 class imbalance through enhanced model architecture and cost-sensitive learning.

**Phase 3: Enhanced Model Development & Class Imbalance Solutions ✅**

**Enhanced Model Architecture:**
• **Enhanced XGBoost:** 500 trees, depth=8, reduced regularization (reg_alpha=0.01, reg_lambda=0.01) ✅  
• **Enhanced Random Forest:** 300 trees, depth=20, reduced min_samples constraints ✅  
• **Underfitting resolution:** Increased complexity parameters to address Phase 2 UNDERFITTING status ✅  

**Cost-Sensitive Learning Implementation:**
• **Class weight computation:** Balanced class weights with 23.3x emphasis on Very Bad health class ✅  
• **Sample weight application:** Dynamic weighting during training to address 1:39.2 imbalance ✅  
• **Training integration:** Seamless incorporation into XGBoost and Random Forest frameworks ✅  

**Advanced Ensemble Strategy Evaluation:**
• **Hard Voting Ensemble:** Simple majority voting between enhanced models ✅  
• **Soft Voting Ensemble:** Performance-weighted probability averaging ✅  
• **Individual vs Ensemble Analysis:** Enhanced XGBoost outperformed all ensemble approaches ✅  
• **Threshold Optimization:** Fine-tuned decision boundaries for minority class detection ✅  

**Final Model Selection & Evaluation:**
• **Unbiased test evaluation:** Single test set evaluation preserving statistical validity ✅  
• **Selected Model:** Optimized Enhanced XGBoost (Test F1-Macro: 0.3620, Accuracy: 45.54%) ✅  
• **Model artifacts:** Final model saved for Week 5-6 XAI implementation ✅  

**Key Achievements:**
- Successfully addressed underfitting through enhanced model complexity
- Implemented cost-sensitive learning for severe class imbalance (1:39.2 ratio)
- Demonstrated individual enhanced models outperform ensemble approaches
- Final model ready for XAI analysis with validated performance benchmarks  
• Develop ensemble strategies if multiple models show similar validation performance  
• **Final test set evaluation:** Apply selected model(s) to test set only once for unbiased performance estimate  
• Compare final test performance against Week 1-2 baseline results (XGBoost 49.3%)  

**Technical Infrastructure:**
• Implement automated hyperparameter tuning pipeline optimized for Mac M1/M2 architecture  
• Create comprehensive evaluation framework comparing all tuned models  
• Develop visualization suite for hyperparameter sensitivity analysis  
• Update model serialization to preserve best hyperparameters and class weights  
• Enhanced metrics tracking across tuning iterations with computational efficiency monitoring  

**Deliverables:** Fully tuned model suite + best model identification + class-balanced optimization + comprehensive performance comparison  
**Reading:** Hyperparameter Optimization Techniques · Learning from Imbalanced Datasets · Model Selection Strategies

### Weeks 5–6 (Nov 17 – Dec 1): Local Explainability Integration (XAI)

• Implement LIME and SHAP for selected model.  
• Generate SHAP summary, force plots, and LIME explanations.  
• Compare local explanations across models.  
• Interpret healthcare-related insights from local explanations.  
• Ensure XAI modules run inside Docker.  
• Continue writing State of the Art and Results sections.

**Deliverables:** XAI visualizations + interpretability report + Dockerized XAI workflow  
**Reading:** Interpretable ML Ch. 4–6 · Hands-On ML Ch. 11 · Designing ML Systems Ch. 8

### Weeks 7–8 (Dec 2 – Dec 15): Gradio Demo Development & Report Progress

• Build an interactive Gradio app (real-time predictions + explanations).    
• Test usability, latency, and visual clarity.  
• Containerize demo (EXPOSE 7860) and test locally.  
• Continue report writing (Results + Discussion).

**Deliverables:** Functional Gradio demo (classical + NN models) + Meeting 4 summary  
**Reading:** Hands-On ML Ch. 19 · Designing ML Systems Ch. 4

### Weeks 9–10 (Dec 16 – Jan 1): Evaluation, Refinement & Discussion

• Evaluate final model on validation and test sets.  
• Assess stability and consistency of local explanations.  
• Refine XAI visuals and final discussion.  
• Update Docker image with final model.  
• Finalize Discussion and State of the Art sections.

**Deliverables:** Evaluation results + refined XAI visuals + updated demo + Meeting 5 summary  
**Reading:** Interpretable ML Ch. 7 · Designing ML Systems Ch. 9

### Weeks 11–12 (Jan 2 – Jan 15): Final Report & Defense Preparation

• Finalize Gradio demo and Docker image.  
• Write final report (Introduction, State of the Art, Methods, Results, Discussion, Conclusion).  
• Prepare presentation slides and defense.  
• Submit report + Docker package to Professor and Nightingale Heart.

**Deliverables:** Final report + Gradio demo + Docker image + Meeting 6 summary  
**Reading:** Hands-On ML Appendix · Designing ML Systems Ch. 10

## 📅 Summary of Biweekly Meetings

| Meeting | Week | Focus | Key Deliverable |
|---------|------|-------|----------------|
| 1 | 2 | EDA + Baseline + Error Analysis | Clean dataset + baseline models + comprehensive error analysis |
| 2 | 4 | Class Imbalance Solutions + Model Enhancement | Cost-sensitive models + threshold optimization + ensemble strategies |
| 3 | 6 | Local XAI Integration | LIME/SHAP visualizations + interpretation |
| 4 | 8 | Gradio Demo | Interactive demo (Dockerized) |
| 5 | 10 | Evaluation + Refinement | Final metrics + discussion draft |
| 6 | 12 | Final Presentation | Report + Gradio demo + Docker image |
