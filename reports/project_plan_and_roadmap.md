# Updated Research Project Plan

**Title:** Prediction and Local Explainable AI (XAI) in Healthcare  
**Duration:** October 2025 – January 2026  
**Supervisor:** Prof. Dr. Beate Rhein  
**Industry Partner:** Nightingale Heart – Mr. Håkan Lane  

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

### Weeks 1–2 (Oct 20 – Nov 2): Data Understanding, Baseline Modeling & Error Analysis

• Load and explore the dataset.  
• Conduct Full EDA.  
• Data preprocessing and feature engineering.  
• Train baseline models using Logistic Regression, Random Forest, XGBoost, SVM, and Neural Network with PyTorch (using AdamW with patience set to 10).  
• Evaluate with accuracy, precision, recall, F1, ROC curve, classification report, and confusion matrix  
• Perform misclassified samples  
• Perform full error analysis  
• Initialize the GitHub repository, create a requirements.txt file, and create a Dockerfile.  
• Begin writing the Introduction and Methods sections.

**Deliverables:** Clean dataset + baseline results + error plots + Docker setup  
**Reading:** Interpretable ML Ch. 2–3 · Hands-On ML Ch. 2–4 · Designing ML Systems Ch. 2

### Weeks 3–4 (Nov 3 – Nov 16): Model Optimization, Early Validation & Literature Review

• Tune hyperparameters (RandomizedSearchCV).  
• Validate optimized models on unseen data (early performance check).  
• Analyze misclassifications and document patterns.  
• Begin literature review ("State of the Art") informed by error findings.  
• Update Docker setup for reproducible experiments.  
• Continue writing the Methods section.

**Deliverables:** Optimized models + validation results + error summary + initial paper notes  
**Reading:** Interpretable ML Ch. 5 · Hands-On ML Ch. 6–8 · Designing ML Systems Ch. 3

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
• Integrate classical and NN models for comparison.  
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
| 1 | 2 | EDA + Baseline + Error Analysis | Clean dataset + metrics + confusion matrix |
| 2 | 4 | Model Optimization + Early Validation | Optimized models + validation results + literature insights |
| 3 | 6 | Local XAI Integration | LIME/SHAP visualizations + interpretation |
| 4 | 8 | Gradio Demo | Interactive demo (Dockerized) |
| 5 | 10 | Evaluation + Refinement | Final metrics + discussion draft |
| 6 | 12 | Final Presentation | Report + Gradio demo + Docker image |
