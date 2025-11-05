# Research Project Plan & Roadmap

**Title:** Prediction and Local Explainable AI (XAI) in Healthcare  
**Duration:** October 2025 – January 2026  
**Supervisor:** Prof. Dr. Beate Rhein  
**Industry Partner:** Nightingale Heart — Mr. Håkan Lane

---

## Project Overview

The project targets predictive modeling of heart-related risks using ~40,000 survey responses from a European health and lifestyle study. Logistic Regression, Random Forest, XGBoost, and a PyTorch-based neural network establish the modelling baseline; the best-performing tuned model will ultimately power the user-facing experience. Local explainability will be delivered through LIME and SHAP, with results surfaced via a Gradio web interface. Docker will containerize the full workflow for reproducibility and deployment.

### Dataset Highlights

- Structured CSV with demographic, lifestyle, and wellness variables.
- **Primary target:** `hltprhc` (heart condition: 1 = yes, 0 = no)
- **Secondary targets:** `hltprhb` (blood pressure), `hltprdi` (diabetes)

---

## Research Objectives

1. Build and compare baseline predictive models (Logistic Regression, Random Forest, XGBoost, PyTorch NN).
2. Perform early error analysis (accuracy, precision, recall, confusion matrices, misclassified samples).
3. Tune models and validate performance on unseen data.
4. Apply LIME and SHAP for individual-level explanations.
5. Conduct literature review informed by error analysis and model behaviour.
6. Draft Methods, Results, and Discussion sections in parallel with experiments.
7. Develop a Gradio demo for interactive, interpretable predictions.
8. Containerize the full pipeline using Docker.

---

## 3-Month Roadmap (Biweekly Sprints, ~20 hrs/week)

### Weeks 1–2 (Oct 20 – Nov 2): Data Understanding, Baseline Modeling & Error Analysis

- Load and explore dataset (~40k records); perform EDA (distributions, missing values, correlations, target balance).
- Preprocess: normalization, encoding, imputation.
- Train baseline models (Logistic Regression, Random Forest, XGBoost) and a simple PyTorch NN.
- Evaluate with accuracy, precision, recall, F1, ROC-AUC; generate confusion matrices and misclassification plots.
- Initialize GitHub repository, requirements.txt, Dockerfile; draft Introduction & Methods sections.
- **Deliverables:** Clean dataset, baseline metrics, error plots, initial Docker setup.
- **Reading:** *Interpretable ML* Ch. 2–3 · *Hands-On ML* Ch. 2–4 · *Designing ML Systems* Ch. 2

### Weeks 3–4 (Nov 3 – Nov 16): Model Optimization, Early Validation & Literature Review

- Tune hyperparameters (grid/random search); refine NN architecture (depth, learning rate).
- Validate optimised models on unseen data; document misclassification patterns.
- Capture recall-first diagnostics (RandomizedSearchCV with class weighting, recall scoring, early-stopping) and save out `model_diagnostics.csv` + `best_model.*` snapshots for reference.
- Begin literature review (“State of the Art”) guided by error insights; update Docker environment.
- Continue Methods section.
- **Deliverables:** Tuned models, validation metrics, error summary, literature notes.
- **Reading:** *Interpretable ML* Ch. 5 · *Hands-On ML* Ch. 6–8 · *Designing ML Systems* Ch. 3

### Weeks 5–6 (Nov 17 – Dec 1): Local Explainability Integration (XAI)

- Implement LIME and SHAP; generate SHAP summary/force plots and LIME explanations for the leading models from tuning.
- Compare local explanations across candidates, interpret healthcare implications, and shortlist the model that balances recall and usability for the demo.
- Ensure XAI workflows run inside Docker; advance State of the Art & Results sections.
- Begin threshold calibration experiments for the recall-first neural network so its lower precision relative to tree/boosting models is addressed before demo deployment.
- **Deliverables:** XAI visualisations, interpretability report, Dockerised XAI pipeline.
- **Reading:** *Interpretable ML* Ch. 4–6 · *Hands-On ML* Ch. 11 · *Designing ML Systems* Ch. 8

### Weeks 7–8 (Dec 2 – Dec 15): Gradio Demo Development & Report Progress

- Build Gradio app for real-time predictions + explanations; integrate the tuned, best-performing model selected in earlier sprints.
- Test usability, latency, and explanation clarity, and include clinician-facing recommendations alongside LIME/SHAP outputs; containerize demo (EXPOSE 7860).
- Continue Results & Discussion writing.
- **Deliverables:** Functional Gradio demo (Dockerised) + Meeting 4 summary.
- **Reading:** *Hands-On ML* Ch. 19 · *Designing ML Systems* Ch. 4

### Weeks 9–10 (Dec 16 – Jan 1): Evaluation, Refinement & Discussion

- Final evaluation on validation/test sets; assess stability of local explanations.
- Refine XAI visuals and discussion; update Docker image with final models.
- Finalise Discussion and State of the Art sections.
- **Deliverables:** Final metrics, refined XAI artefacts, updated demo, Meeting 5 summary.
- **Reading:** *Interpretable ML* Ch. 7 · *Designing ML Systems* Ch. 9

### Weeks 11–12 (Jan 2 – Jan 15): Final Report & Defense Preparation

- Finalise Gradio demo and Docker image.
- Complete full academic report (Introduction, State of the Art, Methods, Results, Discussion, Conclusion).
- Prepare presentation slides and defense materials; submit final package to supervisor and Nightingale Heart.
- **Deliverables:** Final report (PDF + repo), Gradio demo, Docker image, Meeting 6 summary.
- **Reading:** *Hands-On ML* Appendix · *Designing ML Systems* Ch. 10

---

## Biweekly Meeting Overview

| Meeting | Week | Focus | Key Deliverable |
|---------|------|-------|-----------------|
| 1 | 2 | EDA + Baseline + Error Analysis | Clean dataset, baseline metrics, confusion matrix |
| 2 | 4 | Model Optimisation + Early Validation | Tuned models, validation results, literature insights |
| 3 | 6 | Local XAI Integration | LIME/SHAP visualisations + interpretation notes |
| 4 | 8 | Gradio Demo | Interactive (Dockerised) demo |
| 5 | 10 | Evaluation + Refinement | Final metrics, discussion draft |
| 6 | 12 | Final Presentation | Report, Gradio demo, Docker image |

---

Keep this document as the single reference point for project scope, timelines, and deliverables. Update it in later weeks if milestones shift or new goals emerge.***
