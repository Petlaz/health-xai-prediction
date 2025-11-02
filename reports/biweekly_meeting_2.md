# Biweekly Meeting 2 Summary
# Biweekly Meeting 2 Summary

**Project:** Prediction and Local Explainable AI (XAI) in Healthcare  
**Period:** Weeks 3–4 (03 Nov – 16 Nov)  
**Attendees:** Peter Obi, Prof. Dr. Beate Rhein, Mr. Håkan Lane

---

## 1. Focus
- Hyperparameter tuning for Logistic Regression, Random Forest, XGBoost, and the PyTorch NN.
- Early validation on hold-out data and documentation of persistent misclassification patterns.
- Literature review kick-off (“State of the Art”) and Docker environment updates reflecting new dependencies.

## 2. Key Updates
- _To be completed during the Week 3–4 review._

## 3. Artefacts
- `results/metrics/` (tuning logs, updated metrics summaries)
- `results/confusion_matrices/` & `results/plots/` (post-tuning diagnostics)
- Notes appended to `reports/literature_review.md`

## 4. Action Items (Before Meeting 3)
1. Finalise tuned hyperparameter configurations and record them in `src/train_models.py`/`README.md` notes.
2. Compare tuned models on the validation/test splits to shortlist the best performing candidate for XAI integration.
3. Expand the literature review with insights tied to observed error patterns.
4. Ensure Docker requirements are updated so colleagues can reproduce tuning runs.

---

## Suggested Visuals for Presentation
- Updated metric comparison chart (F1 / Precision / Recall) exported from `notebooks/03_modeling.ipynb`.
- Side-by-side confusion matrices (pre- vs. post-tuning) to illustrate improvements.
