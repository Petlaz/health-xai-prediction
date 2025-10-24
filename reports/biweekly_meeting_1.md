# Biweekly Meeting 1 Summary

**Project:** Prediction and Local Explainable AI (XAI) in Healthcare  
**Period:** Weeks 1–2 (20 Oct – 02 Nov)  
**Attendees:** Peter Obi, Prof. Dr. Beate Rhein, Mr. Håkan Lane  

---

## 1. Data Preparation & EDA Highlights

- Ingested ~42k records from the European Health Survey and cleaned feature names/documentation (`data/processed/feature_names.csv`, `data/data_dictionary.md`).
- Delivered a reproducible EDA notebook (`notebooks/01_exploratory_analysis.ipynb`) covering missingness (overall **0.25%**), class balance (hltprhc = 1 → **11.32%**), correlation heatmaps, VIF, and outlier detection with exported artefacts in `results/metrics/` and `results/plots/`.
- Confirmed feature distribution split: **numeric = 23**, **categorical = 29**, guiding the choice of median imputation + one-hot encoding for the production pipeline.

## 2. Baseline Modeling Status

- Implemented `src/train_models.py` to train Logistic Regression, Random Forest, XGBoost, and a 2-layer PyTorch neural network on the processed dataset with stratified 70/15/15 splits.
- Built `src/evaluate_models.py` to produce accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC/PR curves, classification reports, and misclassification CSVs.
- **Test-set snapshot (`results/metrics/metrics_summary.csv`):** Logistic Regression delivers Accuracy ≈0.755, Recall ≈0.72, F1 ≈0.40 (best at capturing positives), while Random Forest / XGBoost / NN reach Accuracy ≈0.89 but Recall falls to 0.15–0.22, indicating majority-class bias.
- Misclassification analysis shows Logistic Regression produces the majority of false positives—even at high predicted probabilities—highlighting priority cases for Week 3–4 error analysis.

## 3. Artefact Inventory (Week 1–2)

- Processed datasets: `data/processed/{train,validation,test,health_clean}.csv`
- EDA outputs: `results/metrics/eda_summary.csv`, `results/plots/*`
- Model artefacts: `results/models/{standard_scaler.joblib, logistic_regression.joblib, random_forest.joblib, xgboost_classifier.joblib, neural_network.pt, data_splits.joblib}`
- Evaluation reports: `results/metrics/metrics_summary.csv`, `results/metrics/classification_reports/*.csv`, `results/confusion_matrices/*.png`, `results/plots/*_roc_curve.png`, `results/plots/*_precision_recall_curve.png`, `results/metrics/misclassified_samples.csv`

## 4. Discussion Points & Decisions

- Maintain median imputation + standard scaling for numeric features and most-frequent imputation for categoricals; revisit after tuning if recall remains depressed.
- Prioritise recall improvements in Week 3–4 via class-weight tuning, threshold calibration, and potential resampling (SMOTE/undersampling) experiments.
- Use newly generated classification reports (`results/metrics/classification_reports/`) to monitor macro F1 and per-class support when testing tuning hypotheses.
- Document findings in `reports/biweekly_meeting_1.md` and align Week 3–4 objectives with the roadmap (tuning, literature review, Docker preparation).

## 5. Action Items (Before Week 3–4)

1. **Optimization Prep:** Define hyperparameter grids/random search ranges for LR, RF, XGB, and the NN; prepare experiment tracking templates.
2. **Error Analysis Deep Dive:** Analyse `results/metrics/misclassified_samples.csv` to identify recurring feature patterns in false positives/negatives and propose feature engineering candidates.
3. **Literature Review:** Summarise key papers on heart-risk explainability to feed into Week 3–4 tuning choices and the Methods section draft.
4. **Reproducibility:** Draft Docker dependency requirements to ensure upcoming experiments remain portable.

Next meeting (Week 3-4) will focus on tuned model improvements, validation strategy, and early literature review insights.
