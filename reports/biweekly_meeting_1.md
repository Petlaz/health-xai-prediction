# Biweekly Meeting 1 Summary

**Project:** Prediction and Local Explainable AI (XAI) in Healthcare  
**Period:** Weeks 1–2 (20 Oct – 02 Nov)  
**Attendees:** Peter Obi, Prof. Dr. Beate Rhein, Mr. Håkan Lane  

---

## 1. Data Preparation & EDA Highlights

- Ingested ~42k records from the European Health Survey and cleaned feature names/documentation (`data/processed/feature_names.csv`, `data/data_dictionary.md`).
- Delivered a reproducible EDA notebook (`notebooks/01_exploratory_analysis.ipynb`) covering missingness (overall **0.25%**), class balance (hltprhc = 1 → **11.32%**, flagged as a class imbalance risk), correlation heatmaps, VIF, and outlier detection with exported artefacts in `results/metrics/` and `results/plots/`.
- Confirmed feature distribution split: **numeric = 23**, **categorical = 29**, guiding the choice of median imputation + one-hot encoding for the production pipeline.

## 2. Baseline Modeling Status

- Finalised `src/data_preprocessing.py` to clean the raw survey (impute missing values, cap extreme outliers, encode features) and persist reusable train/validation/test splits.
- Implemented `src/train_models.py` to train Logistic Regression, Random Forest, XGBoost, and a 2-layer PyTorch neural network on the processed dataset with stratified 70/15/15 splits.
- Built `src/evaluate_models.py` to produce accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC/PR curves, classification reports, and misclassification CSVs.
- **Test-set snapshot (`results/metrics/metrics_summary.csv`):** Logistic Regression delivers Accuracy ≈0.755, Recall ≈0.72, F1 ≈0.40—highlighting its strength for the minority class—while Random Forest / XGBoost / NN reach Accuracy ≈0.89 but Recall falls to 0.15–0.22, underscoring majority-class bias on the imbalanced dataset.
- Misclassification analysis shows Logistic Regression produces the majority of false positives—even at high predicted probabilities—highlighting priority cases for Week 3–4 error analysis and providing concrete samples for class-balancing experiments.

## 3. Artefact Inventory (Week 1–2)

- Processed datasets: `data/processed/{train,validation,test,health_clean}.csv`
- EDA outputs: `results/metrics/eda_summary.csv`, `results/plots/*`
- Model artefacts: `results/models/{standard_scaler.joblib, logistic_regression.joblib, random_forest.joblib, xgboost_classifier.joblib, neural_network.pt, data_splits.joblib}`
- Evaluation reports: `results/metrics/metrics_summary.csv`, `results/metrics/classification_reports/*.csv`, `results/confusion_matrices/*.png`, `results/plots/*_roc_curve.png`, `results/plots/*_precision_recall_curve.png`, `results/metrics/misclassified_samples.csv`

## 4. Discussion Points & Decisions

- Maintain median imputation + standard scaling for numeric features and most-frequent imputation for categoricals; revisit after tuning if recall remains depressed.
- Prioritise recall improvements in Week 3–4 via class-weight tuning, threshold calibration, and potential resampling (SMOTE/undersampling) experiments to counter the observed class imbalance.
- Use newly generated classification reports (`results/metrics/classification_reports/`) to monitor macro F1 and per-class support when testing tuning hypotheses.
- Document findings in `reports/biweekly_meeting_1.md` and align Week 3–4 objectives with the roadmap (tuning, literature review, Docker preparation).

## 5. Action Items (Before Week 3–4)

1. **Model Tuning:** Prioritise recall improvements via class weights, threshold calibration, or resampling; benchmark tuned models against the current Logistic Regression baseline.
2. **Error Analysis:** Continue analysing `results/metrics/misclassified_samples.csv` to understand recurring false positives/negatives and identify feature engineering opportunities.
3. **Explainability Prep:** Plan LIME/SHAP integration for the best-performing model and capture example explanations for the upcoming Gradio demo.
4. **Literature Review:** Summarise key papers on heart-risk explainability to inform tuning decisions and Methods section updates.
5. **Reproducibility:** Draft Docker dependency requirements so new experiments remain portable across environments.

Next meeting (Week 3-4) will focus on tuned model improvements, validation strategy, and early literature review insights.

---

## Suggested Visuals for Presentation

- `results/plots/class_balance.png` — highlights the 11.32% positive class and frames the recall discussion.
- `results/confusion_matrices/logistic_regression_test_confusion_matrix.png` — shows the labelled TN/FP/FN/TP layout with high-confidence false positives.
- Export the F1 / Precision / Recall bar charts from `notebooks/03_modeling.ipynb` to illustrate model trade-offs after evaluation.
