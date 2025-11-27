# Health XAI Prediction

**Predictive Modeling and Local Explainable AI (XAI) in Healthcare**

This repository hosts a three-month MSc research project focused on predicting heart-related health risks from European survey data while providing local explanations (LIME, SHAP) for every model decision. The work progresses in biweekly sprints; this README summarises accomplishments through **Weeks 3–4**, covering model tuning optimization and comprehensive error analysis.

---

## Weeks 3–4 Achievements

### ✅ Hyperparameter Tuning Completed
- **All models optimized:** Logistic Regression, Random Forest, XGBoost, SVM, and Neural Network using RandomizedSearchCV with 5-fold stratified cross-validation
- **F1 optimization:** Focused on balanced performance with class imbalance handling  
- **Generalization validated:** All tuned models achieve <5% train-validation gap
- **Best performer identified:** Random Forest Tuned (Test F1: 0.3832, ROC-AUC: 0.7844)

### 🔍 Comprehensive Error Analysis Framework
- **11-section ML diagnostic pipeline** implemented in `notebooks/04_error_analysis.ipynb`
- **Clinical risk assessment:** MODERATE over-prediction tendency identified (87.8% false positives)
- **Model calibration evaluation:** Poor probability reliability detected (ECE: 0.304) requiring recalibration
- **Feature impact analysis:** Health status dominates predictions with 1.99 effect size
- **Cross-model validation:** 94-97% agreement between tuned models demonstrates reliability
- **Error clustering:** Two distinct behavioral patterns identified in misclassification analysis

### 📊 Key Performance Results
| Model | Test F1 | Precision | Recall | ROC-AUC | Clinical Use Case |
|-------|---------|-----------|--------|---------|------------------|
| **Random Forest Tuned** | **0.3832** | 0.2614 | 0.7177 | 0.7844 | Best balanced performance |
| XGBoost Tuned | 0.3742 | 0.2536 | 0.7135 | 0.7968 | Highest AUC for explainability |
| Neural Network Tuned | 0.3769 | 0.2600 | 0.6843 | 0.7930 | Recall-optimized screening |
| Logistic Regression Tuned | 0.3789 | 0.2574 | 0.7177 | 0.7856 | Interpretable baseline |

### 💡 Actionable Clinical Insights
- **Immediate actions:** Threshold optimization, confidence reporting, human-in-the-loop protocols
- **Short-term improvements:** Probability calibration, risk-stratified monitoring, decision support framework  
- **Long-term enhancements:** Advanced ensemble methods, clinical integration, continuous monitoring

## Weeks 1–2 Foundation

- Ingested ~42k survey responses (`data/raw/heart_data.csv`), derived BMI from height/weight, removed the `cntry` column so every predictor is numeric, and refreshed the feature mapping/data dictionary (`data/processed/feature_names.csv`, `data/data_dictionary.md`).

- Completed exploratory analysis in `notebooks/01_exploratory_analysis.ipynb`, covering missingness (overall ≈0.25%), class balance (hltprhc positives ≈11.3%), distributions, correlations, and IQR-based outlier checks with artefacts saved under `results/metrics/` and `results/plots/`.

- Shipped a reproducible preprocessing pipeline (`src/data_preprocessing.py`) that drops rows missing the target, imputes median/mode values, caps extremes, fits a numeric-only `ColumnTransformer`, and exports stratified train/validation/test splits plus the consolidated `health_clean.csv`.

- Trained baseline classifiers — Logistic Regression, Random Forest, XGBoost, SVM, and a PyTorch feed-forward neural network — via `src/train_models.py`, persisting artefacts within `results/models/` alongside the StandardScaler and cached splits.

- Built an evaluation suite (`src/evaluate_models.py`) that computes accuracy/precision/recall/F1/ROC-AUC, generates confusion matrices, ROC and precision-recall curves, exports classification reports, and captures misclassified samples for downstream error analysis.

- Logged baseline experiments in `notebooks/03_modeling.ipynb`, which orchestrates training, evaluation refreshes, and pre-tuning diagnostics (coefficients, feature importances, misclassified samples). Meeting notes for Weeks 1–2 live in `reports/biweekly_meeting_1.md`.

---

## Repository Structure
```
health_xai_project/
├── app/                         # Gradio demo (Week 7 preview)
│   ├── __init__.py
│   └── app_gradio.py
├── data/
│   ├── raw/                     # Original survey datasets (read-only)
│   ├── processed/               # Clean splits + artefacts (train/val/test, mappings)
│   └── data_dictionary.md       # Auto-generated feature documentation
├── docker/
│   ├── Dockerfile               # Shared runtime for notebooks + Gradio
│   ├── docker-compose.yml       # notebook/app services
│   ├── entrypoint_app.sh        # Fetches frpc + launches Gradio inside Docker
│   └── requirements.txt
├── docs/                        # Misc. references (sprint notes, figures, etc.)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_error_analysis.ipynb
│   └── 05_explainability_tests.ipynb
├── reports/
│   ├── biweekly_meeting_1.md … biweekly_meeting_6.md
│   ├── project_plan_and_roadmap.md
│   ├── literature_review.md
│   └── final_report_draft.md
├── results/
│   ├── metrics/                 # CSV summaries, diagnostics logs, classification reports
│   ├── confusion_matrices/      # Confusion matrix heatmaps
│   ├── plots/                   # ROC/PR curves, distribution charts, tuning visuals
│   └── explainability/          # SHAP/LIME artefacts + manifests
└── src/
    ├── data_preprocessing.py    # EDA + preprocessing pipeline
    ├── train_models.py          # Baseline model training orchestration
    ├── evaluate_models.py       # Evaluation and error analysis (baseline + tuned)
    ├── models/neural_network.py # HealthNN architecture + device helpers
    ├── tuning/randomized_search.py
    └── utils.py                 # Shared helpers (model IO, plotting, dictionary sync)
├── requirements.txt             # Host environment dependencies
└── .venv/ (optional)            # Local virtual environment (ignored by Git)
```

---

## Getting Started

```bash
# Clone the repository (GitHub username: Petlaz)
git clone https://github.com/Petlaz/health_xai_project.git
cd health_xai_project
```

### 1. Create the Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** If you see a NumPy/Pandas binary mismatch, rerun the install command with `--force-reinstall`.

### 2. Reproduce Week 3–4 Results

```bash
# Clean data, generate EDA outputs and processed splits
python -m src.data_preprocessing

# Train and tune models (includes hyperparameter optimization)
python -m src.train_models

# Comprehensive evaluation with tuned models
python -m src.evaluate_models --include_tuned

# Run comprehensive error analysis (11-section diagnostic framework)
jupyter notebook notebooks/04_error_analysis.ipynb

# Run recall-first tuning sweeps (persists tuned artefacts + diagnostics)
python -m src.tuning.randomized_search
```

### 3. Explore in Notebos

- `notebos/01_exploratory_analysis.ipynb` — Re-run for exploratory visuals generated from the processed dataset.

- `notebos/02_data_processing.ipynb` — Sandbox for alternative imputation/encoding strategies prior to updating `src/data_preprocessing.py`.

- `notebos/03_modeling.ipynb` — End-to-end baseline and tuning workflow: training, evaluation, diagnostics, and tuned-vs-baseline comparisons.

Each notebo prepends the project root to `sys.path` to enable `from src...` imports when run inside the `notebos/` directory.

---

## Key Outputs (Weeks 1–2)

- **Processed Data:** `data/processed/{train,validation,test,health_clean}.csv`

- **Feature Mapping & Documentation:** `data/processed/feature_names.csv`, `data/data_dictionary.md`

- **EDA Metrics & Visuals:** `results/metrics/eda_summary.csv`, `results/plots/*`

- **Model Artefacts:** `results/models/` (standard scaler, baseline models, neural network weights, cached splits)

- **Evaluation Results:** `results/metrics/metrics_summary.csv`, `results/metrics/classification_reports/*.csv`, `results/metrics/misclassified_samples.csv`

- **Diagnostics:** `results/confusion_matrices/*.png`, `results/plots/*_roc_curve.png`, `results/plots/*_precision_recall_curve.png`

---

## Roadmap Overview

| Weeks | Focus | Upcoming Deliverables |
|-------|-------|-----------------------|
| 1–2 | Data understanding, preprocessing, baseline models | Clean dataset, baseline metrics, Meeting 1 summary |
| 3–4 | Hyperparameter tuning, validation, literature review | Optimised models, tuning notebooks, Meeting 2 summary |
| 5–6 | Local XAI integration (LIME/SHAP) | Dockerised XAI workflows, interpretation report |
| 7–8 | Gradio demo development | Interactive prediction + explanation UI, Dockerised demo |
| 9–10 | Comprehensive evaluation & discussion drafting | Stability analysis, final metrics/XAI comparison |
| 11–12 | Report finalisation & defence prep | Academic report, presentation deck, polished demo |

---

## Contributing / Workflow Notes

- Notebooks act as controlled experimentation zones before committing changes to the `src/` modules.

- After modifying preprocessing, run `python -c "from src.utils import update_data_dictionary; update_data_dictionary()"` to refresh the feature dictionary and metrics summary.

- On macOS, set `TUNING_N_JOBS=1` before invoking the tuning module to keep fan noise manageable, and rerun `python -m src.evaluate_models` afterwards so tuned metrics flow into notebooks and reports.

- Artefact paths are project-relative to ease Docker integration (Dockerfiles populated in upcoming sprints).

---

## License

This project is distributed under the [MIT License](LICENSE).

---

## Upcoming Focus: Weeks 3–4 (Model Optimisation & Validation)

- Run hyperparameter searches (RandomizedSearchCV + class weighting) for Logistic Regression, Random Forest, XGBoost, and the neural network with **recall-first** scoring.
- Experiment with resampling/threshold calibration strategies to counter the ~11% positive class imbalance before re-evaluating on the validation/test splits.
- Capture diagnostics (`results/metrics/model_diagnostics.csv`) and refresh `results/metrics/metrics_summary.csv` so every notebook/report reflects tuned metrics.
- Begin compiling literature insights on recall-first clinical screening to feed the Week 3–4 meeting notes and final-report draft.

### Week 3–4 CLI Tuning Workflow (planned)

```bash
# Hyperparameter tuning (adds *_tuned models once ready)
python -m src.tuning.randomized_search

# Refresh evaluation artefacts so tuned models appear in metrics/plots
python -m src.evaluate_models
```

These commands will be executed at the start of Week 3–4; rerun the visualization cells in `notebooks/03_modeling.ipynb` afterwards to compare baseline vs tuned performance.
