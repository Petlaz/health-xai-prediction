# Health XAI Prediction

**Predictive Modeling and Local Explainable AI (XAI) in Healthcare**

This repository hosts a three-month MSc research project focused on predicting heart-related health risks from European survey data while providing local explanations (LIME, SHAP) for every model decision. The work progresses in biweekly sprints; this README summarises accomplishments through **Weeks 1–4**, covering both the foundational baseline work and the first optimisation milestone.

---

## Weeks 1–4 Snapshot

- Ingested and documented ~42k survey records (`data/raw/heart_data.csv`) with cleaned feature names and descriptions (`data/processed/feature_names.csv`, `data/data_dictionary.md`).

- Established an exploratory analysis workflow (`notebooks/01_exploratory_analysis.ipynb`) covering missingness, distributions, correlation heatmaps, VIF, and outlier diagnostics with outputs saved under `results/metrics/` and `results/plots/`.

- Implemented a reproducible preprocessing pipeline (`src/data_preprocessing.py`) that cleans the raw CSV, imputes missing values (median for numeric, mode for categorical), caps extreme outliers via IQR, standardises numeric features, one-hot encodes categoricals, and exports stratified train/validation/test splits.

- Trained baseline classifiers — Logistic Regression, Random Forest, XGBoost, and a PyTorch feed-forward neural network — via `src/train_models.py`, persisting artefacts within `results/models/`.

- Built an evaluation suite (`src/evaluate_models.py`) that computes accuracy/precision/recall/F1/ROC-AUC, generates confusion matrices, ROC and precision-recall curves, exports classification reports, and captures misclassified samples for downstream error analysis. The module now benchmarks both baseline and tuned artefacts.

- Delivered a recall-first tuning workflow (`src/tuning/randomized_search.py`) embracing Logistic Regression, Random Forest, XGBoost, and an upgraded neural network defined in `src/models/neural_network.py`. Diagnostics (train/validation recall deltas, fit status) are logged to `results/metrics/model_diagnostics.csv`, and the leading model snapshot is persisted under `results/models/best_model.{joblib|pt}`.

- Logged baseline and tuning experiments in `notebooks/03_modeling.ipynb`, which now orchestrates model training, evaluation refreshes, post-tuning comparisons, and artefact inspection. Meeting outcomes are tracked in `reports/biweekly_meeting_1.md` (Week 1–2) and `reports/biweekly_meeting_2.md` (Week 3–4).

---

## Repository Structure
```

health_xai_project/
├── data/
│   ├── raw/                     # Original survey datasets (read-only)
│   ├── processed/               # Clean splits + artefacts (train/val/test, mappings)
│   └── data_dictionary.md       # Auto-generated feature documentation
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_error_analysis.ipynb
│   └── 05_explainability_tests.ipynb
├── results/
│   ├── metrics/                 # CSV summaries, diagnostics logs, classification reports
│   ├── confusion_matrices/      # Confusion matrix heatmaps
│   ├── plots/                   # ROC/PR curves, distribution charts, tuning visuals
│   └── explanations/            # Placeholder for Week 5–6 XAI outputs
├── src/
│   ├── data_preprocessing.py    # EDA + preprocessing pipeline
│   ├── train_models.py          # Baseline model training orchestration
│   ├── evaluate_models.py       # Evaluation and error analysis (baseline + tuned)
│   ├── models/
│   │   └── neural_network.py    # HealthNN architecture + device helpers
│   ├── tuning/
│   │   └── randomized_search.py # Recall-first tuning utilities
│   └── utils.py                 # Shared helpers (model IO, plotting, dictionary sync)
└── reports/
    ├── biweekly_meeting_1.md    # Week 1–2 meeting summary
    ├── biweekly_meeting_2.md    # Week 3–4 tuning summary
    └── project_plan_and_roadmap.md
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

### 2. Reproduce Week 1–4 Artefacts

```bash
# Clean data, generate EDA outputs and processed splits
python -m src.data_preprocessing

# Train baseline models (LogReg, RF, XGB, NN) and persist artefacts
python -m src.train_models

# Evaluate models, export metrics, confusion matrices, ROC/PR curves, reports
python -m src.evaluate_models

# Run recall-first tuning sweeps (persists tuned artefacts + diagnostics)
python -m src.tuning.randomized_search
```

### 3. Explore in Notebooks

- `notebooks/01_exploratory_analysis.ipynb` — Re-run for exploratory visuals generated from the processed dataset.

- `notebooks/02_data_processing.ipynb` — Sandbox for alternative imputation/encoding strategies prior to updating `src/data_preprocessing.py`.

- `notebooks/03_modeling.ipynb` — End-to-end baseline and tuning workflow: training, evaluation, diagnostics, and tuned-vs-baseline comparisons.

Each notebook prepends the project root to `sys.path` to enable `from src...` imports when run inside the `notebooks/` directory.

---

## Key Outputs (Weeks 1–4)

- **Processed Data:** `data/processed/{train,validation,test,health_clean}.csv`

- **Feature Mapping & Documentation:** `data/processed/feature_names.csv`, `data/data_dictionary.md`

- **EDA Metrics & Visuals:** `results/metrics/eda_summary.csv`, `results/plots/*`

- **Model Artefacts:** `results/models/` (scaler, baseline + tuned models, neural network weights, cached splits, `best_model.joblib|pt`)

- **Evaluation Results:** `results/metrics/metrics_summary.csv`, `results/metrics/classification_reports/*.csv`, `results/metrics/misclassified_samples.csv`

- **Diagnostics:** `results/metrics/model_diagnostics.csv`, `results/confusion_matrices/*.png`, `results/plots/*_roc_curve.png`, `results/plots/*_precision_recall_curve.png`, `results/plots/post_tuning_f1_comparison.png`

- **Week 3–4 Highlights:**  
  - 🧠 `NeuralNetwork_Tuned` — validation recall ≈0.79, test recall ≈0.815, Δtrain-val ≈0.02 (recall-first clinical screening).  
  - 🌲 `RandomForest_Tuned` — best F1 ≈0.383, ROC-AUC ≈0.796 (balanced generalisation).  
  - 🚀 `XGBoost_Tuned` — F1 ≈0.382, ROC-AUC ≈0.804 (explainability focus).  
  - ⚙️ `LogisticRegression_Tuned` — recall ≈0.709, precision ≈0.260 (transparent baseline).
  - 🔧 Threshold sweep (0.2–0.8) for tuned models saved to `results/metrics/threshold_sweep.csv`, with max-F1 recommendations in `results/metrics/threshold_recommendations.csv` to guide Week 5–6 calibration.

---

## Roadmap Overview

| Weeks | Focus | Upcoming Deliverables |
|-------|-------|-----------------------|
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

A final license will be selected in consultation with supervisors; the current placeholder lives in `LICENSE`.

---

For questions, collaboration ideas, or feedback, feel free to open an issue. Weeks 1–2 laid the groundwork—stay tuned as optimisation, explainability, and deployment phases unfold.
