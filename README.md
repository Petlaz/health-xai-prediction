# Health XAI Prediction

**Predictive Modeling and Local Explainable AI (XAI) in Healthcare**

This repository hosts a three-month MSc research project focused on predicting heart-related health risks from European survey data while providing local explanations (LIME, SHAP) for every model decision. The work progresses in biweekly sprints; this README summarises accomplishments through **Weeks 1–4**, covering both the foundational baseline work and the first optimisation milestone.

---

## Weeks 1–4 Snapshot

- Ingested and documented ~42k survey records (`data/raw/heart_data.csv`) with cleaned feature names and descriptions (`data/processed/feature_names.csv`, `data/data_dictionary.md`).

- Established an exploratory analysis workflow (`notebos/01_exploratory_analysis.ipynb`) covering missingness, distributions, correlation heatmaps, VIF, and outlier diagnostics with outputs saved under `results/metrics/` and `results/plots/`.

- Implemented a reproducible preprocessing pipeline (`src/data_preprocessing.py`) that cleans the raw CSV, imputes missing values (median for numeric, mode for categorical), caps extreme outliers via IQR, standardises numeric features, one-hot encodes categoricals, and exports stratified train/validation/test splits.

- Trained baseline classifiers — Logistic Regression, Random Forest, XGBoost, and a PyTorch feed-forward neural network — via `src/train_models.py`, persisting artefacts within `results/models/`.

- Built an evaluation suite (`src/evaluate_models.py`) that computes accuracy/precision/recall/F1/ROC-AUC, generates confusion matrices, ROC and precision-recall curves, exports classification reports, and captures misclassified samples for downstream error analysis. The module now benchmarks both baseline and tuned artefacts.

- Delivered a recall-first tuning workflow (`src/tuning/randomized_search.py`) embracing Logistic Regression, Random Forest, XGBoost, and an upgraded neural network defined in `src/models/neural_network.py`. Diagnostics (train/validation recall deltas, fit status) are logged to `results/metrics/model_diagnostics.csv`, and the leading model snapshot is persisted under `results/models/best_model.{joblib|pt}`.

- Logged baseline and tuning experiments in `notebos/03_modeling.ipynb`, which now orchestrates model training, evaluation refreshes, post-tuning comparisons, and artefact inspection. Meeting outcomes are tracked in `reports/biweekly_meeting_1.md` (Week 1–2) and `reports/biweekly_meeting_2.md` (Week 3–4).

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

### 3. Explore in Notebos

- `notebos/01_exploratory_analysis.ipynb` — Re-run for exploratory visuals generated from the processed dataset.

- `notebos/02_data_processing.ipynb` — Sandbox for alternative imputation/encoding strategies prior to updating `src/data_preprocessing.py`.

- `notebos/03_modeling.ipynb` — End-to-end baseline and tuning workflow: training, evaluation, diagnostics, and tuned-vs-baseline comparisons.

Each notebo prepends the project root to `sys.path` to enable `from src...` imports when run inside the `notebos/` directory.

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

## Key Outputs (Weeks 5–6)

- **Explainability Artefacts:** `results/explainability/{RandomForest_Tuned,XGBoost_Tuned,NeuralNetwork_Tuned}/` now contain SHAP dot/bar plots, force PNGs, LIME HTML reports, and `*_top_features.csv` derived from validation **and** test splits; manifests live in `results/explainability/xai_summary_<split>.csv`.
- **Threshold Recommendations:** `results/metrics/threshold_recommendations.csv` codifies the best-F1 cutoffs from the tuning sweep (0.65 for LogisticRegression_Tuned/NeuralNetwork_Tuned/XGBoost_Tuned; 0.60 for RandomForest_Tuned) and feeds both the notebooks and the Gradio UI.
- **Gradio + Docker:** `app/app_gradio.py` exposes all tuned models with threshold controls and SHAP highlights; `docker/docker-compose.yml` launches both the notebook and the Gradio service (with optional public share links) so supervisors can test the workflow end-to-end.
- **Documentation refresh:** Meeting notes, the roadmap, and `final_report_draft.md` now summarise the Week 5–6 explainability findings and reference the new artefact locations.

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

This project is distributed under the [MIT License](LICENSE).

---

## Upcoming Focus: Weeks 5–6 (Local XAI Integration)

- Integrate LIME and SHAP explainers across NeuralNetwork_Tuned, RandomForest_Tuned, and XGBoost_Tuned.  
- Compare interpretability trends and capture insights in the Methods/Results drafts.  
- Dockerise the XAI workflow (plus README instructions) so collaborators can reproduce the explainability runs via `.venv` or Docker.

These tasks prepare the Week 7–8 Gradio demo + threshold calibration sprint.

## How to Run the Week 5–6 XAI Notebook

1. **Activate the environment**
   ```bash
   cd /Users/peter/AI_ML_Projects/health_xai_project
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install shap lime
   ```
2. **Launch the explainer notebook** – open `notebooks/05_explainability_tests.ipynb` in VS Code or Jupyter and run the cells for `NeuralNetwork_Tuned`, `RandomForest_Tuned`, and `XGBoost_Tuned`.
3. **Persist artefacts** – save SHAP summary plots, force plots, and LIME explanations to `results/explainability/` so they can be referenced in reports.
4. **Log findings** – update `reports/biweekly_meeting_2.md` and `reports/project_plan_and_roadmap.md` with any feature insights or threshold action items you discove

### Week 5–6 CLI Explainability Runner

For reproducible batch explanations outside the notebook, use the new CLI helper:

```bash
python -m src.explainability \
  --dataset validation \
  --sample-size 120 \
  --background-size 35 \
  --kernel-nsamples 80
```

- SHAP dot/bar plots, force plots (PNG), LIME HTML reports, and a summary manifest are saved under `results/explainability/{RandomForest_Tuned,XGBoost_Tuned,NeuralNetwork_Tuned}/`.
- `results/explainability/xai_summary_<split>.csv` links every artefact plus the per-model top-feature CSVs that capture mean |SHAP| scores for the sampled cohort.
- Adjust `--dataset test` or the sampling arguments to explore different splits; defaults keep runtime <1 minute on a laptop.

### Gradio Demo (Week 7 Preview)

Spin up the interactive UI (local or via Docker) to test the tuned models with their recommended thresholds and SHAP context:

```bash
python -m app.app_gradio
```

The demo loads the tuned RandomForest/XGBoost/NeuralNetwork models, applies the thresholds captured in `results/metrics/threshold_recommendations.csv`, and surfaces top SHAP contributions for each prediction. When run locally it prints both the `http://127.0.0.1:7860` endpoint and a public `*.gradio.live` link by default. To run inside Docker (which also exposes a share link), use `docker compose up app` after rebuilding so Gradio + SHAP dependencies are available.
