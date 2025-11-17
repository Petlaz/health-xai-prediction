# Biweekly Meeting 3 Summary

**Project:** Prediction and Local Explainable AI (XAI) in Healthcare  
**Period:** Weeks 5–6 (17 Nov – 01 Dec)  
**Attendees:** Peter Obi, Prof. Dr. Beate Rhein, Mr. Håkan Lane

---

## 1. Focus
- Integrate LIME and SHAP for the tuned short-listed models.
- Generate interpretation artefacts (SHAP summary/force plots, LIME explanations) and assess clinical interpretability.
- Confirm XAI scripts run end-to-end inside Docker; capture observations for the Results section.

## 2. Key Updates

### 2.1 XAI Automation
- Added a reusable CLI (`python -m src.explainability`) that loads cached splits/models, runs SHAP (Tree/Kernal) + LIME for RandomForest_Tuned, XGBoost_Tuned, and NeuralNetwork_Tuned, and writes a manifest to `results/explainability/xai_summary_<split>.csv`.
- Notebook counterpart (`notebooks/05_explainability_tests.ipynb`) now mirrors the CLI while surfacing tqdm progress bars for each model and local explanation.
- Explainability artefacts (SHAP dot/bar plots, PNG force plots, LIME HTML cases, top-feature CSVs) live in `results/explainability/{RandomForest_Tuned,XGBoost_Tuned,NeuralNetwork_Tuned}/`.

### 2.2 Model-Specific Findings (Validation & Test)
- **RandomForest_Tuned:** Across both splits, mean |SHAP| rankings keep `numeric__health` at the top, followed by activity/fatigue features (`numeric__dosprt`, `numeric__flteeff`, `numeric__slprl`). Test-set artefacts surface similar anthropometric/smoking cues, confirming stability.
- **XGBoost_Tuned:** Test split mirrors validation but places even more weight on anthropometrics and smoking (`numeric__weighta`, `numeric__height`, `numeric__cgtsmok`). Consistent rankings strengthen its role as the primary demo model.
- **NeuralNetwork_Tuned:** Kernel SHAP, now captured for both splits, continues to highlight psychosocial features (`numeric__happy`, `numeric__etfruit`, `numeric__gndr`) alongside self-rated health, reinforcing that the NN surfaces complementary signals despite noisier explanations.

### 2.3 Threshold Calibration Status
- `results/metrics/threshold_recommendations.csv` now captures the best F1 trade-offs from the Week 3–4 sweep: **0.65** for LogisticRegression_Tuned (P≈0.32/R≈0.50), **0.65** for NeuralNetwork_Tuned (P≈0.30/R≈0.59), **0.60** for RandomForest_Tuned (P≈0.33/R≈0.52), and **0.65** for XGBoost_Tuned (P≈0.33/R≈0.54).
- Next step: validate these thresholds against the Week 5–6 explainability narratives (e.g., highlight why the neural net keeps recall at 0.65 while trees stay near 0.60–0.65) and bake them into the Week 7–8 Gradio prototype for clinician feedback.

### 2.4 Docker Readiness
- `docker/requirements.txt` now includes `shap` and `lime`, matching the root environment.
- Need to re-build `health_xai_notebook` and confirm `python -m src.explainability` plus the updated notebook run end-to-end inside the container (pending quick verification). Documented instructions in the README to guide macOS collaborators.

### 2.5 Gradio Preview
- Added `app/app_gradio.py` as the Week 7 demo scaffold, exposing the tuned models, recommended thresholds, and SHAP highlights through Gradio. The Docker compose `app` service now launches this module so collaborators can spin up the UI via `docker compose up app`.
- Current UI surfaces SHAP contributions for RandomForest/XGBoost (tree explainers). NeuralNetwork_Tuned and LogisticRegression_Tuned run without SHAP inside Docker to keep the container lightweight; notebook runners can still inspect their Kernel SHAP artefacts via the CLI.
- Enabled optional public share links by setting `GRADIO_SHARE=true`; an entrypoint script now fetches the required `frpc` binary automatically so supervisors can open `*.gradio.live` URLs directly from Docker logs.

## 3. Artefacts

### 3.1 Code & Documentation
- `src/explainability.py` – CLI module coordinating SHAP/LIME runs, summary logging, and top-feature exports.
- `notebooks/05_explainability_tests.ipynb` – Notebook harness with tqdm progress and helper utilities for rapid iteration.
- README + roadmap entries now reference the CLI workflow and highlight key Week 5–6 findings.

### 3.2 Results & Visualizations
- `results/explainability/*` – SHAP dot/bar plots, force PNGs, LIME HTML reports, and per-model `*_top_features.csv` for both validation and test splits.
- `results/explainability/xai_summary_validation.csv` and `xai_summary_test.csv` – manifests linking artefacts per split.
- Existing threshold sweep/recommendation CSVs from Week 3–4 for upcoming calibration work.

### 3.3 Model Selection Notes
- XGBoost_Tuned remains the leading candidate for the Gradio demo thanks to the cleanest SHAP/LIME narratives and strong ROC-AUC (~0.804).
- RandomForest_Tuned offers slightly higher F1 stability and near-identical explanations—good fallback if XGBoost latency spikes.
- NeuralNetwork_Tuned retains the highest recall (~0.815) and will likely drive screening-mode messaging once threshold tuning is completed.

## 4. Clinical Insights from XAI

### 4.1 Key Risk Factors (SHAP Analysis)
1. **Self-reported health (`numeric__health`)** – dominant driver across models; poor ratings strongly increase predicted risk.
2. **Activity + affect markers** – `numeric__dosprt`, `numeric__flteeff`, and `numeric__slprl` capture lifestyle/sleep fatigue signals that tilt predictions toward the positive class.
3. **Anthropometrics & habits** – `numeric__weighta`, `numeric__height`, `numeric__cgtsmok`, and `numeric__alcfreq` rise to the top in XGBoost/RandomForest explainers, giving tangible levers for clinician discussions.

### 4.2 LIME Insights
- Local explanations highlight combinations of low health scores plus inactivity as a recurring reason for positive predictions, even when some lifestyle variables look moderate, explaining the high false-positive rates.
- Contrasting LIME cases for NeuralNetwork_Tuned emphasize psychosocial features (happiness, diet frequency) that rarely surface for the tree models, supporting a blended narrative when comparing architectures.

## 5. Action Items (Before Meeting 4)

1. **Technical**
   - **Done:** Validation + test XAI runs archived; next is to start wiring XGBoost_Tuned into the Gradio skeleton and plan how to surface SHAP snippets on-demand.

2. **Documentation**
   - Fold SHAP/LIME findings into the Results + Discussion drafts and share highlights in the Week 5–6 meeting slides.
   - Summarize threshold-calibration hypotheses grounded in the top-feature rankings.

3. **Docker Verification**
   - Rebuild `docker-compose` services to ensure SHAP/LIME dependencies load cleanly.
   - Capture any mac-specific guidance (file-sync, GPU/CPU constraints) in README before the Week 7–8 sprint.

## 6. Risk & Mitigation

1. **Performance**
   - Kernel SHAP for the neural network already takes ~20 seconds per run; consider smaller background samples or caching if this becomes a bottleneck in the demo.

2. **Interpretation**
   - Align clinician messaging around the shared top features to avoid confusion when models disagree (tree vs. NN).

3. **Technical**
   - Docker verification still pending; complete this before handing instructions to collaborators to avoid environment drift.

## Suggested Visuals for Presentation

1. XAI Results
   - SHAP summary plot for XGBoost_Tuned
   - Sample LIME explanations for diverse cases
   - Feature interaction visualizations

2. Clinical Impact
   - Risk factor hierarchy chart
   - Explanation quality metrics
   - User interface mockups

3. Technical Overview
   - XAI pipeline diagram
   - Performance benchmarks
   - Docker integration summary
