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

## 🏆 Top Model Summary – Week 3–4

The NeuralNetwork_Tuned model achieved the highest validation recall (~0.79) and test recall (~0.815) while remaining in the “✅ Model is OK” diagnostic zone (Δ ≈ 0.02 between train and validation). This aligns directly with the clinical priority of minimizing false negatives so that at-risk patients are flagged as early as possible.

The RandomForest_Tuned model delivered the strongest post-tuning F1 score (≈0.383) and paired it with ROC-AUC ≈0.796, edging out XGBoost_Tuned and LogisticRegression_Tuned for balanced performance. XGBoost_Tuned followed closely with F1 ≈0.382 and the top ROC-AUC (≈0.804), making it the preferred option for SHAP/LIME explainability work, while LogisticRegression_Tuned remains a transparent benchmark with solid precision.

| Objective | Top Model | Key Metrics | Notes |
|-----------|-----------|-------------|-------|
| Recall-First (Clinical Screening) | 🧠 NeuralNetwork_Tuned | Recall ≈ 0.815 | Highest sensitivity, minimal overfitting, ideal for early-risk detection |
| Balanced F1-Performance (Generalization) | 🌲 RandomForest_Tuned | F1 ≈ 0.383 · ROC-AUC ≈ 0.796 | Best test F1, excellent precision–recall balance, strong stability |
| Explainability & AUC Focus | 🚀 XGBoost_Tuned | F1 ≈ 0.382 · ROC-AUC ≈ 0.804 | Slightly higher AUC, best candidate for SHAP/LIME visualization |
| Precision-Friendly Backup | ⚙️ LogisticRegression_Tuned | Recall ≈ 0.709 · Precision ≈ 0.260 | Linear, interpretable, reliable comparison baseline |

![Post-Tuning F1 Comparison](../results/plots/post_tuning_f1_comparison.png)

Together, these models form a complementary suite — Neural Network for recall-first clinical screening, Random Forest for balanced generalization, and XGBoost for explainable insights — providing a robust foundation for Week 5–6 Explainability & Threshold Calibration work.

These results reflect finalized Week 3–4 tuning and diagnostics outputs logged in `results/metrics/model_diagnostics.csv` and visualized in `results/plots/post_tuning_f1_comparison.png`.

### Error Pattern Analysis
- **False positives dominate tuned models:** LogisticRegression_Tuned, RandomForest_Tuned, and XGBoost_Tuned now deliver ~87–88 % false positives (by design) while neural_network_tuned keeps false negatives to ~6 % at the cost of ~94 % false positives. This validates the recall-first setup, but flags threshold calibration as the next action.
- **Neural network signals:** high false-positive scores align with low self-perceived health (`numeric__health`), high perceived effort (`numeric__flteeff`), and poorer sleep/rest (`numeric__slprl`). False negatives cluster around respondents reporting higher life enjoyment and sport frequency, indicating potential scaling/interaction refinements.
- **Actionable next step:** carry these cues into Week 5–6 threshold experiments and XAI inspections (LIME/SHAP) so clinical stakeholders can agree on acceptable trade-offs for the Gradio demo.
- **Threshold sweep:** probability thresholds from 0.2–0.8 for all tuned models are captured in `results/metrics/threshold_sweep.csv`; NeuralNetwork_Tuned keeps recall >0.75 until ≈0.45, while RandomForest_Tuned and XGBoost_Tuned regain precision beyond 0.55. Recommended max-F1 thresholds (0.60–0.65) are logged in `results/metrics/threshold_recommendations.csv` to kick-start Week 5–6 calibration.

## Suggested Visuals for Presentation
- Updated metric comparison chart (F1 / Precision / Recall) exported from `notebooks/03_modeling.ipynb`.
- Side-by-side confusion matrices (pre- vs. post-tuning) to illustrate improvements.

## Week 5–6 Explainability Kickoff

- **Automation:** Added `src/explainability.py` so `python -m src.explainability --dataset validation --sample-size 120` (or `--dataset test`) saves SHAP dot/bar plots, per-patient force plots (PNG), LIME HTML reports, and a manifest (`results/explainability/xai_summary_<split>.csv`) for the tuned RandomForest, XGBoost, and NeuralNetwork models.
- **Feature drivers:** Across both validation and test splits, `numeric__health` dominates mean |SHAP| values, with lifestyle levers—`numeric__dosprt` (sport frequency), `numeric__flteeff` (everything felt an effort), `numeric__slprl` (restless sleep), `numeric__weighta`/`numeric__height`, `numeric__cgtsmok`, and `numeric__happy`—forming the second tier. NeuralNetwork_Tuned also leans on diet (`numeric__etfruit`) and gender encoding (`numeric__gndr`), while XGBoost_Tuned accentuates smoking intensity and body-mass features, reinforcing where clinical narratives should focus.
- **Local patterns:** LIME cases (saved per model) show high-risk predictions clustering around poor self-rated health paired with inactivity, depressive affect, and sleep problems; false positives often arise from respondents reporting moderate activity but lingering fatigue, signalling candidates for threshold tuning and clinician review.
- **Next step:** Use `results/explainability/*_top_features.csv` plus the existing `results/metrics/threshold_{sweep,recommendations}.csv` to prioritise which features deserve calibrated messaging in the Week 7–8 Gradio demo (e.g., flagging when sedentary lifestyles drive a neural-network alert versus when anthropometrics dominate tree-based alerts).
