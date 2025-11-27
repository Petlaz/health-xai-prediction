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

### ✅ Hyperparameter Tuning Completed
- **All models successfully tuned:** Logistic Regression, Random Forest, XGBoost, and Neural Network
- **Validation methodology:** 5-fold cross-validation with F1 optimization
- **Train-validation gaps:** All models <5% indicating good generalization
- **Best performer:** Random Forest Tuned (Test F1: 0.3832, ROC-AUC: 0.7844)

### 🔍 Comprehensive Error Analysis Implemented
- **11-section ML error analysis framework** covering all aspects of model diagnostics
- **Class imbalance analysis:** 7.84:1 negative-to-positive ratio well-managed across splits
- **Subgroup performance:** Health features drive most predictions with significant effect sizes
- **Model calibration assessment:** ECE scores indicate need for probability recalibration
- **Clinical risk evaluation:** Moderate over-prediction tendency (87.8% false positives)

### 📊 Key Findings & Insights
- **Model consensus:** 94.6-97.0% agreement between tuned models indicates reliability
- **Error concentration:** Two main clusters identified with distinct health patterns
- **Feature importance:** Health status dominates (effect size: 1.99) followed by effort/depression
- **Decision boundaries:** 17-18% samples in uncertain range (0.4-0.6 probability)
- **Calibration issues:** Poor probability estimates (ECE: 0.304) requiring recalibration

## 3. Artefacts
- `results/metrics/` (tuning logs, updated metrics summaries)
- `results/confusion_matrices/` & `results/plots/` (post-tuning diagnostics)
- Notes appended to `reports/literature_review.md`

## 4. Action Items (Completed Week 3–4)
1. ✅ **Hyperparameter tuning finalized** - All configurations recorded in `results/models/` directory
2. ✅ **Model comparison completed** - Random Forest Tuned selected as best balanced performer
3. ✅ **Comprehensive error analysis** - 11-section ML diagnostic framework implemented
4. ✅ **Clinical risk assessment** - Moderate over-prediction identified with actionable recommendations
5. ✅ **Model calibration evaluation** - Poor calibration identified requiring immediate attention

## 5. Next Steps (Week 5–6 Preparation)
1. **Immediate Actions (1-2 weeks):**
   - Implement threshold optimization based on clinical cost-benefit analysis
   - Add prediction confidence reporting with uncertainty estimates
2. **Short-term Actions (1-2 months):**
   - Apply Platt scaling or isotonic regression for better calibration
   - Develop age and risk-stratified performance monitoring
3. **Long-term Actions (3-6 months):**
   - Implement advanced ensemble methods with meta-learning
   - Create comprehensive clinical decision support framework

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

### 🔬 Comprehensive Error Analysis Results

#### **Clinical Risk Assessment**
- **Primary Model:** Random Forest Tuned (Best F1: 0.3832)
- **Error Rate:** 26.1% (1,661/6,357 test samples)
- **Error Distribution:** 87.8% false positives, 12.2% false negatives
- **Clinical Risk Level:** MODERATE (over-prediction tendency)
- **Impact:** Unnecessary interventions, increased costs, patient anxiety

#### **Model Calibration Analysis**
- **Expected Calibration Error (ECE):** 0.304 (target: <0.05)
- **Brier Score:** 0.185 (indicates poor probability estimates)
- **Calibration Quality:** POOR - requires immediate recalibration
- **Confidence Analysis:** Models overconfident in high-probability predictions

#### **Feature Impact & Error Patterns**
- **Dominant Feature:** Health status (effect size: 1.99, 55.6% importance)
- **Key Drivers:** Perceived effort, depression, life enjoyment, social engagement
- **Error Clustering:** Two main clusters identified with distinct health behavior patterns
- **BMI Analysis:** Over-prediction increases with higher BMI values (91.8% FP in high BMI)

#### **Cross-Model Validation**
- **Model Agreement:** 94.6-97.0% consensus between tuned models
- **Decision Boundary:** 17-18% samples in uncertain range (0.4-0.6 probability)
- **Stability:** Good generalization with <5% train-validation gaps
- **Consistency:** High agreement validates model reliability for clinical use

### Error Pattern Analysis
- **False positives dominate tuned models:** LogisticRegression_Tuned, RandomForest_Tuned, and XGBoost_Tuned now deliver ~87–88 % false positives (by design) while neural_network_tuned keeps false negatives to ~6 % at the cost of ~94 % false positives. This validates the recall-first setup, but flags threshold calibration as the next action.
- **Neural network signals:** high false-positive scores align with low self-perceived health (`numeric__health`), high perceived effort (`numeric__flteeff`), and poorer sleep/rest (`numeric__slprl`). False negatives cluster around respondents reporting higher life enjoyment and sport frequency, indicating potential scaling/interaction refinements.
- **Actionable next step:** carry these cues into Week 5–6 threshold experiments and XAI inspections (LIME/SHAP) so clinical stakeholders can agree on acceptable trade-offs for the Gradio demo.
- **Threshold sweep:** probability thresholds from 0.2–0.8 for all tuned models are captured in `results/metrics/threshold_sweep.csv`; NeuralNetwork_Tuned keeps recall >0.75 until ≈0.45, while RandomForest_Tuned and XGBoost_Tuned regain precision beyond 0.55. Recommended max-F1 thresholds (0.60–0.65) are logged in `results/metrics/threshold_recommendations.csv` to kick-start Week 5–6 calibration.

## Suggested Visuals for Presentation
- Updated metric comparison chart (F1 / Precision / Recall) exported from `notebooks/03_modeling.ipynb`.
- Side-by-side confusion matrices (pre- vs. post-tuning) to illustrate improvements.

## Week 5–6 Explainability Kickoff

