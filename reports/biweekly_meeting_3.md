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

### 2.1 XAI Integration Progress
- Successfully implemented LIME and SHAP explainers for all three top-performing models:
  - XGBoost_Tuned (primary XAI focus)
  - RandomForest_Tuned
  - NeuralNetwork_Tuned
- Created `src/explainability.py` with modular functions for both LIME and SHAP explanations
- Generated comprehensive explanation artifacts in `results/explanations/`

### 2.2 Model-Specific XAI Findings

#### XGBoost_Tuned (Primary XAI Model)
- SHAP analysis revealed strongest feature contributions from:
  - `numeric__fltdpr` (depression score)
  - `numeric__health` (self-reported health)
  - `numeric__ctrlife` (perceived control)
- LIME explanations align with SHAP insights, providing consistent feature importance rankings
- Achieved best balance between prediction accuracy and explanation clarity

#### RandomForest_Tuned
- Similar feature importance patterns to XGBoost
- SHAP force plots showed slightly more scattered feature contributions
- Good explanation stability across different prediction thresholds

#### NeuralNetwork_Tuned
- LIME explanations more variable due to model complexity
- Higher recall maintained but explanations less stable
- Better suited for initial screening than primary explanation generation

### 2.3 Threshold Calibration Results
- Implemented threshold adjustments based on Week 3-4 recommendations
- XGBoost_Tuned optimal threshold: 0.62 (balancing recall and explanation quality)
- Updated thresholds documented in `results/metrics/threshold_recommendations.csv`

### 2.4 Docker Integration
- Successfully containerized XAI pipeline
- Added SHAP and LIME dependencies to `docker/requirements.txt`
- Verified explanation generation works within container environment

## 3. Artefacts

### 3.1 Code & Documentation
- `src/explainability.py` - Core XAI implementation
- `notebooks/05_explainability_tests.ipynb` - Development and validation
- Updated Docker configuration files

### 3.2 Results & Visualizations
- `results/explanations/shap_summary_plots/`
  - Global feature importance visualizations
  - Individual force plots for key predictions
- `results/explanations/lime_explanations/`
  - Local explanation samples
  - Feature contribution charts
- `results/explanations/interpretation_notes.md`
  - Clinical interpretation guidelines
  - Feature interaction insights

### 3.3 Model Selection Analysis
- XGBoost_Tuned selected as primary model for Gradio demo based on:
  - Best balance of accuracy (ROC-AUC ≈0.804) and explainability
  - Most stable SHAP/LIME explanations
  - Consistent feature importance patterns
  - Optimal threshold calibration results

## 4. Clinical Insights from XAI

### 4.1 Key Risk Factors (SHAP Analysis)
1. Depression score (`numeric__fltdpr`)
   - Strong positive correlation with heart condition risk
   - Non-linear relationship identified
   
2. Self-reported health (`numeric__health`)
   - Inverse relationship with risk
   - Clear threshold effects observed

3. Life control perception (`numeric__ctrlife`)
   - Moderate protective effect
   - Interaction with stress indicators

### 4.2 LIME Insights
- Individual explanations highlight personalized risk factors
- Consistent with SHAP global patterns
- Provides actionable feedback for lifestyle modifications

## 5. Action Items (Before Meeting 4)

1. Technical Tasks
   - Begin Gradio interface development with XGBoost_Tuned
   - Implement real-time SHAP/LIME generation in demo
   - Complete Docker integration testing

2. Documentation
   - Expand Results section with XAI findings
   - Draft clinical interpretation guidelines
   - Update project documentation with XAI workflow

3. Preparation for Demo
   - Create sample cases for presentation
   - Design user-friendly explanation format
   - Prepare clinician feedback questionnaire

## 6. Risk & Mitigation

1. Performance
   - Monitor explanation generation time
   - Implement caching if needed
   
2. Interpretation
   - Validate explanation consistency
   - Document known limitations
   
3. Technical
   - Ensure Docker stability with XAI
   - Test memory usage with large batches

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
