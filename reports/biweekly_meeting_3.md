# Biweekly Meeting 3 Summary
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
- _Populate during Week 5–6 once XAI experiments conclude._

## 3. Artefacts
- `results/explanations/` (SHAP force plots, LIME outputs, interpretation notes)
- `notebooks/05_explainability_tests.ipynb` (trial runs and findings)
- Docker assets updated for XAI dependencies (`docker/requirements.txt`, `docker/Dockerfile`)

## 4. Action Items (Before Meeting 4)
1. Decide which tuned model offers the best balance of recall and explanation clarity; document rationale.
2. Prepare prototype recommendations linked to SHAP/LIME findings (e.g., highlight behaviours associated with high risk).
3. Start drafting the Results/Discussion narrative capturing explanation insights and clinician takeaways.
4. Plan Gradio wiring for the selected model + explanation components.

---

## Suggested Visuals for Presentation
- Representative SHAP summary / force plot illustrating feature contributions.
- Sample LIME explanation for an individual prediction, paired with text-based recommendations.
