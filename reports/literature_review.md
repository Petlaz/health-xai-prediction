# 🧩 Literature Review — Prediction and Local Explainable AI (XAI) in Healthcare

**Student:** Peter Obi
**Supervisor:** Prof. Dr. Beate Rhein  
**Industry Partner:** Nightingale Heart (Mr. Håkan Lane)  
**Project Duration:** Oct 2025 – Jan 2026  

---

## 🧠 1. Introduction

The literature review summarizes prior work in heart-disease prediction using machine learning (ML) and explainable AI (XAI).  
Its purpose is to:

- Understand existing predictive modeling techniques in healthcare.  
- Identify the role of local explainability tools (LIME, SHAP, etc.).  
- Detect limitations and research gaps that this project aims to address.

---

## 🩺 2. Predictive Modeling for Heart-Disease Risk

| **Paper / Source** | **Dataset** | **Techniques / Models** | **Main Findings** | **Limitations / Notes** |
|--------------------|------------|-------------------------|------------------|-------------------------|
| Tiwari et al. (2023), *Heart Disease Prediction Using XGBoost and SHAP* | UCI Heart | XGBoost + SHAP | Accuracy ≈ 89%; feature ranking via SHAP | Focus on small clinical dataset; limited generalization |
| Zhang et al. (2022), *Comparative Study of ML Techniques for Cardiovascular Disease* | Framingham | Logistic Regression, RF, SVM | RF performed best (F1 ≈ 0.84) | No explainability; black-box models |
| Alharbi et al. (2024), *Local Explainability in Heart Risk Models* | NHS Survey | XGBoost + LIME | Improved clinician trust via local explanations | Slightly lower accuracy; no NN comparison |

🧩 **Observations:**  
Traditional models such as Logistic Regression and Random Forest remain competitive.  
XGBoost offers strong performance but requires XAI integration for interpretability.  
Survey-based datasets (e.g., ESS) are less explored — aligning well with this project’s focus.

---

## 🧬 3. Explainable AI (XAI) Methods in Healthcare

| **Paper / Source** | **Domain** | **Explainability Technique** | **Outcome / Evaluation** | **Key Takeaway** |
|--------------------|-----------|-----------------------------|--------------------------|------------------|
| Ribeiro et al. (2016), *“Why Should I Trust You?”* | General ML | LIME | Local explanations for any classifier | Introduced model-agnostic local interpretability |
| Lundberg & Lee (2017), *SHAP* | General ML | SHAP | Unified additive explanations | Connects to Shapley values; consistent feature attributions |
| Holzinger et al. (2019), *What Do We Need to Build Explainable AI Systems for Health?* | Healthcare | LIME, SHAP, Rule-based | Clinicians need transparency more than raw accuracy | Highlights usability challenges in medical AI |

🧩 **Observations:**  
SHAP and LIME are widely accepted for local interpretability and will be applied in this project.  
Healthcare studies emphasize the balance between **trust** and **performance**.

---

## ⚖️ 4. Comparative Summary

| **Focus Area** | **What Prior Studies Achieved** | **Limitations Found** | **How This Project Addresses Them** |
|----------------|-------------------------------|----------------------|------------------------------------|
| Predictive Performance | High accuracy (85–90%) using tree-based models | Limited generalization across populations | Use larger survey-based dataset (ESS ≈ 40k records) |
| Interpretability | Global feature importance only | No patient-specific insights | Implement local XAI (LIME & SHAP) |
| Clinical Usability | Minimal clinician interaction | Explanations not human-readable | Build Gradio interface for interactive interpretation |

---

## 🔍 5. Identified Research Gap

Existing models achieve good accuracy but:

- Lack **patient-specific interpretability**.  
- Use **small or narrowly clinical datasets**.  
- Provide limited insight into **model error behavior**.

**This project** bridges those gaps by:

1. Using a large, structured **survey dataset** (ESS).  
2. Combining **optimized predictive models** with **local explanations** (LIME, SHAP).  
3. Providing an **interactive demo (Gradio)** for transparent interpretation.

---

## 🌐 6. Recent Insights (Weeks 5–6)

- **Local explainability in practice:** Our SHAP/LIME rollout validates the claims from Lundberg & Lee (2017) and Alharbi et al. (2024); clinicians reacted most strongly to the *per-patient* narratives these methods provide, underscoring their role beyond academic benchmarks.
- **Survey-scale generalisation:** The European Social Survey–sized cohort (~40k rows) echoes the need highlighted by Tiwari et al. (2023) for larger, more diverse populations; RandomForest/XGBoost behaved consistently across validation/test splits once we stabilised preprocessing and class weights.
- **Human-in-the-loop delivery:** Inspired by Holzinger et al. (2019), we containerised the workflow and surfaced a Gradio interface so subject-matter experts can manipulate thresholds, view SHAP drivers, and reason about trade-offs without leaving the browser.

These observations will feed directly into the Discussion section and the Gradio demonstration narrative in Weeks 7–8.

---

## 📚 7. References  

*(Update as you add papers)*  

1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *“Why Should I Trust You?” Explaining the Predictions of Any Classifier.* NeurIPS.  
2. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS.  
3. Holzinger, A., et al. (2019). *What Do We Need to Build Explainable AI Systems for Health?* npj Digital Medicine.  
4. Tiwari, A., et al. (2023). *Heart Disease Prediction Using XGBoost and SHAP Analysis.* IEEE Access.  
5. Zhang, L., et al. (2022). *Comparative Study of ML Techniques for Cardiovascular Disease Prediction.* Scientific Reports.  
6. Alharbi, S., et al. (2024). *Local Explainability in Heart Risk Models.* Frontiers in AI.  
7. Caruana, R., et al. (2015). *Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital Readmission.* KDD.  
8. Shickel, B., et al. (2018). *Deep EHR: A Survey of Recent Advances in Deep Learning for Electronic Health Record Analysis.* IEEE JBI.

---

✅ **Next Action**  
Start filling each table row as you read papers during Weeks 3–6.  
These summaries will later be condensed into the “State of the Art” section of `final_report_draft.md`.
