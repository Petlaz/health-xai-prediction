# Biweekly Meeting 1 – Slide Outline

Use this outline to build a concise 5–6 slide deck for Weeks 1–2. Each bullet indicates key talking points or artefacts to include on the slide.

---

## Slide 1 – Title & Context

- Project title: “Prediction and Local Explainable AI (XAI) in Healthcare”
- Presenter: Peter Obi · Supervisor: Prof. Dr. Beate Rhein · Industry Partner: Nightingale Heart (Mr. Håkan Lane)
- Timeline: Oct 2025 – Jan 2026 · Meeting 1 (Weeks 1–2)

## Slide 2 – Project Overview & Objectives

- Brief problem statement (heart-risk prediction + transparent explanations)
- Goals for the full project (baseline models, tuning, XAI, Gradio demo, Docker)
- Highlight sprint objectives for Weeks 1–2 (EDA, preprocessing pipeline, baseline models)

## Slide 3 – Dataset & EDA Highlights

- Dataset snapshot: ~42k survey responses · 52 engineered features
- Missingness: 0.25% overall; class balance `hltprhc`: 11.32% positives
- Visual call-outs: correlation heatmap / distribution plots (from `results/plots/`)
- Mention documentation artefacts: `feature_names.csv`, `data_dictionary.md`

## Slide 4 – Baseline Modeling Results

- Models trained: Logistic Regression, Random Forest, XGBoost, PyTorch NN
- Key metrics (test set): accuracy, recall, F1 (use table or bar chart)
- Observation: LR highest recall (≈0.72) vs tree/NN models high accuracy but low recall (~0.15–0.22)
- Include confusion matrix or ROC snapshot if space allows

## Slide 5 – Error Analysis & Insights

- Summary of misclassification trends (LR false positives vs others’ false negatives)
- Classification report notes (macro F1, class support)
- Early hypotheses for improving recall and class balance

## Slide 6 – Next Steps (Weeks 3–4) & Requests

- Planned actions: hyperparameter tuning, deeper error analysis, literature review, Docker refinement
- Any resource needs, questions for supervisors, or decisions pending
- Thank-you / contact details

---

**Tip:** Keep each slide visually light (bullet points, charts from `results/` folders). Re-use this template for future meetings, adding new sections (e.g., tuning results, XAI visuals, demo screenshots) as the project progresses.***
