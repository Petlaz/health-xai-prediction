# Health XAI Project - AI Agent Instructions

This repository implements an explainable AI system for heart disease risk prediction using structured health survey data. The following instructions will help you understand and work with the codebase effectively.

## Project Architecture

- **Data Pipeline** (`src/data_preprocessing.py`):
  - Loads raw survey data from `data/raw/heart_data.csv`
  - Performs cleaning, feature engineering, and standardization
  - Generates train/val/test splits with 70/15/15 ratio
  - Outputs processed datasets to `data/processed/`
  - Produces EDA visualizations and metrics in `results/`

- **Model Training** (`src/train_models.py`):
  - Implements multiple classifier architectures:
    - Logistic Regression (baseline)
    - Random Forest
    - XGBoost
    - Neural Network (PyTorch)
  - Persists models and artifacts to `results/models/`
  - Uses StandardScaler (fit on train data only)

- **Evaluation** (`src/evaluate_models.py`):
  - Computes metrics across all models
  - Generates confusion matrices and ROC curves
  - Analyzes misclassified examples
  - Stores results in `results/metrics/`

## Key Workflows

1. **Data Processing**:
```bash
python src/data_preprocessing.py
```

2. **Model Training**:
```bash
python -m src.train_models  # Creates artifacts in results/models/
```

3. **Evaluation**:
```bash
python -m src.evaluate_models  # Generates metrics and visualizations
```

## Project Conventions

1. **Data Organization**:
   - Raw data in `data/raw/`
   - Processed datasets in `data/processed/`
   - Results organized by type (metrics/plots/models)

2. **Coding Patterns**:
   - Target variable is 'hltprhc' (heart condition)
   - Random seed fixed at 42 for reproducibility
   - Standard train/val/test workflow with no data leakage
   - Extensive metrics logging and visualization

3. **Dependencies**:
   - Core ML: pandas, numpy, scikit-learn, xgboost
   - Deep Learning: PyTorch
   - Visualization: matplotlib, seaborn
   - Version ranges specified in requirements.txt

## Integration Points

1. **Model Loading**:
```python
from src.utils import load_model
model = load_model('results/models/random_forest.joblib')
```

2. **Data Preprocessing**:
```python
from src.data_preprocessing import load_dataset
df = load_dataset()  # Returns cleaned DataFrame
```

3. **Evaluation Pipeline**:
```python
from src.evaluate_models import evaluate_models
metrics_df = evaluate_models()  # Runs full suite of metrics
```

## Work in Progress

1. **Explainability Module** (`src/explainability.py`):
   - Implementation pending for local explainability methods
   - Will integrate with existing model predictions

2. **Demo Application** (`demo/app_gradio.py`):
   - Gradio interface implementation planned
   - Will expose model predictions and explanations

3. **Docker Deployment**:
   - Basic configuration in `docker/`
   - Requirements to be finalized

## Common Gotchas

1. Always use absolute paths via:
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

2. Handle data splitting consistently:
```python
from src.train_models import stratified_split
X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
```

3. Model evaluation requires all artifacts:
   - StandardScaler
   - Trained models
   - Neural network config
   - Data splits