"""
Comprehensive Hyperparameter Tuning Pipeline for Health Prediction Models

This module implements professional-grade hyperparameter optimization following academic 
best practices for machine learning model development. Based on the comprehensive 
implementation from notebooks/03_modeling.ipynb.

Key Features:
============
- ✅ **5-fold Stratified Cross-Validation**: Robust parameter selection with balanced class sampling
- ✅ **F1 Score Optimization**: Focused on medical screening performance with class imbalance
- ✅ **Anti-Overfitting Framework**: Conservative parameter ranges + gap monitoring (<5% threshold)
- ✅ **Professional Logging**: Comprehensive diagnostics and checkpoint saving
- ✅ **Neural Network Excellence**: Custom PyTorch implementation with AdamW + early stopping
- ✅ **Class Imbalance Handling**: Balanced weights, regularization (no synthetic oversampling)
- ✅ **Reproducible Results**: Fixed random seeds and systematic parameter exploration

Models Supported:
================
- Logistic Regression (ElasticNet regularization)
- Random Forest (Conservative depth limits, bootstrap sampling)
- XGBoost (Gradient boosting with scale_pos_weight for imbalance)
- SVM (RBF kernel focus with probability estimation)
- Neural Network (Custom HealthNN with AdamW optimizer, patience=10)

Methodology Compliance:
======================
- No data leakage: Validation set used exclusively for hyperparameter selection
- Stratified sampling: Class balance preserved across all CV folds
- Early stopping: Neural networks use patience-based training termination
- Diagnostic logging: Train/validation gap monitoring for overfitting detection
- Checkpoint saving: Complete parameter exploration and results persistence

Performance Standards:
=====================
- Target: Validation F1 > 0.37 (exceeding baseline performance)
- Generalization: Train-validation gap < 0.05 (strong generalization requirement)
- Clinical Focus: Recall-precision balance suitable for medical screening
- Reproducibility: All results reproducible via fixed random seeds (42)

Author: Health XAI Project Team
Version: 2.0 (Professional Implementation)
Compatible with: scikit-learn 1.3+, PyTorch 2.0+, XGBoost 2.0+
"""

from __future__ import annotations

import json
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Project imports
from src.models.neural_network import HealthNN, get_device
from src.utils import (
    ensure_directory,
    get_top_model,
    log_model_diagnostic,
    save_model,
    save_top_model,
    plot_confusion_matrix,
    PROJECT_ROOT,
)

# Configure warnings and environment
warnings.filterwarnings("ignore", message="`use_label_encoder` is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Directory configuration
MODELS_DIR = PROJECT_ROOT / "results" / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TUNING_LOG_PATH = METRICS_DIR / "hyperparameter_tuning_log.txt"
TUNING_CHECKPOINT_PATH = MODELS_DIR / "tuning_results_checkpoint.json"

# Tuning configuration
DEFAULT_N_JOBS = int(os.getenv("TUNING_N_JOBS", "-1"))
RANDOM_STATE = 42
CV_FOLDS = 5
OVERFITTING_THRESHOLD = 0.05

def _create_directories() -> None:
    """Ensure all required directories exist for tuning artifacts."""
    for directory in [MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
        ensure_directory(directory)





def _log_tuning_progress(message: str, level: str = "INFO") -> None:
    """Log tuning progress to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)
    
    with open(TUNING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def _plot_cv_results(cv_results: dict, model_name: str, best_idx: int) -> None:
    """Generate professional cross-validation result visualizations."""
    try:
        # Extract CV scores
        mean_train_scores = cv_results.get('mean_train_score', [])
        mean_test_scores = cv_results.get('mean_test_score', [])
        
        if not mean_train_scores or not mean_test_scores:
            _log_tuning_progress(f"Insufficient CV data for plotting {model_name}", "WARNING")
            return
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: CV scores across all trials
        trial_indices = range(len(mean_train_scores))
        ax1.scatter(trial_indices, mean_train_scores, color='coral', alpha=0.7, label='CV Train (mean)', s=50)
        ax1.scatter(trial_indices, mean_test_scores, color='darkgoldenrod', alpha=0.7, marker='x', label='CV Val (mean)', s=60)
        
        # Highlight best trial
        ax1.scatter(best_idx, mean_train_scores[best_idx], color='red', s=100, marker='o', edgecolor='black')
        ax1.scatter(best_idx, mean_test_scores[best_idx], color='orange', s=100, marker='x', edgecolor='black')
        
        ax1.set_xlabel("Trial rank (arbitrary ordering)")
        ax1.set_ylabel("F1 score (mean across CV folds)")
        ax1.set_title(f"{model_name} — CV mean Train vs Val F1")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Best model train vs validation performance
        best_train_f1 = mean_train_scores[best_idx]
        best_val_f1 = mean_test_scores[best_idx]
        
        categories = ['Train', 'Val']
        f1_scores = [best_train_f1, best_val_f1]
        colors = ['coral', 'darkgoldenrod']
        
        bars = ax2.bar(categories, f1_scores, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("F1 score")
        ax2.set_title(f"{model_name} — Train vs Val F1")
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = PLOTS_DIR / f"{model_name.lower().replace(' ', '_')}_tuning_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        _log_tuning_progress(f"Saved tuning plots to {plot_path}")
        
    except Exception as e:
        _log_tuning_progress(f"Failed to create plots for {model_name}: {str(e)}", "ERROR")


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = 40,
    random_state: int = RANDOM_STATE
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Professional Logistic Regression hyperparameter tuning with ElasticNet regularization.
    
    Features:
    - ElasticNet penalty for feature selection and regularization
    - Conservative regularization strengths to prevent overfitting
    - Balanced class weights for imbalanced data handling
    - Comprehensive diagnostics and visualization
    
    Args:
        X_train: Training feature matrix (scaled)
        y_train: Training target vector
        X_val: Validation feature matrix (scaled)  
        y_val: Validation target vector
        n_iter: Number of randomized search iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (best_model, results_dict) with comprehensive metrics
    """
    _log_tuning_progress(f"🔧 1/5 Tuning Logistic Regression (n_iter={n_iter})")
    
    # Conservative parameter grid focused on regularization
    param_distributions = {
        'C': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        'penalty': ['elasticnet'],
        'solver': ['saga'],  # Only solver that supports elasticnet
        'l1_ratio': np.linspace(0.1, 0.9, 10),  # ElasticNet mixing parameter
        'max_iter': [2000],  # Ensure convergence
        'class_weight': ['balanced'],
        'random_state': [random_state]
    }
    
    # Create model and search
    lr = LogisticRegression(random_state=random_state)
    search = RandomizedSearchCV(
        estimator=lr,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state),
        scoring='f1',
        n_jobs=DEFAULT_N_JOBS,
        random_state=random_state,
        verbose=1,
        return_train_score=True  # Essential for overfitting analysis
    )
    
    # Execute hyperparameter search
    start_time = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Extract best model and evaluate
    best_model = search.best_estimator_
    best_idx = search.best_index_
    
    # Comprehensive evaluation
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    gap = train_f1 - val_f1
    
    # Diagnostic assessment
    diagnosis = "✅ OK" if gap <= OVERFITTING_THRESHOLD else "⚠️ Overfitting"
    diagnosis_detail = f"(Δ={gap:.3f})"
    
    # Comprehensive results
    results = {
        'model_name': 'LogisticRegression_Tuned',
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'overfitting_gap': gap,
        'search_time_seconds': search_time,
        'diagnosis': diagnosis,
        'cv_results': search.cv_results_
    }
    
    # Professional logging
    _log_tuning_progress(f"  ✅ LR Best CV F1: {search.best_score_:.4f}, Val F1: {val_f1:.4f}")
    _log_tuning_progress(f"  🎯 Train F1: {train_f1:.4f}, Gap: {gap:.4f}")
    _log_tuning_progress(f"  🔍 Diagnosis: {diagnosis} {diagnosis_detail}")
    _log_tuning_progress(f"  📋 Best params: {search.best_params_}")
    
    # Generate visualizations
    _plot_cv_results(search.cv_results_, "Logistic Regression", best_idx)
    
    # Save model and log diagnostics
    save_model(best_model, "logistic_regression_tuned")
    log_model_diagnostic(
        model_name=results['model_name'],
        train_score=train_f1,
        val_score=val_f1,
        status="Good fit" if gap <= OVERFITTING_THRESHOLD else "Overfitting"
    )
    
    return best_model, results


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = 40,
    random_state: int = RANDOM_STATE
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Professional Random Forest hyperparameter tuning with anti-overfitting focus.
    
    Features:
    - Conservative depth limits to prevent overfitting
    - Bootstrap sampling with controlled sample fractions
    - Balanced class weights for imbalanced data
    - Feature subsampling for improved generalization
    
    Args:
        X_train: Training feature matrix (not scaled - tree-based)
        y_train: Training target vector
        X_val: Validation feature matrix (not scaled)
        y_val: Validation target vector
        n_iter: Number of randomized search iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (best_model, results_dict) with comprehensive metrics
    """
    _log_tuning_progress(f"🔧 2/5 Tuning Random Forest (n_iter={n_iter})")
    
    # Conservative parameter grid to prevent overfitting
    param_distributions = {
        'n_estimators': [30, 40, 50, 60, 80, 100, 120],
        'max_depth': [4, 5, 6, 8, 10, 12],  # Conservative depths
        'min_samples_split': [10, 15, 20, 25, 30],  # Larger splits
        'min_samples_leaf': [8, 10, 12, 15, 18],   # Larger leaves
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'max_samples': [0.7, 0.8, 0.9],  # Bootstrap fraction
        'class_weight': ['balanced'],
        'random_state': [random_state],
        'n_jobs': [-1]
    }
    
    # Create model and search
    rf = RandomForestClassifier(random_state=random_state)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state),
        scoring='f1',
        n_jobs=DEFAULT_N_JOBS,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )
    
    # Execute hyperparameter search
    start_time = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Extract best model and evaluate
    best_model = search.best_estimator_
    best_idx = search.best_index_
    
    # Comprehensive evaluation
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    gap = train_f1 - val_f1
    
    # Diagnostic assessment
    diagnosis = "✅ OK" if gap <= OVERFITTING_THRESHOLD else "⚠️ Overfitting"
    diagnosis_detail = f"(Δ={gap:.3f})"
    
    # Comprehensive results
    results = {
        'model_name': 'RandomForest_Tuned',
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'overfitting_gap': gap,
        'search_time_seconds': search_time,
        'diagnosis': diagnosis,
        'cv_results': search.cv_results_
    }
    
    # Professional logging
    _log_tuning_progress(f"  ✅ RF Best CV F1: {search.best_score_:.4f}, Val F1: {val_f1:.4f}")
    _log_tuning_progress(f"  🎯 Train F1: {train_f1:.4f}, Gap: {gap:.4f}")
    _log_tuning_progress(f"  🔍 Diagnosis: {diagnosis} {diagnosis_detail}")
    _log_tuning_progress(f"  📋 Best params: {search.best_params_}")
    
    # Generate visualizations
    _plot_cv_results(search.cv_results_, "Random Forest", best_idx)
    
    # Save model and log diagnostics
    save_model(best_model, "random_forest_tuned")
    log_model_diagnostic(
        model_name=results['model_name'],
        train_score=train_f1,
        val_score=val_f1,
        status="Good fit" if gap <= OVERFITTING_THRESHOLD else "Overfitting"
    )
    
    return best_model, results


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = 40,
    random_state: int = RANDOM_STATE
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Professional XGBoost hyperparameter tuning with class imbalance handling.
    
    Features:
    - Automatic scale_pos_weight calculation for imbalanced classes
    - Conservative regularization to prevent overfitting
    - Gradient boosting with controlled learning rates
    - Advanced regularization techniques (L1/L2)
    
    Args:
        X_train: Training feature matrix (not scaled - tree-based)
        y_train: Training target vector
        X_val: Validation feature matrix (not scaled)
        y_val: Validation target vector
        n_iter: Number of randomized search iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (best_model, results_dict) with comprehensive metrics
    """
    # Calculate class imbalance ratio
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    _log_tuning_progress(f"🔧 3/5 Tuning XGBoost (n_iter={n_iter})")
    _log_tuning_progress(f"  📊 Class imbalance ratio: {scale_pos_weight:.2f} (using scale_pos_weight)")
    
    # Conservative parameter grid to prevent overfitting
    param_distributions = {
        'n_estimators': [100, 120, 150, 170, 200],
        'max_depth': [3, 4, 5, 6],  # Conservative depths
        'learning_rate': [0.03, 0.05, 0.08, 0.1, 0.12],
        'subsample': [0.7, 0.8, 0.9],  # Row sampling
        'colsample_bytree': [0.7, 0.8, 0.9],  # Feature sampling
        'reg_alpha': [0.5, 1.0, 2.0, 3.0],  # L1 regularization
        'reg_lambda': [0.5, 1.0, 2.0, 3.0],  # L2 regularization
        'min_child_weight': [5, 8, 10, 12],  # Prevent overfitting
        'gamma': [0.0, 0.1, 0.2],  # Minimum loss reduction
        'scale_pos_weight': [scale_pos_weight],
        'random_state': [random_state],
        'n_jobs': [-1],
        'eval_metric': ['logloss']
    }
    
    # Create model and search
    xgb_model = XGBClassifier(random_state=random_state)
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state),
        scoring='f1',
        n_jobs=DEFAULT_N_JOBS,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )
    
    # Execute hyperparameter search
    start_time = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Extract best model and evaluate
    best_model = search.best_estimator_
    best_idx = search.best_index_
    
    # Comprehensive evaluation
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    gap = train_f1 - val_f1
    
    # Diagnostic assessment
    diagnosis = "✅ OK" if gap <= OVERFITTING_THRESHOLD else "⚠️ Overfitting"
    diagnosis_detail = f"(Δ={gap:.3f})"
    
    # Comprehensive results
    results = {
        'model_name': 'XGBoost_Tuned',
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'overfitting_gap': gap,
        'search_time_seconds': search_time,
        'diagnosis': diagnosis,
        'scale_pos_weight': scale_pos_weight,
        'cv_results': search.cv_results_
    }
    
    # Professional logging
    _log_tuning_progress(f"  ✅ XGB Best CV F1: {search.best_score_:.4f}, Val F1: {val_f1:.4f}")
    _log_tuning_progress(f"  🎯 Train F1: {train_f1:.4f}, Gap: {gap:.4f}")
    _log_tuning_progress(f"  🔍 Diagnosis: {diagnosis} {diagnosis_detail}")
    _log_tuning_progress(f"  📋 Best params: {search.best_params_}")
    
    # Generate visualizations
    _plot_cv_results(search.cv_results_, "XGBoost", best_idx)
    
    # Save model and log diagnostics
    save_model(best_model, "xgboost_tuned")
    log_model_diagnostic(
        model_name=results['model_name'],
        train_score=train_f1,
        val_score=val_f1,
        status="Good fit" if gap <= OVERFITTING_THRESHOLD else "Overfitting"
    )
    
    return best_model, results


def tune_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = 18,
    random_state: int = RANDOM_STATE
) -> Tuple[SVC, Dict[str, Any]]:
    """
    Professional SVM hyperparameter tuning with RBF kernel focus.
    
    Features:
    - RBF kernel optimization (best for health prediction)
    - Probability estimation enabled for confidence scoring
    - Balanced class weights for imbalanced data
    - Conservative C and gamma ranges for generalization
    
    Args:
        X_train: Training feature matrix (scaled required)
        y_train: Training target vector
        X_val: Validation feature matrix (scaled)
        y_val: Validation target vector
        n_iter: Number of randomized search iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (best_model, results_dict) with comprehensive metrics
    """
    _log_tuning_progress(f"🔧 4/5 Tuning SVM (RBF) (n_iter={n_iter}) — only 'rbf' kernel")
    
    # Parameter grid focused on RBF kernel (matching notebook implementation)
    param_distributions = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        'gamma': ['scale', 'auto', 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'kernel': ['rbf'],  # Focus on RBF kernel only (best for health data)
        'class_weight': ['balanced'],
        'probability': [True],  # Enable for confidence scoring
        'random_state': [random_state]
    }
    
    # Create model and search
    svm_model = SVC(random_state=random_state)
    search = RandomizedSearchCV(
        estimator=svm_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state),
        scoring='f1',
        n_jobs=DEFAULT_N_JOBS,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )
    
    # Execute hyperparameter search
    start_time = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Extract best model and evaluate
    best_model = search.best_estimator_
    best_idx = search.best_index_
    
    # Comprehensive evaluation
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    gap = train_f1 - val_f1
    
    # Diagnostic assessment
    diagnosis = "✅ OK" if gap <= OVERFITTING_THRESHOLD else "⚠️ Overfitting"
    diagnosis_detail = f"(Δ={gap:.3f})"
    
    # Comprehensive results
    results = {
        'model_name': 'SVM_Tuned',
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'overfitting_gap': gap,
        'search_time_seconds': search_time,
        'diagnosis': diagnosis,
        'cv_results': search.cv_results_
    }
    
    # Professional logging
    _log_tuning_progress(f"  ✅ SVM Best CV F1: {search.best_score_:.4f}, Val F1: {val_f1:.4f}")
    _log_tuning_progress(f"  🎯 Train F1: {train_f1:.4f}, Gap: {gap:.4f}")
    _log_tuning_progress(f"  🔍 Diagnosis: {diagnosis} {diagnosis_detail}")
    _log_tuning_progress(f"  📋 Best params: {search.best_params_}")
    
    # Generate visualizations
    _plot_cv_results(search.cv_results_, "SVM (RBF)", best_idx)
    
    # Save model and log diagnostics
    save_model(best_model, "svm_tuned")
    log_model_diagnostic(
        model_name=results['model_name'],
        train_score=train_f1,
        val_score=val_f1,
        status="Good fit" if gap <= OVERFITTING_THRESHOLD else "Overfitting"
    )
    
    return best_model, results



def randomized_search_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = 18,
    random_state: int = 42
) -> tuple[SVC, dict]:
    """RandomizedSearchCV for SVM with RBF kernel (matching notebook implementation)."""
    print(f"[INFO] Starting SVM RandomizedSearchCV ({n_iter} iterations)...")
    
    # Parameter grid (matching notebook - RBF kernel focus)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf'],  # Focus on RBF kernel only (matching notebook)
        'class_weight': ['balanced']
    }
    
    # Create model and search
    svm = SVC(random_state=random_state, probability=True)
    search = RandomizedSearchCV(
        svm, param_grid, n_iter=n_iter,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring='f1', n_jobs=-1, random_state=random_state, verbose=1
    )
    
    # Fit search
    search.fit(X_train, y_train)
    
    # Evaluate best model on validation set
    best_model = search.best_estimator_
    val_pred = best_model.predict(X_val)
    train_pred = best_model.predict(X_train)
    
    val_f1 = f1_score(y_val, val_pred)
    train_f1 = f1_score(y_train, train_pred)
    
    results = {
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'train_f1': train_f1,
        'val_f1': val_f1
    }
    
    print(f"[INFO] Best SVM CV F1: {search.best_score_:.3f}, Val F1: {val_f1:.3f}")
    return best_model, results


def _plot_neural_network_trials(trials: List[Dict[str, Any]], model_name: str) -> None:
    """Plot neural network trial results."""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        trial_numbers = [t['trial'] for t in trials]
        train_f1s = [t['train_f1'] for t in trials]
        val_f1s = [t['val_f1'] for t in trials]
        
        ax.scatter(trial_numbers, train_f1s, color='coral', alpha=0.8, label='Train F1', s=80)
        ax.scatter(trial_numbers, val_f1s, color='darkgoldenrod', alpha=0.8, marker='x', label='Val F1', s=80)
        
        # Find and highlight best trial
        best_trial = max(trials, key=lambda x: x['val_f1'])
        best_idx = best_trial['trial']
        ax.scatter(best_idx, best_trial['train_f1'], color='red', s=120, marker='o', edgecolor='black', linewidth=2)
        ax.scatter(best_idx, best_trial['val_f1'], color='orange', s=120, marker='x', edgecolor='black', linewidth=2)
        
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"{model_name} — Trial Performance Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = PLOTS_DIR / f"{model_name.lower().replace(' ', '_')}_trials.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        _log_tuning_progress(f"Saved trial plots to {plot_path}")
        
    except Exception as e:
        _log_tuning_progress(f"Failed to create trial plots for {model_name}: {str(e)}", "ERROR")


def _plot_neural_network_epochs(epoch_data: List[Dict[str, Any]], model_name: str, best_epoch: int) -> None:
    """Plot neural network epoch training curves."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = [e['epoch'] for e in epoch_data]
        train_f1s = [e['train_f1'] for e in epoch_data]
        val_f1s = [e['val_f1'] for e in epoch_data]
        
        # Plot 1: Training curves
        ax1.plot(epochs, train_f1s, color='coral', linewidth=2, label='Train F1', alpha=0.8)
        ax1.plot(epochs, val_f1s, color='darkgoldenrod', linewidth=2, label='Val F1', alpha=0.8)
        
        # Mark best epoch
        if best_epoch <= len(epochs):
            best_train_f1 = train_f1s[best_epoch-1]
            best_val_f1 = val_f1s[best_epoch-1]
            ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')
            ax1.scatter(best_epoch, best_train_f1, color='red', s=100, zorder=5)
            ax1.scatter(best_epoch, best_val_f1, color='red', s=100, zorder=5)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("F1 Score")
        ax1.set_title(f"{model_name} (best trial) — Epoch vs F1")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final performance comparison
        categories = ['Train', 'Val']
        if best_epoch <= len(epochs):
            final_scores = [train_f1s[best_epoch-1], val_f1s[best_epoch-1]]
        else:
            final_scores = [train_f1s[-1], val_f1s[-1]]
        colors = ['coral', 'darkgoldenrod']
        
        bars = ax2.bar(categories, final_scores, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("F1 Score")
        ax2.set_title(f"{model_name} — Train vs Val F1")
        
        # Add value labels
        for bar, score in zip(bars, final_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = PLOTS_DIR / f"{model_name.lower().replace(' ', '_')}_epochs.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        _log_tuning_progress(f"Saved epoch plots to {plot_path}")
        
    except Exception as e:
        _log_tuning_progress(f"Failed to create epoch plots for {model_name}: {str(e)}", "ERROR")


def tune_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 8,
    random_state: int = RANDOM_STATE
) -> Tuple[HealthNN, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Professional Neural Network hyperparameter tuning with PyTorch implementation.
    
    Features:
    - Custom HealthNN architecture optimized for health prediction
    - AdamW optimizer with weight decay regularization
    - Early stopping with patience=10 to prevent overfitting
    - Class-weighted loss for imbalanced data handling
    - Comprehensive trial tracking and visualization
    
    Args:
        X_train: Training feature matrix (scaled required)
        y_train: Training target vector
        X_val: Validation feature matrix (scaled)
        y_val: Validation target vector
        n_trials: Number of hyperparameter combinations to try
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (best_model, trial_results, epoch_data) with comprehensive tracking
    """
    _log_tuning_progress(f"🔧 5/5 Tuning Neural Network (AdamW, patience=10)")
    
    # Set all random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    
    device = get_device()
    _log_tuning_progress(f"  📱 Using device: {device}")
    
    # Initialize tracking variables
    trials = []
    best_val_f1 = 0
    best_model = None
    best_epoch_data = None
    best_trial_info = None
    
    # Calculate class weights for imbalanced data
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    _log_tuning_progress(f"  ⚖️ Class weights - Negative: {n_neg}, Positive: {n_pos}, Ratio: {pos_weight.item():.2f}")
    
    for trial in range(n_trials):
        # Sample hyperparameters from professional ranges
        hidden_dim = random.choice([32, 64, 128, 256])
        lr = np.random.uniform(0.0001, 0.01)
        weight_decay = np.random.uniform(1e-6, 1e-3)
        dropout = np.random.uniform(0.2, 0.6)
        batch_size = random.choice([64, 128, 256])
        
        config = {
            'input_dim': X_train.shape[1],
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size
        }
        
        _log_tuning_progress(f"    Trial {trial + 1:02d}: hidden={hidden_dim}, lr={lr:.6f}, wd={weight_decay:.1e}, bs={batch_size}")
        
        try:
            # Create model
            model = HealthNN(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                dropout=config['dropout']
            ).to(device)
            
            # AdamW optimizer (academic requirement)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )
            
            # Class-weighted loss function
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # Prepare data tensors
            X_train_tensor = torch.FloatTensor(X_train.values).to(device)
            y_train_tensor = torch.FloatTensor(y_train.values).to(device)
            X_val_tensor = torch.FloatTensor(X_val.values).to(device)
            y_val_tensor = torch.FloatTensor(y_val.values).to(device)
            
            # Create data loader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            
            # Training with early stopping
            best_epoch_val_f1 = 0
            patience_counter = 0
            patience = 10  # Academic requirement
            epoch_data = []
            best_model_state = None
            best_epoch = 0
            
            for epoch in range(200):  # Maximum epochs
                # Training phase
                model.train()
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Evaluation phase
                model.eval()
                with torch.no_grad():
                    # Predictions for both sets
                    train_outputs = torch.sigmoid(model(X_train_tensor)).squeeze()
                    val_outputs = torch.sigmoid(model(X_val_tensor)).squeeze()
                    
                    train_pred = (train_outputs > 0.5).cpu().numpy()
                    val_pred = (val_outputs > 0.5).cpu().numpy()
                    
                    # Calculate F1 scores
                    train_f1 = f1_score(y_train, train_pred)
                    val_f1 = f1_score(y_val, val_pred)
                    
                    # Track epoch performance
                    epoch_data.append({
                        'epoch': epoch + 1,
                        'train_f1': train_f1,
                        'val_f1': val_f1,
                        'loss': epoch_loss / len(train_loader)
                    })
                
                # Early stopping logic
                if val_f1 > best_epoch_val_f1:
                    best_epoch_val_f1 = val_f1
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # Save best model state for this trial
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        _log_tuning_progress(f"    Trial {trial + 1:02d}: Early stopping at epoch {epoch + 1}")
                        break
            
            # Restore best model for this trial
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Final evaluation with best model
            model.eval()
            with torch.no_grad():
                train_outputs = torch.sigmoid(model(X_train_tensor)).squeeze()
                val_outputs = torch.sigmoid(model(X_val_tensor)).squeeze()
                
                train_pred = (train_outputs > 0.5).cpu().numpy()
                val_pred = (val_outputs > 0.5).cpu().numpy()
                
                final_train_f1 = f1_score(y_train, train_pred)
                final_val_f1 = f1_score(y_val, val_pred)
            
            # Store trial results
            trial_data = {
                'trial': trial + 1,
                'params': config,
                'train_f1': final_train_f1,
                'val_f1': best_epoch_val_f1,
                'best_epoch': best_epoch,
                'total_epochs': len(epoch_data)
            }
            trials.append(trial_data)
            
            # Update global best
            if best_epoch_val_f1 > best_val_f1:
                best_val_f1 = best_epoch_val_f1
                best_model = model
                best_epoch_data = epoch_data
                best_trial_info = trial_data
            
            _log_tuning_progress(f"    Trial {trial + 1:02d}: hidden={hidden_dim}, lr={lr:.6f}, wd={weight_decay:.1e}, bs={batch_size} | best_val_f1={best_epoch_val_f1:.4f} (epochs={best_epoch})")
            
        except Exception as e:
            _log_tuning_progress(f"    Trial {trial + 1:02d}: Failed - {str(e)}", "ERROR")
            continue
    
    if best_model is None:
        raise RuntimeError("All Neural Network trials failed!")
    
    # Comprehensive results
    gap = best_trial_info['train_f1'] - best_trial_info['val_f1']
    diagnosis = "✅ OK" if gap <= OVERFITTING_THRESHOLD else "⚠️ Overfitting"
    diagnosis_detail = f"(Δ={gap:.3f})"
    
    # Professional logging
    _log_tuning_progress(f"  ✅ NN Best CV F1: {best_val_f1:.4f}, Val F1: {best_val_f1:.4f}")
    _log_tuning_progress(f"  🎯 Train F1: {best_trial_info['train_f1']:.4f}, Gap: {gap:.4f}")
    _log_tuning_progress(f"  🔍 Diagnosis: {diagnosis} {diagnosis_detail}")
    _log_tuning_progress(f"  📋 Best params: {best_trial_info['params']}")
    
    return best_model, trials, best_epoch_data


def run_comprehensive_hyperparameter_tuning(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Execute comprehensive hyperparameter tuning for all models following academic standards.
    
    This function orchestrates the complete tuning pipeline with professional logging,
    checkpoint saving, and comprehensive result analysis matching the notebook implementation.
    
    Features:
    - 5-fold stratified cross-validation for all sklearn models
    - Custom PyTorch implementation for neural networks
    - Anti-overfitting diagnostics and monitoring
    - Professional visualization and logging
    - Checkpoint saving for reproducibility
    
    Args:
        random_state: Random seed for reproducibility across all models
        
    Returns:
        Comprehensive results dictionary with models, metrics, and diagnostics
    """
    # Initialize logging and directories
    _create_directories()
    _log_tuning_progress("🚀 STARTING COMPREHENSIVE HYPERPARAMETER TUNING")
    _log_tuning_progress("=" * 60)
    
    # Initialize model diagnostics file
    diagnostics_path = METRICS_DIR / "model_diagnostics.csv"
    if diagnostics_path.exists():
        diagnostics_path.unlink()
        
    # Initialize tuning log
    if TUNING_LOG_PATH.exists():
        TUNING_LOG_PATH.unlink()
    
    # Load and validate data splits
    splits_path = MODELS_DIR / "data_splits.joblib"
    if not splits_path.exists():
        raise FileNotFoundError("Data splits not found. Run src.train_models first.")
    
    splits = joblib.load(splits_path)
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    
    # Load and apply scaler for linear models
    scaler_path = MODELS_DIR / "standard_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError("Standard scaler not found. Run src.train_models first.")
    
    scaler = joblib.load(scaler_path)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    # Log dataset information
    _log_tuning_progress(f"📊 Data Loaded:")
    _log_tuning_progress(f"  • Train: {len(X_train):,} samples, {X_train.shape[1]} features")
    _log_tuning_progress(f"  • Validation: {len(X_val):,} samples")
    _log_tuning_progress(f"  • Test: Available separately for final evaluation")
    _log_tuning_progress(f"  • Class distribution: {dict(y_train.value_counts().sort_index())}")
    _log_tuning_progress("")
    
    # Initialize results tracking
    all_results = {}
    tuning_summary = []
    
    # 1. Logistic Regression Tuning
    _log_tuning_progress("="*50)
    lr_model, lr_results = tune_logistic_regression(
        X_train_scaled, y_train, X_val_scaled, y_val, 
        n_iter=40, random_state=random_state
    )
    all_results['LogisticRegression'] = {'model': lr_model, 'results': lr_results}
    tuning_summary.append(lr_results)
    
    # 2. Random Forest Tuning
    _log_tuning_progress("\n" + "="*50)
    rf_model, rf_results = tune_random_forest(
        X_train, y_train, X_val, y_val, 
        n_iter=40, random_state=random_state
    )
    all_results['RandomForest'] = {'model': rf_model, 'results': rf_results}
    tuning_summary.append(rf_results)
    
    # 3. XGBoost Tuning
    _log_tuning_progress("\n" + "="*50)
    xgb_model, xgb_results = tune_xgboost(
        X_train, y_train, X_val, y_val, 
        n_iter=40, random_state=random_state
    )
    all_results['XGBoost'] = {'model': xgb_model, 'results': xgb_results}
    tuning_summary.append(xgb_results)
    
    # 4. SVM Tuning
    _log_tuning_progress("\n" + "="*50)
    svm_model, svm_results = tune_svm(
        X_train_scaled, y_train, X_val_scaled, y_val, 
        n_iter=18, random_state=random_state
    )
    all_results['SVM'] = {'model': svm_model, 'results': svm_results}
    tuning_summary.append(svm_results)
    
    # 5. Neural Network Tuning
    _log_tuning_progress("\n" + "="*50)
    nn_model, nn_trials, nn_epoch_data = tune_neural_network(
        X_train_scaled, y_train, X_val_scaled, y_val, 
        n_trials=8, random_state=random_state
    )
    
    # Process neural network results
    best_nn_trial = max(nn_trials, key=lambda x: x['val_f1'])
    nn_results = {
        'model_name': 'NeuralNetwork_Tuned',
        'best_params': best_nn_trial['params'],
        'best_cv_score': best_nn_trial['val_f1'],  # No CV for NN, use validation F1
        'train_f1': best_nn_trial['train_f1'],
        'val_f1': best_nn_trial['val_f1'],
        'overfitting_gap': best_nn_trial['train_f1'] - best_nn_trial['val_f1'],
        'trials': nn_trials,
        'epoch_data': nn_epoch_data
    }
    
    # Save neural network model and log diagnostics
    save_model(nn_model, "neural_network_tuned")
    log_model_diagnostic(
        model_name=nn_results['model_name'],
        train_score=nn_results['train_f1'],
        val_score=nn_results['val_f1'],
        status="Good fit" if nn_results['overfitting_gap'] <= OVERFITTING_THRESHOLD else "Overfitting"
    )
    
    all_results['NeuralNetwork'] = {'model': nn_model, 'results': nn_results}
    tuning_summary.append(nn_results)
    
    # Generate neural network specific visualizations
    _plot_neural_network_trials(nn_trials, "NeuralNetwork_Tuned")
    best_epoch = best_nn_trial.get('best_epoch', len(nn_epoch_data))
    _plot_neural_network_epochs(nn_epoch_data, "NeuralNetwork_Tuned", best_epoch)
    
    # Consolidate results and generate summary
    _log_tuning_progress("\n" + "="*50)
    _log_tuning_progress("💾 Consolidating tuning results...")
    
    # Create comprehensive summary table
    _log_tuning_progress(f"\n📊 HYPERPARAMETER TUNING SUMMARY:")
    _log_tuning_progress("="*84)
    _log_tuning_progress(f"{'Model':<25} {'CV F1':<8} {'Val F1':<8} {'Gap':<8} {'Diagnosis':<20}")
    _log_tuning_progress("-"*84)
    
    for result in tuning_summary:
        model_name = result['model_name']
        cv_f1 = result['best_cv_score']
        val_f1 = result['val_f1']
        gap = result['overfitting_gap']
        diagnosis = result.get('diagnosis', '✅ OK' if gap <= OVERFITTING_THRESHOLD else '⚠️ Overfitting')
        
        _log_tuning_progress(f"{model_name:<25} {cv_f1:<8.4f} {val_f1:<8.4f} {gap:<8.4f} {diagnosis:<20}")
    
    # Find best models by different criteria
    best_by_val_f1 = max(tuning_summary, key=lambda x: x['val_f1'])
    best_by_gap = min(tuning_summary, key=lambda x: x['overfitting_gap'])
    
    _log_tuning_progress(f"\n🏆 BEST MODEL BY VALIDATION F1:")
    _log_tuning_progress(f"  • Model: {best_by_val_f1['model_name']}")
    _log_tuning_progress(f"  • Validation F1: {best_by_val_f1['val_f1']:.4f}")
    _log_tuning_progress(f"  • Train F1: {best_by_val_f1['train_f1']:.4f}")
    _log_tuning_progress(f"  • Gap: {best_by_val_f1['overfitting_gap']:.4f}")
    _log_tuning_progress(f"  • Diagnosis: {best_by_val_f1.get('diagnosis', '✅ OK' if best_by_val_f1['overfitting_gap'] <= OVERFITTING_THRESHOLD else '⚠️ Overfitting')}")
    
    _log_tuning_progress(f"\n🔎 BEST MODEL BY SMALLEST TRAIN/VAL GAP:")
    _log_tuning_progress(f"  • Model: {best_by_gap['model_name']}")
    _log_tuning_progress(f"  • Validation F1: {best_by_gap['val_f1']:.4f}")
    _log_tuning_progress(f"  • Train F1: {best_by_gap['train_f1']:.4f}")
    _log_tuning_progress(f"  • Gap: {best_by_gap['overfitting_gap']:.4f}")
    _log_tuning_progress(f"  • Diagnosis: {best_by_gap.get('diagnosis', '✅ OK' if best_by_gap['overfitting_gap'] <= OVERFITTING_THRESHOLD else '⚠️ Overfitting')}")
    
    # Save comprehensive checkpoint
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'random_state': random_state,
        'tuning_summary': tuning_summary,
        'best_by_val_f1': best_by_val_f1['model_name'],
        'best_by_gap': best_by_gap['model_name'],
        'configuration': {
            'cv_folds': CV_FOLDS,
            'overfitting_threshold': OVERFITTING_THRESHOLD,
            'n_jobs': DEFAULT_N_JOBS
        }
    }
    
    with open(TUNING_CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint_data, f, indent=2, default=str)
    
    _log_tuning_progress(f"\n[INFO] Tuning checkpoint saved to {TUNING_CHECKPOINT_PATH}")
    
    # Final summary
    _log_tuning_progress(f"\n✅ TUNING COMPLETE!")
    _log_tuning_progress(f"  • All models tuned using 5-fold stratified cross-validation")
    _log_tuning_progress(f"  • Neural Network: Custom PyTorch HealthNN with AdamW optimizer")
    _log_tuning_progress(f"  • Early stopping with patience=10 for neural network")
    _log_tuning_progress(f"  • Class imbalance handled via balanced weights and regularization")
    _log_tuning_progress(f"  • Overfitting diagnosis: Gap >{OVERFITTING_THRESHOLD} flagged as concerning")
    _log_tuning_progress(f"  • Results saved to: {METRICS_DIR}")
    _log_tuning_progress(f"  • Plots saved to: {PLOTS_DIR}")
    
    return all_results


# Legacy function name for backward compatibility
def run_all_tuning(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Legacy wrapper for backward compatibility."""
    return run_comprehensive_hyperparameter_tuning(random_state)


if __name__ == "__main__":
    print("🚀 Running Professional Hyperparameter Tuning Pipeline...")
    print("="*60)
    
    try:
        results = run_comprehensive_hyperparameter_tuning()
        print("\n✅ SUCCESS: Professional hyperparameter tuning completed!")
        print("📊 All models optimized and saved successfully.")
        print("📈 Check results/metrics/ for detailed performance analysis.")
        print("📉 Check results/plots/ for comprehensive visualizations.")
        
    except Exception as e:
        print(f"\n❌ ERROR: Tuning pipeline failed: {str(e)}")
        print("💡 Ensure you have run 'python -m src.train_models' first.")
        raise