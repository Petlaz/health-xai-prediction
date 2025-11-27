"""
Week 5-6: Professional XAI Implementation for Healthcare Decision Support

This module implements comprehensive explainable AI (XAI) for the Random Forest Tuned model
with healthcare-focused interpretability, clinical decision support templates, and 
automated explanation generation pipeline.

Focus: LIME and SHAP integration with clinical interpretation guidelines.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# XAI libraries
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
XAI_OUTPUT_DIR = RESULTS_DIR / 'explainability'
CLINICAL_OUTPUT_DIR = RESULTS_DIR / 'explanations'

# Clinical risk classification thresholds
RISK_THRESHOLDS = {
    'high_risk': 0.7,      # >70% predicted probability
    'medium_risk': 0.3,    # 30-70% predicted probability
    'low_risk': 0.0        # <30% predicted probability
}


@dataclass
class XAIConfig:
    """Configuration for XAI analysis pipeline"""
    model_name: str = 'random_forest_tuned'
    dataset_split: str = 'validation'  # 'validation' or 'test'
    sample_size: int = 200
    n_lime_features: int = 10
    random_state: int = 42
    save_plots: bool = True
    generate_clinical_reports: bool = True


class HealthcareXAIAnalyzer:
    """
    Professional XAI analyzer for healthcare decision support
    
    Implements SHAP and LIME explanations with clinical interpretation
    and automated report generation for healthcare practitioners.
    """
    
    def __init__(self, config: XAIConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_sample = None
        self.y_sample = None
        self.feature_names = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.shap_values = None
        self.expected_value = None
        
        # Ensure output directories exist
        XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CLINICAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Clinical domain mapping
        self.clinical_domains = {
            'health': 'Self-Reported Health Status',
            'dosprt': 'Physical Activity Frequency', 
            'bmi': 'Body Mass Index',
            'slprl': 'Sleep Quality & Relaxation',
            'flteeff': 'Emotional Wellbeing (Effectiveness)',
            'fltdpr': 'Mental Health (Depression Symptoms)',
            'cgtsmok': 'Smoking Behavior',
            'enjlf': 'Life Satisfaction',
            'fltlnl': 'Social Isolation (Loneliness)',
            'alcfreq': 'Alcohol Consumption',
            'ctrlife': 'Life Control & Autonomy',
            'happy': 'Happiness & Mood',
            'gndr': 'Gender Demographics',
            'age': 'Age Factor',
            'inprdsc': 'Social Interaction Quality'
        }
    
    def load_data_and_model(self) -> bool:
        """Load processed data and trained model"""
        try:
            print("📂 Loading data and model...")
            
            # Load data splits
            if self.config.dataset_split == 'validation':
                data = pd.read_csv(DATA_DIR / 'validation.csv')
            else:
                data = pd.read_csv(DATA_DIR / 'test.csv')
            
            train_data = pd.read_csv(DATA_DIR / 'train.csv')
            
            # Prepare features and targets
            feature_cols = [col for col in data.columns if col not in ['target', 'hltprhc']]
            target_col = 'target' if 'target' in data.columns else 'hltprhc'
            
            self.X_train = train_data[feature_cols]
            X_full = data[feature_cols]
            y_full = data[target_col]
            
            # Sample data for analysis
            np.random.seed(self.config.random_state)
            sample_size = min(self.config.sample_size, len(X_full))
            sample_indices = np.random.choice(len(X_full), sample_size, replace=False)
            
            self.X_sample = X_full.iloc[sample_indices].copy()
            self.y_sample = y_full.iloc[sample_indices].copy()
            self.feature_names = feature_cols
            
            # Load model and scaler
            self.model = joblib.load(MODELS_DIR / f'{self.config.model_name}.joblib')
            
            try:
                self.scaler = joblib.load(MODELS_DIR / 'standard_scaler.joblib')
            except:
                self.scaler = None  # Model doesn't require scaling
            
            print(f"✅ Data loaded: {sample_size} samples from {self.config.dataset_split}")
            print(f"✅ Model loaded: {self.config.model_name}")
            print(f"📊 Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data/model: {e}")
            return False
    
    def initialize_explainers(self) -> bool:
        """Initialize SHAP and LIME explainers"""
        try:
            print("🔧 Initializing XAI explainers...")
            
            # Initialize SHAP TreeExplainer (optimized for Random Forest)
            try:
                # Try with feature_perturbation='tree_path_dependent' for more stable computation
                self.shap_explainer = shap.TreeExplainer(
                    self.model, 
                    feature_perturbation='tree_path_dependent'
                )
            except:
                # Fallback to default configuration
                self.shap_explainer = shap.TreeExplainer(self.model)
            
            # Initialize LIME TabularExplainer
            self.lime_explainer = LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=self.feature_names,
                class_names=['No Heart Condition', 'Heart Condition'],
                categorical_features=[],  # All features are numeric after preprocessing
                mode='classification',
                discretize_continuous=True
            )
            
            print("✅ SHAP TreeExplainer initialized")
            print("✅ LIME TabularExplainer initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing explainers: {e}")
            return False
    
    def compute_shap_values(self) -> bool:
        """Compute SHAP values for sample data"""
        try:
            print("📊 Computing SHAP values...")
            
            print("Computing SHAP values...")
            # Convert to numpy array to ensure proper format
            X_array = np.array(self.X_sample)
            shap_values = self.shap_explainer(X_array)
            
            # Handle SHAP Explanation object (newer SHAP versions)
            if hasattr(shap_values, 'values'):
                # Extract values from Explanation object
                if len(shap_values.values.shape) == 3:  # Binary classification
                    self.shap_values = shap_values.values[:, :, 1]  # Heart condition class
                else:
                    self.shap_values = shap_values.values
                # Extract expected value
                if hasattr(shap_values, 'base_values'):
                    base_vals = shap_values.base_values
                    if isinstance(base_vals, (list, np.ndarray)) and len(base_vals.shape) > 0:
                        self.expected_value = float(base_vals[0]) if len(base_vals.shape) == 1 else float(base_vals[0, 1])
                    else:
                        self.expected_value = float(base_vals)
                else:
                    expected_vals = self.shap_explainer.expected_value
                    self.expected_value = float(expected_vals[1]) if isinstance(expected_vals, (list, np.ndarray)) else float(expected_vals)
            # Handle legacy SHAP format (list of arrays)
            elif isinstance(shap_values, list):
                self.shap_values = shap_values[1]  # Heart condition class
                expected_vals = self.shap_explainer.expected_value
                self.expected_value = float(expected_vals[1]) if isinstance(expected_vals, (list, np.ndarray)) else float(expected_vals)
            else:
                self.shap_values = shap_values
                expected_vals = self.shap_explainer.expected_value
                self.expected_value = float(expected_vals) if np.ndim(expected_vals) == 0 else float(expected_vals[0])
            
            print(f"✅ SHAP computation complete")
            print(f"   • Expected value: {float(self.expected_value):.4f}")
            print(f"   • SHAP values shape: {self.shap_values.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error computing SHAP values: {e}")
            return False
    
    def generate_global_analysis(self) -> pd.DataFrame:
        """Generate global feature importance analysis"""
        print("🌍 Generating global feature importance analysis...")
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.X_sample.columns,
            'mean_abs_shap': feature_importance,
            'clinical_domain': [self.clinical_domains.get(
                f.replace('numeric__', ''), 'Other Health Factor'
            ) for f in self.X_sample.columns]
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Add clinical relevance categories
        importance_df['clinical_relevance'] = importance_df['mean_abs_shap'].apply(
            lambda x: 'Critical' if x > 0.05 else 'Significant' if x > 0.02 else 'Moderate'
        )
        
        # Save global importance
        importance_df.to_csv(XAI_OUTPUT_DIR / 'global_feature_importance.csv', index=False)
        
        print(f"✅ Global analysis complete: {len(importance_df)} features analyzed")
        print(f"💾 Saved: global_feature_importance.csv")
        
        return importance_df
    
    def create_shap_visualizations(self, importance_df: pd.DataFrame):
        """Generate SHAP visualization suite"""
        if not self.config.save_plots:
            return
            
        print("🎨 Generating SHAP visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        
        # 1. Summary plot (feature importance + distribution)
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.X_sample,
                         plot_type="dot", 
                         color_bar_label="Feature Value",
                         show=False)
        plt.title("SHAP Feature Importance - Heart Condition Risk Factors", 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
        plt.tight_layout()
        plt.savefig(XAI_OUTPUT_DIR / 'shap_summary_plot.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot for feature ranking
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_sample,
                         plot_type="bar", show=False)
        plt.title("Mean SHAP Values - Clinical Risk Factor Ranking", 
                  fontsize=14, fontweight='bold')
        plt.xlabel("Mean |SHAP Value| (Average Impact)", fontsize=12)
        plt.tight_layout()
        plt.savefig(XAI_OUTPUT_DIR / 'shap_bar_plot.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ SHAP summary visualizations saved")
    
    def generate_case_explanations(self) -> List[Dict]:
        """Generate explanations for representative cases"""
        print("🔍 Generating case-specific explanations...")
        
        # Get model predictions
        y_pred_proba = self.model.predict_proba(self.X_sample)[:, 1]
        
        # Select representative cases
        high_risk_idx = np.argmax(y_pred_proba)
        low_risk_idx = np.argmin(y_pred_proba)  
        medium_risk_idx = np.argsort(y_pred_proba)[len(y_pred_proba)//2]
        
        cases = [
            (high_risk_idx, "high_risk", y_pred_proba[high_risk_idx]),
            (medium_risk_idx, "medium_risk", y_pred_proba[medium_risk_idx]),
            (low_risk_idx, "low_risk", y_pred_proba[low_risk_idx])
        ]
        
        case_results = []
        
        for idx, risk_label, prob in cases:
            # Generate SHAP waterfall plot
            if self.config.save_plots:
                self._create_waterfall_plot(idx, risk_label, prob)
            
            # Generate LIME explanation
            lime_result = self._create_lime_explanation(idx, risk_label, prob)
            
            case_results.append({
                'case': risk_label,
                'index': idx,
                'prediction': prob,
                'actual': bool(self.y_sample.iloc[idx]),
                'lime_explanation': lime_result
            })
            
            print(f"   ✅ {risk_label.replace('_', ' ').title()} case explained (Pred: {prob:.1%})")
        
        return case_results
    
    def _create_waterfall_plot(self, idx: int, risk_label: str, prob: float):
        """Create SHAP waterfall plot for specific case"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create SHAP explanation object
        shap_exp = shap.Explanation(
            values=self.shap_values[idx],
            base_values=self.expected_value,
            data=self.X_sample.iloc[idx],
            feature_names=self.X_sample.columns
        )
        
        # Generate waterfall plot
        shap.plots.waterfall(shap_exp, show=False)
        plt.title(f"SHAP Waterfall - {risk_label.replace('_', ' ').title()} Patient (Pred: {prob:.1%})", 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(XAI_OUTPUT_DIR / f'waterfall_{risk_label}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_lime_explanation(self, idx: int, risk_label: str, prob: float) -> Dict:
        """Create LIME explanation for specific case"""
        # Prediction function for LIME
        def predict_fn(X):
            return self.model.predict_proba(X)
        
        # Generate LIME explanation
        instance = self.X_sample.iloc[idx].values
        lime_exp = self.lime_explainer.explain_instance(
            instance,
            predict_fn,
            num_features=self.config.n_lime_features,
            top_labels=1
        )
        
        # Save HTML explanation
        html_filename = f'lime_explanation_{risk_label}.html'
        lime_exp.save_to_file(XAI_OUTPUT_DIR / html_filename)
        
        # Extract explanation data
        # Check available labels and use the first one (typically positive class)
        available_labels = lime_exp.available_labels()
        target_label = available_labels[0] if available_labels else 0
        exp_list = lime_exp.as_list(label=target_label)
        exp_df = pd.DataFrame(exp_list, columns=['feature', 'importance'])
        
        return {
            'explanation_df': exp_df,
            'html_file': html_filename,
            'top_factors': exp_list[:5]
        }
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete XAI analysis pipeline"""
        print("🚀 Starting Week 5-6 XAI Analysis Pipeline")
        print("=" * 70)
        
        # Step 1: Load data and model
        if not self.load_data_and_model():
            return {'success': False, 'error': 'Failed to load data/model'}
        
        # Step 2: Initialize explainers
        if not self.initialize_explainers():
            return {'success': False, 'error': 'Failed to initialize explainers'}
        
        # Step 3: Compute SHAP values
        if not self.compute_shap_values():
            return {'success': False, 'error': 'Failed to compute SHAP values'}
        
        # Step 4: Global analysis
        importance_df = self.generate_global_analysis()
        
        # Step 5: Create visualizations
        self.create_shap_visualizations(importance_df)
        
        # Step 6: Case-specific explanations
        case_results = self.generate_case_explanations()
        
        # Summary
        results = {
            'success': True,
            'model': self.config.model_name,
            'dataset': self.config.dataset_split,
            'samples_analyzed': len(self.X_sample),
            'features_analyzed': len(importance_df),
            'cases_explained': len(case_results),
            'output_directory': str(XAI_OUTPUT_DIR),
            'clinical_directory': str(CLINICAL_OUTPUT_DIR)
        }
        
        print(f"\n✅ XAI Analysis Complete!")
        print(f"   📊 Samples analyzed: {results['samples_analyzed']}")
        print(f"   🔍 Features analyzed: {results['features_analyzed']}")
        print(f"   📋 Cases explained: {results['cases_explained']}")
        print(f"   📂 Output saved to: {XAI_OUTPUT_DIR}")
        print(f"   🏥 Clinical reports: {CLINICAL_OUTPUT_DIR}")
        
        return results


def main():
    """CLI interface for XAI analysis"""
    parser = argparse.ArgumentParser(
        description="Week 5-6 XAI Analysis for Healthcare Decision Support"
    )
    
    parser.add_argument('--model', default='random_forest_tuned',
                       help='Model name to analyze')
    parser.add_argument('--dataset', choices=['validation', 'test'], 
                       default='validation', help='Dataset split to use')
    parser.add_argument('--sample-size', type=int, default=200,
                       help='Number of samples to analyze')
    parser.add_argument('--lime-features', type=int, default=10,
                       help='Number of features for LIME explanations')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--no-clinical', action='store_true',
                       help='Skip clinical report generation')
    
    args = parser.parse_args()
    
    # Create configuration
    config = XAIConfig(
        model_name=args.model,
        dataset_split=args.dataset,
        sample_size=args.sample_size,
        n_lime_features=args.lime_features,
        random_state=args.random_state,
        save_plots=not args.no_plots,
        generate_clinical_reports=not args.no_clinical
    )
    
    # Run analysis
    analyzer = HealthcareXAIAnalyzer(config)
    results = analyzer.run_complete_analysis()
    
    if results['success']:
        print("\n🎯 Week 5-6 XAI Implementation: COMPLETED")
        print("📋 Ready for Week 7-8: Clinical Decision Support & Gradio Demo")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

# Run the module as a script for CLI usage: 
# Run as a Python module (if you're in the project root): python -m src.explainability
# Run for full validation set with 200 samples (default): python src/explainability.py --sample-size 200 --dataset validation
# With options: python -m src.explainability --dataset test --sample-size 300