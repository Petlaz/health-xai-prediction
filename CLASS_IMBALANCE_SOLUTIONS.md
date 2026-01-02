# Class Imbalance Solutions for Health Prediction

**Project:** Prediction and Local Explainable AI (XAI) in Healthcare  
**Context:** 5-class health status prediction with severe 1:39.2 class imbalance  
**Baseline Results:** XGBoost 49.3% accuracy, ECE=0.009 (excellent calibration)

---

## Problem Statement

Our Week 1-2 baseline implementation revealed a **severe class imbalance** in the European Health Survey dataset:
- **Dataset:** 11,322 records with 5-class health status prediction
- **Class ratio:** 1:39.2 between smallest and largest health classes
- **Impact:** All baseline models exhibit majority class bias despite initial class balancing

---

## Solution Categories

### 1. 🔧 **Cost-Sensitive Learning**
Assign different misclassification costs to minority vs. majority classes.

**Implementation Options:**
- **Class weights:** Inverse frequency weighting (`class_weight='balanced'`)
- **Custom penalties:** Manual weight assignment based on clinical importance
- **Loss function modification:** Weighted cross-entropy, focal loss

**XGBoost Example:**
```python
# Using our 1:39.2 ratio
xgb_model = XGBClassifier(
    scale_pos_weight=39.2,
    class_weight='balanced',
    eval_metric='mlogloss'
)
```

**SVM Example (RBF Kernel Only - Mac M1/M2 Optimized):**
```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

# Focus on RBF kernel for computational efficiency
svm_param_dist = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
    'class_weight': ['balanced', None]
}

svm_model = RandomizedSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    param_distributions=svm_param_dist,
    n_iter=50,  # Efficient search for Mac M1/M2
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)
```

---

### 2. ⚖️ **Threshold Optimization**
Adjust decision thresholds to better classify minority classes.

**Approaches:**
- **ROC-based optimization:** Maximize Youden's J statistic
- **F1-score optimization:** Find thresholds that maximize macro F1-score
- **Per-class thresholds:** Different optimal thresholds for each health class

**Implementation Strategy:**
```python
from sklearn.metrics import f1_score
import numpy as np

def optimize_thresholds(y_true, y_proba, n_classes=5):
    """Find optimal threshold for each class"""
    best_thresholds = []
    for class_idx in range(n_classes):
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        
        for thresh in thresholds:
            y_pred_class = (y_proba[:, class_idx] > thresh).astype(int)
            f1 = f1_score(y_true == class_idx, y_pred_class)
            f1_scores.append(f1)
        
        best_thresh = thresholds[np.argmax(f1_scores)]
        best_thresholds.append(best_thresh)
    
    return best_thresholds
```

---

### 3. 🎯 **Ensemble Methods for Imbalance**
Combine multiple models trained with different strategies.

**Balanced Bagging:**
- Train multiple models on balanced subsets
- Vote or average predictions across ensemble members

**Boosting with Class Focus:**
- AdaBoost with class weights
- Gradient boosting with minority class emphasis

**Heterogeneous Ensembles:**
```python
from sklearn.ensemble import VotingClassifier

# Combine different approaches
ensemble = VotingClassifier([
    ('xgb_balanced', XGBClassifier(scale_pos_weight=39.2)),
    ('rf_balanced', RandomForestClassifier(class_weight='balanced')),
    ('svm_weighted', SVC(class_weight='balanced', probability=True))
], voting='soft')
```

---

### 4. 🧠 **Algorithm-Specific Optimizations**

**XGBoost Parameters:**
- `max_delta_step`: Controls maximum delta step for leaf value estimation
- `min_child_weight`: Higher values prevent overfitting to minority classes
- `subsample`: Random sampling can help with class balance

**Neural Network Approaches:**
- **Focal Loss:** Focuses learning on hard examples
- **Weighted Cross-Entropy:** Direct class weight application
- **Synthetic Minority Oversampling Technique (SMOTE)** as data augmentation

---

### 5. 📊 **Enhanced Evaluation Strategies**

**Metrics Focus:**
- **Macro-averaged F1:** Equal weight to all classes
- **Weighted F1:** Accounts for class frequency
- **Per-class precision/recall:** Individual class performance
- **Cohen's Kappa:** Agreement accounting for chance

**Validation Approach:**
```python
from sklearn.model_selection import StratifiedKFold

# Ensure all classes represented in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 🎯 **Recommended Implementation Strategy**

Based on our strong Week 1-2 foundation (XGBoost 49.3% accuracy, excellent calibration ECE=0.009) and following ML best practices for master's research:

### **Phase 1: Comprehensive Hyperparameter Tuning** *(Primary Focus)*
1. **Tune ALL 5 models systematically** using RandomizedSearchCV with stratified 5-fold CV
2. **XGBoost optimization:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, regularization
3. **Random Forest tuning:** `n_estimators`, `max_depth`, depth/sample controls, `max_features`
4. **SVM optimization:** Focus on RBF kernel only - tune `C` and `gamma` for Mac M1/M2 efficiency
5. **Logistic Regression:** `C`, penalty types (L1, L2, ElasticNet), solver optimization
6. **Neural Network:** Architecture search, learning rates, dropout, batch sizes optimized for Apple Silicon
7. **Evaluation metric:** Macro F1-score during tuning to handle class imbalance

### **Phase 2: Model Selection & Validation** *(Critical for Research Integrity)*
1. **Compare tuned models** using validation set only (NO test set usage for selection)
2. **Select best 1-2 models** based purely on validation performance
3. **Maintain excellent calibration** (current ECE=0.009) through optimization
4. **Document improvements** over Week 1-2 baseline systematically

### **Phase 3: Class Imbalance Solutions for Selected Models** *(Targeted Application)*
1. **Apply cost-sensitive learning** to selected best models using our 1:39.2 ratio
2. **Implement threshold optimization** using validation set for calibrated probability tuning
3. **Strategic ensemble development** if multiple models show similar validation performance
4. **Final test evaluation** - single unbiased assessment of selected approach

### **Hardware Optimization (Mac M1/M2):**
- **Computational efficiency monitoring** during hyperparameter searches
- **Apple Silicon optimized** frameworks (sklearn, xgboost with proper compilation)
- **Memory-efficient CV strategies** for large parameter search spaces

### **Expected Outcomes:**
- **Improved minority class recall** without sacrificing overall performance
- **Maintained calibration quality** (current ECE=0.009)
- **Enhanced clinical utility** through balanced class performance
- **Robust foundation** for XAI implementation in subsequent phases

---

## 📈 **Success Metrics**

### **Primary Indicators:**
- **Macro F1-score improvement** (equal weight to all health classes)
- **Per-class recall enhancement** (especially minority classes)
- **Maintained calibration quality** (ECE ≤ 0.01)

### **Secondary Indicators:**
- **Balanced accuracy** across all 5 health classes
- **Cohen's Kappa** improvement (chance-adjusted agreement)
- **Clinical relevance** of feature importance patterns

---

## 📋 **Implementation Checklist**

### **Week 3-4 Phase 1: Comprehensive Tuning**
- [ ] Set up RandomizedSearchCV pipeline with stratified 5-fold CV for all 5 models
- [ ] Implement XGBoost hyperparameter search with comprehensive parameter grid
- [ ] Optimize Random Forest with depth and sampling parameters
- [ ] Tune SVM with RBF kernel focusing on C and gamma parameters
- [ ] Optimize Logistic Regression with penalty types and regularization
- [ ] Implement Neural Network architecture search optimized for Mac M1/M2
- [ ] Track computational efficiency and tuning convergence

### **Week 3-4 Phase 2: Model Selection**
- [ ] Compare all tuned models on validation set using macro F1-score
- [ ] Assess calibration quality (ECE) for top performers
- [ ] Select best 1-2 models based purely on validation performance
- [ ] Document improvement over Week 1-2 baseline (XGBoost 49.3%)
- [ ] Reserve test set strictly for final evaluation

### **Week 3-4 Phase 3: Class Imbalance Optimization**
- [ ] Apply cost-sensitive learning to selected models with 1:39.2 class weights
- [ ] Implement threshold optimization framework using validation set
- [ ] Develop ensemble strategies if multiple models show similar performance
- [ ] Perform single final test set evaluation for unbiased performance estimate
- [ ] Compare final results against Week 1-2 baseline

### **Documentation & Research Standards:**
- [ ] Update notebooks/03_modeling.ipynb with comprehensive Week 3-4 results
- [ ] Create visualization comparing baseline vs. tuned vs. class-balanced performance
- [ ] Document all methodological choices following ML best practices
- [ ] Prepare technical sections for master's thesis with proper experimental design

---

**Status:** Ready for implementation based on strong Week 1-2 baseline foundation  
**Priority:** High - Critical for clinical application success  
**Dependencies:** None - leverages existing calibrated models and comprehensive error analysis