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

**Random Forest Example:**
```python
rf_model = RandomForestClassifier(
    class_weight='balanced_subsample',
    n_estimators=100
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

Based on our strong Week 1-2 foundation (XGBoost 49.3% accuracy, excellent calibration ECE=0.009):

### **Phase 1: Cost-Sensitive XGBoost Optimization**
1. **Implement class-weighted XGBoost** using our 1:39.2 ratio
2. **Tune hyperparameters** with focus on imbalance-specific parameters
3. **Validate calibration** maintains excellent ECE scores

### **Phase 2: Threshold Optimization**
1. **Leverage excellent calibration** to optimize decision thresholds
2. **Per-class threshold tuning** for 5-class health status
3. **Macro F1-score optimization** for balanced performance

### **Phase 3: Strategic Ensemble**
1. **Exploit high correlation** between XGBoost and Random Forest (0.85)
2. **Combine complementary approaches** (cost-sensitive + threshold-optimized)
3. **Validate ensemble diversity** while maintaining calibration quality

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

### **Week 3-4 Tasks:**
- [ ] Implement cost-sensitive XGBoost with 1:39.2 class weights
- [ ] Develop threshold optimization framework
- [ ] Create per-class performance monitoring
- [ ] Validate calibration retention across approaches
- [ ] Compare ensemble strategies using high XGBoost-RF correlation
- [ ] Document performance improvements with comprehensive metrics

### **Documentation:**
- [ ] Update error analysis with class-specific insights
- [ ] Create visualization comparing baseline vs. optimized performance
- [ ] Document methodology for reproducible implementation
- [ ] Prepare technical report section on imbalance handling

---

**Status:** Ready for implementation based on strong Week 1-2 baseline foundation  
**Priority:** High - Critical for clinical application success  
**Dependencies:** None - leverages existing calibrated models and comprehensive error analysis