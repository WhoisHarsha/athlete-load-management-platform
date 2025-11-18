"""
=============================================================================
INJURY RISK PREDICTION MODEL
=============================================================================
Purpose: Build and evaluate ML models for predicting athlete injury risk
Author: Harsha Prakash (Health Data Science)
Date: November 2025
Project: Athlete Load Management & Performance Optimization Platform

Models: Random Forest & XGBoost
Evaluation: Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("INJURY RISK PREDICTION MODEL - TRAINING & EVALUATION")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA FROM CSV FILES
# =============================================================================
print("\n[1/7] Loading data from CSV files...")

# Load all datasets
df_athletes = pd.read_csv('../../data_generation/athletes.csv')
df_injuries = pd.read_csv('../../data_generation/injuries.csv')
df_recovery = pd.read_csv('../../data_generation/recovery_metrics.csv')
df_predictions = pd.read_csv('../../data_generation/risk_predictions.csv')

print(f"   ✓ Athletes: {len(df_athletes)} records")
print(f"   ✓ Injuries: {len(df_injuries)} records")
print(f"   ✓ Recovery Metrics: {len(df_recovery)} records")
print(f"   ✓ Risk Predictions: {len(df_predictions)} records")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n[2/7] Engineering features...")

# Merge athlete demographics with predictions
df_model = df_predictions.merge(df_athletes, on='athlete_id', how='left')

# Calculate athlete age from date of birth
df_model['date_of_birth'] = pd.to_datetime(df_model['date_of_birth'])
df_model['age'] = (pd.to_datetime('2024-11-17') - df_model['date_of_birth']).dt.days / 365.25

# Encode categorical variables
position_dummies = pd.get_dummies(df_model['position'], prefix='position')
df_model = pd.concat([df_model, position_dummies], axis=1)

# Calculate injury severity history
injury_severity_by_athlete = df_injuries.groupby('athlete_id').agg({
    'injury_severity': ['mean', 'max', 'std']
}).reset_index()
injury_severity_by_athlete.columns = ['athlete_id', 'avg_injury_severity', 'max_injury_severity', 'std_injury_severity']
df_model = df_model.merge(injury_severity_by_athlete, on='athlete_id', how='left')
df_model['avg_injury_severity'] = df_model['avg_injury_severity'].fillna(0)
df_model['max_injury_severity'] = df_model['max_injury_severity'].fillna(0)
df_model['std_injury_severity'] = df_model['std_injury_severity'].fillna(0)

# Calculate recurrence rate
recurrence_by_athlete = df_injuries.groupby('athlete_id')['is_recurrence'].mean().reset_index()
recurrence_by_athlete.columns = ['athlete_id', 'recurrence_rate']
df_model = df_model.merge(recurrence_by_athlete, on='athlete_id', how='left')
df_model['recurrence_rate'] = df_model['recurrence_rate'].fillna(0)

# Add recovery success rate
recovery_success = df_recovery.groupby('athlete_id').agg({
    'recovery_percentage': 'mean',
    'progress_status': lambda x: (x.isin(['Excellent', 'On Track'])).mean()
}).reset_index()
recovery_success.columns = ['athlete_id', 'avg_recovery_pct', 'good_progress_rate']
df_model = df_model.merge(recovery_success, on='athlete_id', how='left')
df_model['avg_recovery_pct'] = df_model['avg_recovery_pct'].fillna(50)
df_model['good_progress_rate'] = df_model['good_progress_rate'].fillna(0.5)

# Create interaction features
df_model['acwr_fatigue_interaction'] = df_model['acute_chronic_workload_ratio'] * df_model['fatigue_score']
df_model['history_recency_interaction'] = df_model['injury_history_count'] * (1 / (df_model['days_since_last_injury'] + 1))
df_model['sleep_fatigue_interaction'] = df_model['sleep_quality_score'] * (10 - df_model['fatigue_score'])

# Risk flags
df_model['high_acwr_flag'] = (df_model['acute_chronic_workload_ratio'] > 1.5).astype(int)
df_model['high_fatigue_flag'] = (df_model['fatigue_score'] >= 7).astype(int)
df_model['poor_sleep_flag'] = (df_model['sleep_quality_score'] < 6).astype(int)
df_model['high_soreness_flag'] = (df_model['muscle_soreness_level'] >= 7).astype(int)
df_model['recent_injury_flag'] = (df_model['days_since_last_injury'] < 30).astype(int)

print(f"   ✓ Total features engineered: {df_model.shape[1]}")
print(f"   ✓ Key features created: age, position dummies, injury history stats, interaction terms")

# =============================================================================
# 3. PREPARE TRAINING DATA
# =============================================================================
print("\n[3/7] Preparing training data...")

# Filter to only records with actual outcomes
df_train = df_model[df_model['actual_injury_occurred'].notna()].copy()

# Define feature columns
feature_cols = [
    # Workload features
    'acute_chronic_workload_ratio',
    'recent_workload_spike',
    'training_monotony',
    
    # Wellness features
    'fatigue_score',
    'sleep_quality_score',
    'muscle_soreness_level',
    
    # Injury history features
    'injury_history_count',
    'days_since_last_injury',
    'previous_injury_severity_avg',
    'recurrence_risk_multiplier',
    'avg_injury_severity',
    'max_injury_severity',
    'std_injury_severity',
    'recurrence_rate',
    
    # Recovery features
    'avg_recovery_pct',
    'good_progress_rate',
    
    # Athlete demographics
    'age',
    'years_experience',
    
    # Interaction features
    'acwr_fatigue_interaction',
    'history_recency_interaction',
    'sleep_fatigue_interaction',
    
    # Risk flags
    'high_acwr_flag',
    'high_fatigue_flag',
    'poor_sleep_flag',
    'high_soreness_flag',
    'recent_injury_flag'
]

# Add position dummies
position_cols = [col for col in df_train.columns if col.startswith('position_')]
feature_cols.extend(position_cols)

# Prepare X and y
X = df_train[feature_cols].copy()
y = df_train['actual_injury_occurred'].astype(int)

# Handle missing values
X = X.fillna(X.median())

# Convert boolean columns to int
bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)

print(f"   ✓ Training samples: {len(X)}")
print(f"   ✓ Features: {len(feature_cols)}")
print(f"   ✓ Positive class (injuries): {y.sum()} ({y.mean()*100:.2f}%)")
print(f"   ✓ Negative class (no injury): {len(y) - y.sum()} ({(1-y.mean())*100:.2f}%)")

# =============================================================================
# 4. TRAIN-TEST SPLIT
# =============================================================================
print("\n[4/7] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training set: {len(X_train)} samples")
print(f"   ✓ Test set: {len(X_test)} samples")

# Scale features (optional, but helps with some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 5. TRAIN MODELS
# =============================================================================
print("\n[5/7] Training machine learning models...")

# Model 1: Random Forest
print("\n   Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("   ✓ Random Forest training complete")

# Try XGBoost (optional - install with: pip install xgboost)
try:
    import xgboost as xgb
    print("\n   Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    print("   ✓ XGBoost training complete")
    has_xgboost = True
except ImportError:
    print("   ⚠ XGBoost not installed. Using Random Forest only.")
    print("   To install: pip install xgboost")
    has_xgboost = False

# =============================================================================
# 6. MODEL EVALUATION
# =============================================================================
print("\n[6/7] Evaluating model performance...")

# Random Forest predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n" + "="*80)
print("RANDOM FOREST CLASSIFIER - PERFORMANCE METRICS")
print("="*80)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['No Injury', 'Injury']))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:")
print(f"True Negatives: {cm_rf[0,0]}, False Positives: {cm_rf[0,1]}")
print(f"False Negatives: {cm_rf[1,0]}, True Positives: {cm_rf[1,1]}")

# ROC-AUC
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"\nROC-AUC Score: {roc_auc_rf:.4f}")

# Cross-validation score
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-Validation ROC-AUC: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

# XGBoost evaluation (if available)
if has_xgboost:
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*80)
    print("XGBOOST CLASSIFIER - PERFORMANCE METRICS")
    print("="*80)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=['No Injury', 'Injury']))
    
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    print("\nConfusion Matrix:")
    print(f"True Negatives: {cm_xgb[0,0]}, False Positives: {cm_xgb[0,1]}")
    print(f"False Negatives: {cm_xgb[1,0]}, True Positives: {cm_xgb[1,1]}")
    
    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    print(f"\nROC-AUC Score: {roc_auc_xgb:.4f}")

# =============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n[7/7] Analyzing feature importance...")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create output directory
import os
os.makedirs('model_outputs', exist_ok=True)

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Injury', 'Injury'],
            yticklabels=['No Injury', 'Injury'])
plt.title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model_outputs/confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_rf.png")
plt.close()

# 2. ROC Curve
plt.figure(figsize=(10, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', linewidth=2)

if has_xgboost:
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Injury Risk Prediction Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('model_outputs/roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curve.png")
plt.close()

# 3. Precision-Recall Curve
plt.figure(figsize=(10, 6))
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_proba_rf)
pr_auc_rf = auc(recall_rf, precision_rf)
plt.plot(recall_rf, precision_rf, label=f'Random Forest (AUC = {pr_auc_rf:.3f})', linewidth=2)

if has_xgboost:
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_pred_proba_xgb)
    pr_auc_xgb = auc(recall_xgb, precision_xgb)
    plt.plot(recall_xgb, precision_xgb, label=f'XGBoost (AUC = {pr_auc_xgb:.3f})', linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - Injury Risk Prediction', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('model_outputs/precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: precision_recall_curve.png")
plt.close()

# 4. Feature Importance Plot
plt.figure(figsize=(12, 10))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 20 Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('model_outputs/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# 5. Risk Score Distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_pred_proba_rf[y_test == 0], bins=30, alpha=0.7, label='No Injury', color='green')
plt.hist(y_pred_proba_rf[y_test == 1], bins=30, alpha=0.7, label='Injury', color='red')
plt.xlabel('Predicted Injury Probability', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Predicted Risk Score Distribution', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot([y_pred_proba_rf[y_test == 0], y_pred_proba_rf[y_test == 1]], 
            labels=['No Injury', 'Injury'])
plt.ylabel('Predicted Injury Probability', fontsize=11)
plt.title('Risk Score by Actual Outcome', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_outputs/risk_score_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: risk_score_distribution.png")
plt.close()

# =============================================================================
# 9. SAVE RESULTS
# =============================================================================
print("\n" + "="*80)
print("SAVING MODEL AND RESULTS")
print("="*80)

# Save feature importance
feature_importance.to_csv('model_outputs/feature_importance.csv', index=False)
print("✓ Saved: feature_importance.csv")

# Save test predictions
test_results = pd.DataFrame({
    'athlete_id': df_train.iloc[X_test.index]['athlete_id'].values,
    'actual_injury': y_test.values,
    'predicted_injury_rf': y_pred_rf,
    'predicted_probability_rf': y_pred_proba_rf
})

if has_xgboost:
    test_results['predicted_injury_xgb'] = y_pred_xgb
    test_results['predicted_probability_xgb'] = y_pred_proba_xgb

test_results.to_csv('model_outputs/test_predictions.csv', index=False)
print("✓ Saved: test_predictions.csv")

# Save model summary
with open('model_outputs/model_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("INJURY RISK PREDICTION MODEL - SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Training Date: 2024-11-17\n")
    f.write(f"Total Samples: {len(X)}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Test Samples: {len(X_test)}\n")
    f.write(f"Number of Features: {len(feature_cols)}\n")
    f.write(f"Class Balance: {y.sum()} injuries ({y.mean()*100:.2f}%), {len(y)-y.sum()} no injury\n\n")
    
    f.write("RANDOM FOREST PERFORMANCE:\n")
    f.write(f"ROC-AUC: {roc_auc_rf:.4f}\n")
    f.write(f"CV ROC-AUC: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})\n\n")
    
    if has_xgboost:
        f.write("XGBOOST PERFORMANCE:\n")
        f.write(f"ROC-AUC: {roc_auc_xgb:.4f}\n\n")
    
    f.write("TOP 10 FEATURES:\n")
    for idx, row in feature_importance.head(10).iterrows():
        f.write(f"{row['feature']}: {row['importance']:.4f}\n")

print("✓ Saved: model_summary.txt")

print("\n" + "="*80)
print("✅ MODEL TRAINING AND EVALUATION COMPLETE!")
print("="*80)
print("\nOutput files saved in 'model_outputs/' directory:")
print("  - confusion_matrix_rf.png")
print("  - roc_curve.png")
print("  - precision_recall_curve.png")
print("  - feature_importance.png")
print("  - risk_score_distribution.png")
print("  - feature_importance.csv")
print("  - test_predictions.csv")
print("  - model_summary.txt")
print("\n" + "="*80)