import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, recall_score, make_scorer,
    accuracy_score, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- [NEW] Helper function for full evaluation ---
def evaluate_and_save_metrics(model, model_name, X_test, y_test, results_dir):
    """
    Calculates all metrics, prints them, and saves reports and plots.
    """
    print(f"\n--- Evaluating Model: {model_name} ---")
    
    # 1. Get Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class (1)

    # 2. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malicious (1)'])
    
    # 3. Build, Print, and Save Text Report
    report_header = f"--- Performance Report: {model_name} ---\n\n"
    acc_str = f"Overall Accuracy: {acc:.4f}\n"
    auc_str = f"ROC-AUC Score: {auc_score:.4f}\n\n"
    class_report_str = f"Classification Report:\n{report}\n"
    
    full_report = report_header + acc_str + auc_str + class_report_str
    
    print(full_report)
    
    report_filename = os.path.join(results_dir, f"{model_name}_performance_report.txt")
    with open(report_filename, 'w') as f:
        f.write(full_report)
    print(f"-> Text report saved to '{report_filename}'")

    # 4. Generate and Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign (0)', 'Malicious (1)'], 
                yticklabels=['Benign (0)', 'Malicious (1)'])
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    cm_filename = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_filename)
    plt.close() # Close the plot to prevent it from displaying
    print(f"-> Confusion Matrix plot saved to '{cm_filename}'")

    # 5. Generate and Save ROC-AUC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC-AUC Curve', fontsize=16)
    plt.legend(loc="lower right")
    roc_filename = os.path.join(results_dir, f"{model_name}_roc_auc_curve.png")
    plt.savefig(roc_filename)
    plt.close() # Close the plot
    print(f"-> ROC-AUC Curve plot saved to '{roc_filename}'")


def tune_and_train_models(input_file, xgb_model_path="xgboost_detector_tuned.json", lgb_model_path="lightgbm_detector_tuned.txt"):
    """
    Loads corrected data, selects features, tunes hyperparameters using RandomizedSearchCV
    to optimize for RECALL, evaluates the best models, and saves them.
    """
    print("--- Model Hyperparameter Tuning & Training ---")
    
    # --- [NEW] Create results directory ---
    RESULTS_DIR = 'model_performance_reports'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n[INFO] Performance reports will be saved to '{RESULTS_DIR}'")
    
    print(f"\n[INFO] Loading final REFINED dataset from '{input_file}'...")
    if not os.path.exists(input_file):
        print(f"[ERROR] The file '{input_file}' was not found.")
        return

    try:
        df = pd.read_csv(input_file, low_memory=False)
        print(f"-> Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return

    # --- Step 1: Feature Selection & Dtype Setup ---
    print("\n--- Step 1: Selecting and Preparing Final Features ---")
    label_col = 'Label' if 'Label' in df.columns else 'label'

    final_feature_list_base = [
        'FailedLogonCount_5min', 'FileActivityVolume_1min', 'WeakKerberosRequestCount_10min',
        'IsShadowCopyDeletion', 'IsNewService', 'IsLsassAccess', 'cmd_length',
        'cmd_is_encoded', 'cmd_is_hidden', 'targets_sensitive_object', 'is_in_suspicious_dir',
        'ProcessName_freq', 'ParentProcessName_freq', 'SubjectUserName_freq', 'IpAddress_freq',
        'WorkstationName_freq', 'DestinationIp_freq', 'TargetOutboundUserName_freq',
        'Id', 'LogonType', 'AuthenticationPackageName', 'TicketEncryptionType', 'DestinationPort'
    ]
    one_hot_cols_generated = [col for col in df.columns if col.startswith("Role_")]
    final_features_for_model = [f for f in (final_feature_list_base + one_hot_cols_generated) if f in df.columns]

    final_columns_inc_label = final_features_for_model + [label_col]
    df_model = df[final_columns_inc_label].copy()

    # Re-assert category dtypes for the models (essential after CSV read)
    categorical_cols = ['Id', 'LogonType', 'AuthenticationPackageName', 'TicketEncryptionType', 'DestinationPort']
    for col in categorical_cols:
        if col in df_model.columns:
            if pd.api.types.is_numeric_dtype(df_model[col]):
                df_model[col] = df_model[col].fillna(-1).astype(int)
            else:
                df_model[col] = df_model[col].fillna('Unknown').astype(str)
            df_model[col] = df_model[col].astype('category')
    
    # --- Step 2: Split Data ---
    X = df_model.drop(label_col, axis=1)
    y = df_model[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"-> Training data: {len(X_train)} samples")

    # --- Step 3: Imbalance Weight & Scoring Function ---
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    scale_pos_weight = count_neg / count_pos if count_pos > 0 else 1
    
    # Define custom scorer to maximize RECALL for the positive class (Malicious=1)
    # This forces the random search to prioritize finding actual attacks (minimizing FNs).
    recall_scorer = make_scorer(recall_score, pos_label=1)
    print(f"-> Tuning objective: Maximizing Recall (Malicious Class, using scale_pos_weight: {scale_pos_weight:.2f})")

    # --- Step 4: Hyperparameter Tuning - XGBoost ---
    print("\n--- [Tuning XGBoost] ---")
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 12],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        scale_pos_weight=scale_pos_weight, enable_categorical=True,
        use_label_encoder=False, random_state=42, n_jobs=-1
    )
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base, param_distributions=xgb_param_grid, n_iter=10, 
        scoring=recall_scorer, cv=3, verbose=1, random_state=42, n_jobs=1 
    )
    start_time_xgb = time.time()
    xgb_search.fit(X_train, y_train)
    end_time_xgb = time.time()
    print(f"-> XGBoost tuning completed in {end_time_xgb - start_time_xgb:.2f} seconds.")
    print(f"    Best Recall: {xgb_search.best_score_:.4f}")
    print(f"    Best Params: {xgb_search.best_params_}")
    best_xgb_model = xgb_search.best_estimator_


    # --- Step 5: Hyperparameter Tuning - LightGBM ---
    print("\n--- [Tuning LightGBM] ---")
    lgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'num_leaves': [20, 31, 50],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'is_unbalance': [True] # Force imbalance handling
    }
    lgb_base = lgb.LGBMClassifier(objective='binary', metric='logloss', random_state=42, n_jobs=-1)
    lgb_search = RandomizedSearchCV(
        estimator=lgb_base, param_distributions=lgb_param_grid, n_iter=10, 
        scoring=recall_scorer, cv=3, verbose=1, random_state=42, n_jobs=1
    )
    start_time_lgb = time.time()
    lgb_search.fit(X_train, y_train)
    end_time_lgb = time.time()
    print(f"-> LightGBM tuning completed in {end_time_lgb - start_time_lgb:.2f} seconds.")
    print(f"    Best Recall: {lgb_search.best_score_:.4f}")
    print(f"    Best Params: {lgb_search.best_params_}")
    best_lgb_model = lgb_search.best_estimator_


    # --- [UPDATED] Step 6: Final Evaluation and Saving ---
    print("\n--- Final Evaluation and Saving ---")
    
    # 1. Evaluate XGBoost
    evaluate_and_save_metrics(best_xgb_model, "XGBoost", X_test, y_test, RESULTS_DIR)
    
    # 2. Evaluate LightGBM
    evaluate_and_save_metrics(best_lgb_model, "LightGBM", X_test, y_test, RESULTS_DIR)
    
    # 3. Save Models
    best_xgb_model.save_model(xgb_model_path)
    print(f"\n-> Tuned XGBoost model saved to '{xgb_model_path}'")
    lgb_search.best_estimator_.booster_.save_model(lgb_model_path)
    print(f"-> Tuned LightGBM model saved to '{lgb_model_path}'")

    print("\nNEXT STEP: Analyze reports and proceed to the RL phase.")


if __name__ == "__main__":
    # Ensure this points to the file created by refine_labels.py
    tune_and_train_models('FINAL_DATASET_v4_generalized.csv', 
                          xgb_model_path="xgboost_detector_tuned.json",
                          lgb_model_path="lightgbm_detector_tuned.txt")