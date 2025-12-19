import pandas as pd
import numpy as np
from catboost import CatBoostClassifier # Import CatBoost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, make_scorer, # Changed to f1_score
    accuracy_score, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning) # CatBoost can throw FutureWarnings

# --- Helper function for full evaluation (Unchanged from your previous script) ---
def evaluate_and_save_metrics(model, model_name, X_test, y_test, results_dir, cat_features_indices=None):
    """
    Calculates all metrics, prints them, and saves reports and plots.
    Handles CatBoost needing feature names potentially.
    """
    print(f"\n--- Evaluating Model: {model_name} ---")

    # 1. Get Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class (1)

    # 2. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    # Ensure y_true and y_pred_proba are numpy arrays for roc_auc_score
    auc_score = roc_auc_score(np.array(y_test), np.array(y_pred_proba))
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


def tune_and_train_catboost(input_file, catboost_model_path="catboost_detector_tuned.cbm"):
    """
    Loads data, identifies categorical features, tunes CatBoost using RandomizedSearchCV
    to optimize for F1-SCORE, evaluates the best model, and saves it.
    """
    print("--- CatBoost Hyperparameter Tuning & Training ---")

    RESULTS_DIR = 'model_performance_reports'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n[INFO] Performance reports will be saved to '{RESULTS_DIR}'")

    print(f"\n[INFO] Loading final REFINED dataset from '{input_file}'...")
    if not os.path.exists(input_file):
        print(f"[ERROR] The file '{input_file}' was not found.")
        return

    try:
        # Load data, explicitly keeping potential object types for CatBoost
        df = pd.read_csv(input_file, low_memory=False)
        print(f"-> Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return

    # --- Step 1: Feature Selection & Dtype Setup for CatBoost ---
    print("\n--- Step 1: Selecting and Preparing Final Features ---")
    label_col = 'Label' if 'Label' in df.columns else 'label'

    # Define the base features (same as before)
    final_feature_list_base = [
        'FailedLogonCount_5min', 'FileActivityVolume_1min', 'WeakKerberosRequestCount_10min',
        'IsShadowCopyDeletion', 'IsNewService', 'IsLsassAccess', 'cmd_length',
        'cmd_is_encoded', 'cmd_is_hidden', 'targets_sensitive_object', 'is_in_suspicious_dir',
        'ProcessName_freq', 'ParentProcessName_freq', 'SubjectUserName_freq', 'IpAddress_freq',
        'WorkstationName_freq', 'DestinationIp_freq', 'TargetOutboundUserName_freq',
        'Id', 'LogonType', 'AuthenticationPackageName', 'TicketEncryptionType', 'DestinationPort'
    ]
    one_hot_cols_generated = [col for col in df.columns if col.startswith("Role_")] # Keep generated Role_ cols
    final_features_for_model = [f for f in (final_feature_list_base + one_hot_cols_generated) if f in df.columns]

    final_columns_inc_label = final_features_for_model + [label_col]
    df_model = df[final_columns_inc_label].copy()

    # --- [MODIFIED FOR CATBOOST] ---
    # Identify categorical columns by name
    categorical_cols_names = ['Id', 'LogonType', 'AuthenticationPackageName', 'TicketEncryptionType', 'DestinationPort']
    # Filter to only those present in our final features
    categorical_features_in_model = [col for col in categorical_cols_names if col in final_features_for_model]

    print(f"Identified potential categorical features: {categorical_features_in_model}")

    # Fill NaN values in categorical features - CatBoost requires this
    for col in categorical_features_in_model:
        # Check if the column is NOT numeric before filling with 'Unknown'
        if not pd.api.types.is_numeric_dtype(df_model[col]):
             df_model[col] = df_model[col].fillna('Unknown').astype(str)
        else:
             # If it looks numeric (like Id, LogonType), fill with a numeric placeholder like -1
             df_model[col] = df_model[col].fillna(-1) #.astype(str) # Keep as number if possible for CatBoost

    # Fill NaN in numerical features as well (good practice)
    numerical_cols = [col for col in final_features_for_model if col not in categorical_features_in_model]
    for col in numerical_cols:
        if df_model[col].isnull().any():
            df_model[col] = df_model[col].fillna(df_model[col].median()) # Or use 0, mean, etc.

    print("-> NaN values handled for CatBoost.")

    # --- Step 2: Split Data ---
    X = df_model.drop(label_col, axis=1)
    y = df_model[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"-> Training data: {len(X_train)} samples")
    print(f"-> Test data: {len(X_test)} samples")

    # --- Step 3: Imbalance Weight & Scoring Function ---
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    # CatBoost uses scale_pos_weight for imbalance
    scale_pos_weight_val = count_neg / count_pos if count_pos > 0 else 1

    # --- [MODIFIED] Define scorer to maximize F1-SCORE for the positive class ---
    f1_scorer = make_scorer(f1_score, pos_label=1)
    print(f"-> Tuning objective: Maximizing F1-Score (Malicious Class, using scale_pos_weight: {scale_pos_weight_val:.2f})")

    # --- Step 4: Hyperparameter Tuning - CatBoost ---
    print("\n--- [Tuning CatBoost] ---")

    # Get indices of categorical features for CatBoost
    cat_features_indices = [X_train.columns.get_loc(col) for col in categorical_features_in_model]

    catboost_param_grid = {
        'iterations': [100, 200, 300], # Number of trees (like n_estimators)
        'depth': [6, 8, 10],           # Tree depth
        'learning_rate': [0.05, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5],      # L2 regularization
        'border_count': [32, 64]       # Number of splits for numerical features
        # Add 'subsample' if needed, similar to XGB/LGBM
    }

    # Note: Use scale_pos_weight for imbalance
    catboost_base = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='F1', # Use F1 for internal evaluation metric too
        scale_pos_weight=scale_pos_weight_val,
        cat_features=cat_features_indices, # Tell CatBoost which columns are categorical
        random_state=42,
        verbose=0 # Suppress verbose output during grid search fits
    )

    catboost_search = RandomizedSearchCV(
        estimator=catboost_base,
        param_distributions=catboost_param_grid,
        n_iter=10, # Number of parameter settings that are sampled
        scoring=f1_scorer, # Optimize for F1-score
        cv=3,
        verbose=1, # Show progress for RandomizedSearch
        random_state=42,
        n_jobs=-1 # Use all cores if possible
    )

    start_time_cat = time.time()
    catboost_search.fit(X_train, y_train) # Fit happens here
    end_time_cat = time.time()

    print(f"-> CatBoost tuning completed in {end_time_cat - start_time_cat:.2f} seconds.")
    print(f"    Best F1-Score: {catboost_search.best_score_:.4f}")
    print(f"    Best Params: {catboost_search.best_params_}")
    best_catboost_model = catboost_search.best_estimator_

    # --- Step 5: Final Evaluation and Saving ---
    print("\n--- Final Evaluation and Saving ---")

    # 1. Evaluate CatBoost
    evaluate_and_save_metrics(best_catboost_model, "CatBoost", X_test, y_test, RESULTS_DIR)

    # 2. Save Model
    best_catboost_model.save_model(catboost_model_path)
    print(f"\n-> Tuned CatBoost model saved to '{catboost_model_path}'")

    print("\n--- Process Complete ---")


if __name__ == "__main__":
    # Ensure this points to your generalized dataset
    tune_and_train_catboost('FINAL_DATASET_v4_generalized.csv',
                             catboost_model_path="catboost_detector_tuned.cbm")