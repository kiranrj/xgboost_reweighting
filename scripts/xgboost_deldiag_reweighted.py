import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

def cooks_distance(X_train, y_train, model, dtrain, dtest, y_train_full, y_test, baseline_train_preds, baseline_test_preds):
    n = X_train.shape[0]
    k = X_train.shape[1]
    influence_scores = np.zeros(n)
    train_loss_changes = []
    test_loss_changes = []
    
    for i in tqdm(range(n), desc="Running Deletion Diagnostics", ncols=100, unit=" sample"):
        X_train_new = np.delete(X_train, i, axis=0)
        y_train_new = np.delete(y_train, i, axis=0)

        dtrain_new = xgb.DMatrix(X_train_new, label=y_train_new)
        model_new = xgb.train(model, dtrain_new, num_boost_round=50)

        new_train_preds = model_new.predict(dtrain_new)
        new_test_preds = model_new.predict(dtest)

        train_loss_change = log_loss(y_train_new, new_train_preds) - log_loss(y_train_full, baseline_train_preds)
        test_loss_change = log_loss(y_test, new_test_preds) - log_loss(y_test, baseline_test_preds)
        
        train_loss_changes.append(train_loss_change)
        test_loss_changes.append(test_loss_change)
        
        residuals = (y_test - new_test_preds) ** 2
        influence_scores[i] = np.sum(residuals) / ((k + 1) * np.var(residuals))
    
    return influence_scores, train_loss_changes, test_loss_changes

def analyze_dataset(X, y, dataset_name, feature_names):
    print(f"\n--- Analyzing {dataset_name} Dataset ---")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    # Initial XGBoost Model
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0}
    model = xgb.train(params, dtrain, num_boost_round=50)

    # baseline feature importance
    baseline_gain_importance = model.get_score(importance_type='gain')
    baseline_split_importance = model.get_score(importance_type='weight')
    # baseline_gain_importance = model.get_score(importance_type='total_gain')
    # baseline_split_importance = model.get_score(importance_type='total_cover')

    # baseline predictions
    baseline_train_preds = model.predict(dtrain)
    baseline_test_preds = model.predict(dtest)

    # cook's Distance based influence scores
    influence_scores, train_loss_changes, test_loss_changes = cooks_distance(
        X_train, y_train, params, dtrain, dtest, y_train, y_test, baseline_train_preds, baseline_test_preds
    )

    # threshold for removing influential points
    threshold = np.mean(influence_scores) + 2 * np.std(influence_scores)
    influential_indices = np.where(influence_scores > threshold)[0]

    print(f"Identified {len(influential_indices)} Influential Data Points for Further Analysis.")

    # Convert lists to numpy arrays for element-wise comparison
    train_loss_changes = np.array(train_loss_changes)
    test_loss_changes = np.array(test_loss_changes)

    # 1. Outlier Detection using Z-score
    z_scores = np.abs(zscore(X_train))
    outlier_flags = (z_scores > 3).any(axis=1)
    outlier_indices = influential_indices[outlier_flags[influential_indices]]

    # 2. Mislabel Detection (high residuals)
    baseline_preds_binary = np.round(baseline_train_preds)
    residuals = np.abs(y_train - baseline_preds_binary)
    mislabel_flags = residuals > 0  # misclassified points
    mislabel_indices = influential_indices[mislabel_flags[influential_indices]]

    # 3. Important Edge Cases (neither outliers nor mislabeled)
    edge_case_indices = np.setdiff1d(influential_indices, np.union1d(outlier_indices, mislabel_indices))

    # initialize all weights = 1
    sample_weights = np.ones(X_train.shape[0])

    # # reweight based on category
    # sample_weights[outlier_indices] = 0.0  # significantly reduce weight for outliers
    # sample_weights[mislabel_indices] = 0.0   # moderately reduce weight for mislabeled points
    # sample_weights[edge_case_indices] = 0.0  # increase weight for important edge cases
    # Adjust weights based on category and loss impact
    for idx in outlier_indices:
        if train_loss_changes[idx] > 0 and test_loss_changes[idx] > 0:
            sample_weights[idx] = 0.0  # significantly reduce weight if harmful to both train and test
        elif train_loss_changes[idx] > 0 or test_loss_changes[idx] > 0:
            sample_weights[idx] = 0.0  # moderately reduce weight if harmful to either train or test
        else:
            sample_weights[idx] = 0.1   # default reduced weight for outliers

    for idx in mislabel_indices:
        if train_loss_changes[idx] > 0 and test_loss_changes[idx] > 0:
            sample_weights[idx] = 0.0   # significantly reduce weight if harmful to both train and test
        elif train_loss_changes[idx] > 0 or test_loss_changes[idx] > 0:
            sample_weights[idx] = 0.0  # moderately reduce weight if harmful to either train or test
        else:
            sample_weights[idx] = 0.2   # default reduced weight for mislabeled points

    for idx in edge_case_indices:
        if train_loss_changes[idx] < 0 and test_loss_changes[idx] < 0:
            sample_weights[idx] = 0.0   # significantly increase weight if helpful to both train and test
        elif train_loss_changes[idx] < 0 or test_loss_changes[idx] < 0:
            sample_weights[idx] = 0.0   # moderately increase weight if helpful to either train or test
        else:
            sample_weights[idx] = 0.5   # default increased weight for edge cases

    # Retrain the model using adjusted sample weights
    dtrain_reweighted = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, feature_names=feature_names)
    model_reweighted = xgb.train(params, dtrain_reweighted, num_boost_round=50)

    # Evaluate the reweighted model
    reweighted_test_preds = model_reweighted.predict(dtest)
    reweighted_test_preds_binary = np.round(reweighted_test_preds)
    reweighted_log_loss = log_loss(y_test, reweighted_test_preds)
    reweighted_accuracy = accuracy_score(y_test, reweighted_test_preds_binary)
    reweighted_auc = roc_auc_score(y_test, reweighted_test_preds)
    reweighted_precision = precision_score(y_test, reweighted_test_preds_binary, zero_division=0)
    reweighted_recall = recall_score(y_test, reweighted_test_preds_binary)
    reweighted_f1 = f1_score(y_test, reweighted_test_preds_binary)

    # Compare performances
    print("\n--- Performance Comparison (Baseline vs. Reweighted) ---")
    print(f"Log Loss Before: {log_loss(y_test, baseline_test_preds):.4f} | After Reweighting: {reweighted_log_loss:.4f}")
    print(f"Accuracy Before: {accuracy_score(y_test, np.round(baseline_test_preds)):.4f} | After Reweighting: {reweighted_accuracy:.4f}")
    print(f"AUC Before: {roc_auc_score(y_test, baseline_test_preds):.4f} | After Reweighting: {reweighted_auc:.4f}")
    print(f"Precision Before: {precision_score(y_test, np.round(baseline_test_preds), zero_division=0):.4f} | After Reweighting: {reweighted_precision:.4f}")
    print(f"Recall Before: {recall_score(y_test, np.round(baseline_test_preds)):.4f} | After Reweighting: {reweighted_recall:.4f}")
    print(f"F1 Score Before: {f1_score(y_test, np.round(baseline_test_preds)):.4f} | After Reweighting: {reweighted_f1:.4f}")

    # Plot feature importance comparison (Baseline vs. Reweighted)
    reweighted_gain_importance = model_reweighted.get_score(importance_type='total_gain')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pie(baseline_gain_importance.values(), labels=baseline_gain_importance.keys(), autopct='%1.1f%%')
    plt.title("Feature Gain Importance Before")

    plt.subplot(1, 2, 2)
    plt.pie(reweighted_gain_importance.values(), labels=reweighted_gain_importance.keys(), autopct='%1.1f%%')
    plt.title("Feature Gain Importance After Reweighting")
    plt.show()

    # Visualize influential points categories
    plt.figure(figsize=(10, 6))
    categories = ['Outliers', 'Mislabeled', 'Edge Cases']
    counts = [len(outlier_indices), len(mislabel_indices), len(edge_case_indices)]
    sns.barplot(x=categories, y=counts, hue=categories, palette='viridis', legend=False)
    plt.title('Influential Data Points Categorization')
    plt.ylabel('Number of Points')
    plt.show()

# --- Dataset 1: Indian Liver Patient Dataset (ILPD) ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
feature_names = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
                                  'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Proteins', 
                                  'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']
df_ilpd = pd.read_csv(url, names=feature_names)
df_ilpd['Gender'] = LabelEncoder().fit_transform(df_ilpd['Gender'])
X_ilpd = df_ilpd.drop(columns=['Dataset']).values
y_ilpd = np.where(df_ilpd['Dataset'] == 2, 0, 1)
analyze_dataset(X_ilpd, y_ilpd, "ILPD (Liver Disease)", feature_names[:-1])

# --- Dataset 2: Breast Cancer Dataset ---
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
X_cancer = cancer_data.data
y_cancer = cancer_data.target
analyze_dataset(X_cancer, y_cancer, "Breast Cancer", cancer_data.feature_names.tolist())

# --- Dataset 3: Statlog (German Credit Data) ---
from ucimlrepo import fetch_ucirepo

# Fetch dataset
german_credit_data = fetch_ucirepo(id=144)

# Extract features and target
X_german = german_credit_data.data.features
y_german = german_credit_data.data.targets.squeeze()

# Encode categorical variables and explicitly convert to numeric
X_german_encoded = pd.get_dummies(X_german, drop_first=True).astype(float)

# Convert target to binary (1 = Good, 0 = Bad)
y_german_binary = np.where(y_german == 1, 1, 0)

# Analyze dataset
analyze_dataset(X_german_encoded.values, y_german_binary, "Statlog (German Credit Data)", X_german_encoded.columns.tolist())

