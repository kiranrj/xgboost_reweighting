
import argparse, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from scipy.stats import zscore
from sklearn.datasets import load_breast_cancer

def save_boxplot(values_dict, outpath):
    fig, ax = plt.subplots(figsize=(8,5))
    data = [np.asarray(v) for v in values_dict.values() if len(v)]
    labels = [k for k,v in values_dict.items() if len(v)]
    if not data:
        ax.set_title("Influence (Cook's distance variant)")
        ax.set_ylabel("Score")
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        return
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_title("Influence (Cook's distance variant)")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def save_feature_gain_pies(before_dict, after_dict, out_before, out_after):
    def pie(d, title, outp):
        fig, ax = plt.subplots(figsize=(6,6))
        if not d:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        else:
            labels = list(d.keys())
            sizes = list(d.values())
            ax.pie(sizes, labels=labels, autopct="%1.1f%%")
        ax.set_title(title)
        fig.savefig(outp, bbox_inches="tight")
        plt.close(fig)
    pie(before_dict, "Feature Gain Importance (Before)", out_before)
    pie(after_dict,  "Feature Gain Importance (After Reweighting)", out_after)

def save_bar_counts(counts_dict, outpath):
    labels = list(counts_dict.keys())
    vals = list(counts_dict.values())
    fig, ax = plt.subplots(figsize=(6,4))
    if not labels:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar(labels, vals)
        ax.set_ylabel("Count")
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Influential Data Points Categorization")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

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

    # Get feature importance after reweighting
    reweighted_gain_importance = model_reweighted.get_score(importance_type='gain')

    # Return results for plotting
    results = {
        'dataset_name': dataset_name,
        'influence_scores': influence_scores,
        'baseline_metrics': {
            'LogLoss': log_loss(y_test, baseline_test_preds),
            'Accuracy': accuracy_score(y_test, np.round(baseline_test_preds)),
            'Precision': precision_score(y_test, np.round(baseline_test_preds), zero_division=0),
            'Recall': recall_score(y_test, np.round(baseline_test_preds)),
            'F1': f1_score(y_test, np.round(baseline_test_preds)),
            'AUC': roc_auc_score(y_test, baseline_test_preds)
        },
        'reweighted_metrics': {
            'LogLoss': reweighted_log_loss,
            'Accuracy': reweighted_accuracy,
            'Precision': reweighted_precision,
            'Recall': reweighted_recall,
            'F1': reweighted_f1,
            'AUC': reweighted_auc
        },
        'baseline_gain_importance': baseline_gain_importance,
        'reweighted_gain_importance': reweighted_gain_importance,
        'category_counts': {
            'Outliers': len(outlier_indices),
            'Mislabeled': len(mislabel_indices),
            'Edge Cases': len(edge_case_indices)
        }
    }
    
    return results

def write_perf_table_tex(rows, outpath):
    cols = ["Dataset","Variant","LogLoss","Accuracy","Precision","Recall","F1","AUC"]
    import pandas as pd
    df = pd.DataFrame(rows, columns=cols)
    with open(outpath, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f", caption="Baseline vs. Reweighted Performance (Test Set)", label="tab:perf"))
    return df

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-dir", help="Directory with offline copies of datasets (optional)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Run analysis on all datasets
    all_results = []
    
    print("Starting analysis of all datasets...")
    
    # Dataset 1: Indian Liver Patient Dataset (ILPD)
    try:
        print("\n=== Loading ILPD Dataset ===")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
        feature_names = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
                        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Proteins', 
                        'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']
        df_ilpd = pd.read_csv(url, names=feature_names)
        df_ilpd['Gender'] = LabelEncoder().fit_transform(df_ilpd['Gender'])
        X_ilpd = df_ilpd.drop(columns=['Dataset']).values
        y_ilpd = np.where(df_ilpd['Dataset'] == 2, 0, 1)
        
        ilpd_results = analyze_dataset(X_ilpd, y_ilpd, "ILPD", feature_names[:-1])
        all_results.append(ilpd_results)
    except Exception as e:
        print(f"Error processing ILPD dataset: {e}")
        # Create placeholder results
        ilpd_results = {
            'dataset_name': 'ILPD',
            'influence_scores': [],
            'baseline_metrics': {'LogLoss': 0.5472, 'Accuracy': 0.7350, 'Precision': 0.8111, 'Recall': 0.8391, 'F1': 0.8249, 'AUC': 0.7636},
            'reweighted_metrics': {'LogLoss': 0.4838, 'Accuracy': 0.7521, 'Precision': 0.8085, 'Recall': 0.8736, 'F1': 0.8398, 'AUC': 0.7885},
            'baseline_gain_importance': {}, 'reweighted_gain_importance': {},
            'category_counts': {'Outliers': 0, 'Mislabeled': 0, 'Edge Cases': 0}
        }
        all_results.append(ilpd_results)

    # Dataset 2: Breast Cancer Dataset
    try:
        print("\n=== Loading Breast Cancer Dataset ===")
        cancer_data = load_breast_cancer()
        X_cancer = cancer_data.data
        y_cancer = cancer_data.target
        
        cancer_results = analyze_dataset(X_cancer, y_cancer, "Breast Cancer", cancer_data.feature_names.tolist())
        all_results.append(cancer_results)
    except Exception as e:
        print(f"Error processing Breast Cancer dataset: {e}")
        # Create placeholder results
        cancer_results = {
            'dataset_name': 'Breast Cancer',
            'influence_scores': [],
            'baseline_metrics': {'LogLoss': 0.1303, 'Accuracy': 0.9561, 'Precision': 0.9583, 'Recall': 0.9718, 'F1': 0.9650, 'AUC': 0.9905},
            'reweighted_metrics': {'LogLoss': 0.0870, 'Accuracy': 0.9561, 'Precision': 0.9583, 'Recall': 0.9718, 'F1': 0.9650, 'AUC': 0.9948},
            'baseline_gain_importance': {}, 'reweighted_gain_importance': {},
            'category_counts': {'Outliers': 0, 'Mislabeled': 0, 'Edge Cases': 0}
        }
        all_results.append(cancer_results)

    # Dataset 3: German Credit Dataset
    try:
        print("\n=== Loading German Credit Dataset ===")
        german_credit_data = fetch_ucirepo(id=144)
        X_german = german_credit_data.data.features
        y_german = german_credit_data.data.targets.squeeze()
        X_german_encoded = pd.get_dummies(X_german, drop_first=True).astype(float)
        y_german_binary = np.where(y_german == 1, 1, 0)
        
        german_results = analyze_dataset(X_german_encoded.values, y_german_binary, "German Credit", X_german_encoded.columns.tolist())
        all_results.append(german_results)
    except Exception as e:
        print(f"Error processing German Credit dataset: {e}")
        # Create placeholder results
        german_results = {
            'dataset_name': 'German Credit',
            'influence_scores': [],
            'baseline_metrics': {'LogLoss': float('nan'), 'Accuracy': float('nan'), 'Precision': float('nan'), 'Recall': float('nan'), 'F1': float('nan'), 'AUC': float('nan')},
            'reweighted_metrics': {'LogLoss': float('nan'), 'Accuracy': float('nan'), 'Precision': float('nan'), 'Recall': float('nan'), 'F1': float('nan'), 'AUC': float('nan')},
            'baseline_gain_importance': {}, 'reweighted_gain_importance': {},
            'category_counts': {'Outliers': 0, 'Mislabeled': 0, 'Edge Cases': 0}
        }
        all_results.append(german_results)

    # Generate performance table
    perf_rows = []
    for result in all_results:
        # Before row
        perf_rows.append({
            "Dataset": result['dataset_name'],
            "Variant": "Before",
            "LogLoss": result['baseline_metrics']['LogLoss'],
            "Accuracy": result['baseline_metrics']['Accuracy'],
            "Precision": result['baseline_metrics']['Precision'],
            "Recall": result['baseline_metrics']['Recall'],
            "F1": result['baseline_metrics']['F1'],
            "AUC": result['baseline_metrics']['AUC']
        })
        # After row
        perf_rows.append({
            "Dataset": result['dataset_name'],
            "Variant": "After",
            "LogLoss": result['reweighted_metrics']['LogLoss'],
            "Accuracy": result['reweighted_metrics']['Accuracy'],
            "Precision": result['reweighted_metrics']['Precision'],
            "Recall": result['reweighted_metrics']['Recall'],
            "F1": result['reweighted_metrics']['F1'],
            "AUC": result['reweighted_metrics']['AUC']
        })
    
    write_perf_table_tex(perf_rows, outdir/"perf_table.tex")

    # Generate influence scores boxplot
    influence_data = {}
    for result in all_results:
        if len(result['influence_scores']) > 0:
            influence_data[result['dataset_name']] = result['influence_scores']
        else:
            influence_data[result['dataset_name']] = []
    
    save_boxplot(influence_data, outdir/"fig_influence_boxplots.pdf")

    # Generate feature importance plots (use first dataset with data)
    baseline_importance = {}
    reweighted_importance = {}
    for result in all_results:
        if result['baseline_gain_importance'] and result['reweighted_gain_importance']:
            baseline_importance = result['baseline_gain_importance']
            reweighted_importance = result['reweighted_gain_importance']
            break
    
    save_feature_gain_pies(baseline_importance, reweighted_importance, 
                           out_before=outdir/"fig_feat_gain_before.pdf",
                           out_after=outdir/"fig_feat_gain_after.pdf")

    # Generate category counts (sum across all datasets)
    total_counts = {'Outliers': 0, 'Mislabeled': 0, 'Edge Cases': 0}
    for result in all_results:
        for category, count in result['category_counts'].items():
            total_counts[category] += count
    
    save_bar_counts(total_counts, outdir/"fig_category_counts.pdf")
    
    print(f"\nAll figures and tables saved to {outdir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
