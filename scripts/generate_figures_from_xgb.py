
import argparse, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    perf_rows = [
        {"Dataset":"ILPD","Variant":"Before","LogLoss":0.5472,"Accuracy":0.7350,"Precision":0.8111,"Recall":0.8391,"F1":0.8249,"AUC":0.7636},
        {"Dataset":"ILPD","Variant":"After","LogLoss":0.4838,"Accuracy":0.7521,"Precision":0.8085,"Recall":0.8736,"F1":0.8398,"AUC":0.7885},
        {"Dataset":"Breast Cancer","Variant":"Before","LogLoss":0.1303,"Accuracy":0.9561,"Precision":0.9583,"Recall":0.9718,"F1":0.9650,"AUC":0.9905},
        {"Dataset":"Breast Cancer","Variant":"After","LogLoss":0.0870,"Accuracy":0.9561,"Precision":0.9583,"Recall":0.9718,"F1":0.9650,"AUC":0.9948},
        {"Dataset":"German Credit","Variant":"Before","LogLoss":float('nan'),"Accuracy":float('nan'),"Precision":float('nan'),"Recall":float('nan'),"F1":float('nan'),"AUC":float('nan')},
        {"Dataset":"German Credit","Variant":"After","LogLoss":float('nan'),"Accuracy":float('nan'),"Precision":float('nan'),"Recall":float('nan'),"F1":float('nan'),"AUC":float('nan')},
    ]
    write_perf_table_tex(perf_rows, outdir/"perf_table.tex")

    influence_placeholder = {"ILPD": [], "Breast Cancer": [], "German Credit": []}
    save_boxplot(influence_placeholder, outdir/"fig_influence_boxplots.pdf")

    save_feature_gain_pies(before_dict={}, after_dict={}, 
                           out_before=outdir/"fig_feat_gain_before.pdf",
                           out_after=outdir/"fig_feat_gain_after.pdf")

    save_bar_counts({"Outliers":0,"Mislabeled":0,"Edge Cases":0}, outdir/"fig_category_counts.pdf")

if __name__ == "__main__":
    main()
