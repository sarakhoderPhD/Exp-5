import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

def pick_probs(run):
    p_cal = os.path.join(run, "val_probs_cal.npy")
    p_raw = os.path.join(run, "val_probs.npy")
    if os.path.exists(p_cal): return np.load(p_cal)
    return np.load(p_raw)

def summary(run):
    y = np.load(os.path.join(run, "val_labels.npy"))
    p = pick_probs(run)
    yhat = (p>=0.5).astype(int)
    return {
        "run": run,
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "F1@0.5": f1_score(y, yhat),
        "Accuracy@0.5": accuracy_score(y, yhat),
    }

rows = [summary("runs/roberta_bin"), summary("runs/roberta_ctx")]
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("runs/compare_ctx_vs_bin.csv", index=False)
print("\nWrote runs/compare_ctx_vs_bin.csv")
