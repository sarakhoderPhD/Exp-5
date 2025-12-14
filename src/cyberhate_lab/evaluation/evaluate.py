
import argparse, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
from .calibration import expected_calibration_error
from ..data.fairness_metrics import compute_bias_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV with columns: id, pred (probability), label, and identity columns")
    ap.add_argument("--labels", required=False, help="Optional labels CSV to merge by id if preds lacks labels")
    ap.add_argument("--id-cols", required=False, default="", help="Comma-separated identity column names")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    if args.labels and "label" not in df.columns:
        df = df.merge(pd.read_csv(args.labels), on="id", how="left")
    assert "pred" in df.columns and "label" in df.columns, "Need 'pred' and 'label' columns"

    y = df["label"].values
    p = df["pred"].values
    yhat = (p >= 0.5).astype(int)

    metrics = {}
    metrics["roc_auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    metrics.update({"precision": pr, "recall": rc, "f1": f1})
    metrics["ece_15"] = expected_calibration_error(p, y, n_bins=15)

    print("=== Aggregate Metrics ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if args.id_cols:
        id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
        bias_tbl = compute_bias_table(df, id_cols, label_col="label", prob_col="pred")
        print("\n=== Fairness Metrics (per identity) ===")
        print(bias_tbl.to_string(index=False))

if __name__ == "__main__":
    main()
