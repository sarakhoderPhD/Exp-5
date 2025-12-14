import argparse, pandas as pd, numpy as np, sys, os

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    if df.shape[1] == 1:
        return df.columns[0]
    raise SystemExit(f"Could not find any of {candidates} in columns: {list(df.columns)}")

def safe_prf(y_true, y_pred):
    tp = ((y_true==1) & (y_pred==1)).sum()
    fp = ((y_true==0) & (y_pred==1)).sum()
    fn = ((y_true==1) & (y_pred==0)).sum()
    tn = ((y_true==0) & (y_pred==0)).sum()
    prec = tp / (tp+fp) if (tp+fp) else 0.0
    rec  = tp / (tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    acc  = (tp+tn) / max(1, len(y_true))
    return prec, rec, f1, acc

def ece(probs, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idxs = np.digitize(probs, bins) - 1
    total = len(labels)
    e = 0.0
    for b in range(n_bins):
        mask = idxs==b
        if not np.any(mask): continue
        conf = probs[mask].mean()
        acc = (labels[mask]==(probs[mask]>=0.5)).mean()
        e += (mask.sum()/total) * abs(acc - conf)
    return float(e)

def try_auc(y, p):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return float(roc_auc_score(y, p)), float(average_precision_score(y, p))
    except Exception:
        return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="CSV with a 'label' column (0/1) or single column")
    ap.add_argument("--probs", required=True, help="CSV with 'pred_cal' or 'pred' column")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ydf = pd.read_csv(args.labels)
    y_col = "label" if "label" in ydf.columns else pick_col(ydf, ydf.columns)
    y = ydf[y_col].astype(int).to_numpy()

    pdf = pd.read_csv(args.probs)
    p_col = pick_col(pdf, ["pred_cal","pred","prob","score"])
    p = pdf[p_col].astype(float).to_numpy()

    n = min(len(y), len(p))
    if len(y)!=len(p):
        print(f"[warn] length mismatch: labels={len(y)}, probs={len(p)}; truncating to {n}", file=sys.stderr)
    y, p = y[:n], p[:n]

    pred05 = (p >= 0.5).astype(int)
    prec05, rec05, f105, acc05 = safe_prf(y, pred05)

    grid = np.linspace(0.05, 0.95, 19)
    best = (0.0, 0.5)  # (f1, thr)
    for thr in grid:
        pr, rc, f1, _ = safe_prf(y, (p>=thr).astype(int))
        if f1 > best[0] or (abs(f1-best[0])<1e-9 and abs(thr-0.5)<abs(best[1]-0.5)):
            best = (float(f1), float(thr))
    thr_best = best[1]
    predB = (p >= thr_best).astype(int)
    precB, recB, f1B, accB = safe_prf(y, predB)

    roc_auc, pr_auc = try_auc(y, p)
    e = ece(p, y, n_bins=10)

    out = pd.DataFrame([{
        "n": int(n),
        "pos_rate": float(y.mean()),
        "roc_auc": (None if roc_auc is None else roc_auc),
        "pr_auc": (None if pr_auc is None else pr_auc),
        "ece_10": e,
        "thr@0.5": 0.5,
        "precision@0.5": prec05,
        "recall@0.5": rec05,
        "f1@0.5": f105,
        "acc@0.5": acc05,
        "thr@best_f1": thr_best,
        "precision@best_f1": precB,
        "recall@best_f1": recB,
        "f1@best_f1": f1B,
        "acc@best_f1": accB,
        "probs_col": p_col,
        "label_col": y_col
    }])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}\n{out.to_string(index=False)}")

if __name__ == "__main__":
    main()
