#!/usr/bin/env python
import argparse, numpy as np, sys, os, math

# --- deps for metrics ---
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_recall_fscore_support
except Exception as e:
    sys.stderr.write("scikit-learn is required. Try:  pip install scikit-learn\n")
    raise

# McNemar (exact when small)
p_mcnemar = None
try:
    from statsmodels.stats.contingency_tables import mcnemar
except Exception:
    mcnemar = None
    try:
        from scipy.stats import binomtest
        def _mcnemar_exact(b, c):
            # exact binomial test for discordant pairs
            return binomtest(k=min(b, c), n=b+c, p=0.5, alternative="two-sided").pvalue
    except Exception:
        binomtest = None

def binarize(p, thr=0.5):
    return (p >= thr).astype(int)

def safe_clip(p):
    p = np.asarray(p, float)
    return np.clip(p, 0.0, 1.0)

def compute_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = safe_clip(p)
    yhat = binarize(p, thr)
    out = {}
    # AUROC/AP may fail if one class missing in the sample; handle gracefully
    try: out["auroc"] = roc_auc_score(y, p)
    except: out["auroc"] = float("nan")
    try: out["ap"] = average_precision_score(y, p)
    except: out["ap"] = float("nan")
    out["acc"] = accuracy_score(y, yhat)
    out["f1"]  = f1_score(y, yhat, zero_division=0)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, zero_division=0, average="binary")
    out["precision"] = pr; out["recall"] = rc
    return out

def bootstrap_diff(y, pa, pb, nboot=2000, seed=13, thr=0.5):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pa = safe_clip(pa); pb = safe_clip(pb)
    n = len(y)
    keys = ["auroc","ap","acc","f1"]
    A = {k:[] for k in keys}; B = {k:[] for k in keys}; D = {k:[] for k in keys}
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        ya, aa, bb = y[idx], pa[idx], pb[idx]
        ma = compute_metrics(ya, aa, thr)
        mb = compute_metrics(ya, bb, thr)
        for k in keys:
            A[k].append(ma[k]); B[k].append(mb[k]); D[k].append(mb[k]-ma[k])
    def ci(x): 
        x = np.asarray(x, float)
        return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))
    stats = {}
    for k in keys:
        stats[k] = dict(
            a_mean=float(np.nanmean(A[k])),
            b_mean=float(np.nanmean(B[k])),
            d_mean=float(np.nanmean(D[k])),
            a_ci=ci(A[k]),
            b_ci=ci(B[k]),
            d_ci=ci(D[k]),
        )
    return stats

def mcnemar_pvalue(y, pa, pb, thr=0.5):
    y  = np.asarray(y).astype(int)
    ha = binarize(pa, thr); hb = binarize(pb, thr)
    a_right = (ha == y); b_right = (hb == y)
    b = int(( a_right & ~b_right).sum())  # A correct, B wrong
    c = int((~a_right &  b_right).sum())  # A wrong,  B correct
    if b + c == 0:
        return 1.0, b, c
    if mcnemar is not None:
        p = mcnemar([[0,b],[c,0]], exact=(b+c)<=25).pvalue
        return float(p), b, c
    if 'binomtest' in globals() and binomtest is not None:
        return float(_mcnemar_exact(b, c)), b, c
    # fallback: normal approx
    p = 0.0 if b==c else 2*min(b, c)/float(b+c)
    return float(p), b, c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", required=True, help="path to y npy")
    ap.add_argument("--a", required=True, help="probs A npy (baseline)")
    ap.add_argument("--b", required=True, help="probs B npy (treatment)")
    ap.add_argument("--names", nargs=2, default=["A","B"], help="names for A and B")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    y  = np.load(args.y).astype(int)
    pa = np.load(args.a).astype(float)
    pb = np.load(args.b).astype(float)

    n = min(len(y), len(pa), len(pb))
    y, pa, pb = y[:n], pa[:n], pb[:n]

    nameA, nameB = args.names

    # point estimates
    mA = compute_metrics(y, pa, args.thr)
    mB = compute_metrics(y, pb, args.thr)

    # bootstrap
    bs = bootstrap_diff(y, pa, pb, nboot=args.nboot, seed=13, thr=args.thr)

    # McNemar
    p_mcn, b, c = mcnemar_pvalue(y, pa, pb, args.thr)

    lines = []
    lines.append(f"Paired comparison: {nameA} (A) vs {nameB} (B)")
    lines.append(f"N={n}, threshold={args.thr}")
    lines.append("")
    for k, label in [("auroc","AUROC"), ("ap","Average Precision"), ("acc","Accuracy"), ("f1","F1@{:.2f}".format(args.thr))]:
        s = bs[k]
        lines.append(f"{label}:")
        lines.append(f"  {nameA}: {mA[k]:.4f}  (bootstrap 95% CI {s['a_ci'][0]:.4f}–{s['a_ci'][1]:.4f})")
        lines.append(f"  {nameB}: {mB[k]:.4f}  (bootstrap 95% CI {s['b_ci'][0]:.4f}–{s['b_ci'][1]:.4f})")
        lines.append(f"  Δ({nameB}-{nameA}): {s['d_mean']:.4f}  (95% CI {s['d_ci'][0]:.4f}–{s['d_ci'][1]:.4f})")
        lines.append("")
    lines.append(f"McNemar on correctness @ {args.thr:.2f}: p={p_mcn:.4g}  (A-correct/B-wrong b={b}, A-wrong/B-correct c={c})")
    lines.append("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
