import argparse, numpy as np, pandas as pd
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--probs", required=True)
ap.add_argument("--coverages", default="0.7,0.85,0.95")
ap.add_argument("--out", required=True)
args = ap.parse_args()

y = np.load(args.labels); p = np.load(args.probs)
covs = [float(x) for x in args.coverages.split(",")]
rows=[]
order = np.argsort(-np.maximum(p, 1-p))  # confidence = max(p,1-p)
for c in covs:
    k = int(len(p)*c)
    keep_idx = order[:k]
    thr = np.partition(-np.maximum(p,1-p), k-1)[k-1] * -1  # confidence threshold
    yk, pk = y[keep_idx], p[keep_idx]
    yhat = (pk>=0.5).astype(int)
    rows.append({"coverage": c, "conf_threshold": float(thr), "F1": f1_score(yk, yhat), "kept": int(k)})
pd.DataFrame(rows).to_csv(args.out, index=False)
print(pd.DataFrame(rows).to_string(index=False))
