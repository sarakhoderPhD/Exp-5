import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
ap=argparse.ArgumentParser()
ap.add_argument("--y", required=True)
ap.add_argument("--plain", required=True)
ap.add_argument("--ctx", required=True)
ap.add_argument("--out", required=True)
args=ap.parse_args()

y = np.load(args.y).astype(int)
pa = np.load(args.plain).astype(float)
pb = np.load(args.ctx).astype(float)

def pack(name, p):
    from math import isfinite
    def f1_at_thr(p,t):
        yhat=(p>=t).astype(int)
        tp=((yhat==1)&(y==1)).sum(); fp=((yhat==1)&(y==0)).sum(); fn=((yhat==0)&(y==1)).sum()
        return 0. if (2*tp+fp+fn)==0 else (2*tp)/(2*tp+fp+fn)
    ths=np.linspace(0.01,0.99,99); f1s=np.array([f1_at_thr(p,t) for t in ths])
    i=f1s.argmax(); best_thr=float(ths[i]); best_f1=float(f1s[i])
    row = dict(
        model=name,
        roc_auc=float(roc_auc_score(y,p)),
        pr_auc=float(average_precision_score(y,p)),
        f1_at_0_5=float(f1_at_thr(p,0.5)),
        best_f1=best_f1,
        best_thr=best_thr,
        pos_rate=float((p>=0.5).mean()),
    )
    return row

out = pd.DataFrame([pack("plain",pa), pack("context",pb)])
out.to_csv(args.out, index=False)
print("Wrote", args.out); print(out.to_string(index=False))
