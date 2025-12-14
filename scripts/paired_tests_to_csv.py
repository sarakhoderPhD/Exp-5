#!/usr/bin/env python
import argparse, numpy as np, os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

def clip01(x): 
    x = np.asarray(x, float); 
    return np.clip(x, 0, 1)

def metrics(y, p, thr=0.5):
    y = np.asarray(y, int); p = clip01(p); yhat = (p>=thr).astype(int)
    out = {}
    try: out["auroc"] = roc_auc_score(y, p)
    except: out["auroc"] = float("nan")
    try: out["ap"] = average_precision_score(y, p)
    except: out["ap"] = float("nan")
    out["acc"] = accuracy_score(y, yhat)
    out["f1"]  = f1_score(y, yhat, zero_division=0)
    return out

def boot(y, pa, pb, nboot=2000, thr=0.5, seed=13):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int); pa = clip01(pa); pb = clip01(pb)
    n = len(y)
    keys = ["auroc","ap","acc","f1"]
    A = {k:[] for k in keys}; B = {k:[] for k in keys}; D = {k:[] for k in keys}
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        mA = metrics(y[idx], pa[idx], thr); mB = metrics(y[idx], pb[idx], thr)
        for k in keys:
            A[k].append(mA[k]); B[k].append(mB[k]); D[k].append(mB[k]-mA[k])

    def ci(v): v=np.array(v,float); return np.nanpercentile(v,2.5), np.nanpercentile(v,97.5), np.nanmean(v)
    stats={}
    for k in keys:
        aL,aU,aM = ci(A[k]); bL,bU,bM = ci(B[k]); dL,dU,dM = ci(D[k])
        stats[k] = dict(a_mean=aM,b_mean=bM,d_mean=dM,a_lo=aL,a_hi=aU,b_lo=bL,b_hi=bU,d_lo=dL,d_hi=dU)
    return stats

def mcnemar_p(y, pa, pb, thr=0.5):
    y = np.asarray(y,int); ha=(clip01(pa)>=thr).astype(int); hb=(clip01(pb)>=thr).astype(int)
    a_right=(ha==y); b_right=(hb==y)
    b = int(( a_right & ~b_right).sum())  # A correct, B wrong
    c = int((~a_right &  b_right).sum())  # A wrong,  B correct
    if b+c==0: return 1.0,b,c
    p = mcnemar([[0,b],[c,0]], exact=(b+c)<=25).pvalue
    return float(p),b,c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--names", nargs=2, default=["A","B"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    y  = np.load(args.y).astype(int)
    pa = np.load(args.a).astype(float)
    pb = np.load(args.b).astype(float)
    n=min(len(y),len(pa),len(pb)); y,pa,pb = y[:n],pa[:n],pb[:n]

    from math import isnan
    A=metrics(y,pa,args.thr); B=metrics(y,pb,args.thr); S=boot(y,pa,pb,thr=args.thr)
    p,b,c = mcnemar_p(y,pa,pb,args.thr)

    import csv, os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["metric","A_name","A_mean","A_lo","A_hi","B_name","B_mean","B_lo","B_hi","Delta_mean","Delta_lo","Delta_hi","N","thr","McNemar_p","b","c"])
        for k,label in [("auroc","AUROC"),("ap","AP"),("acc","ACC"),("f1","F1")]:
            s=S[k]
            w.writerow([label,args.names[0],A[k],s["a_lo"],s["a_hi"],args.names[1],B[k],s["b_lo"],s["b_hi"],s["d_mean"],s["d_lo"],s["d_hi"],n,args.thr,p,b,c])
    print(f"Wrote {args.out}")
    
if __name__=="__main__":
    main()
