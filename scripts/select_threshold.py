import pandas as pd, numpy as np, argparse, itertools

ap=argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--probs",  required=True)
ap.add_argument("--out",    required=True)
ap.add_argument("--abstain_cost", type=float, default=0.25)
args=ap.parse_args()

y = pd.read_csv(args.labels)["label"].astype(int).to_numpy()
pdf = pd.read_csv(args.probs)
pcol = "pred_cal" if "pred_cal" in pdf.columns else ("pred" if "pred" in pdf.columns else pdf.select_dtypes(float).columns[0])
p = pdf[pcol].astype(float).to_numpy()

def eval_band(band):
    lo, hi = 0.5-band, 0.5+band
    abstain = (p>lo) & (p<hi)
    covered = ~abstain
    y_cov, p_cov = y[covered], p[covered]
    yhat = (p_cov>=0.5).astype(int)
    acc = (yhat==y_cov).mean() if covered.any() else 0.0
    from sklearn.metrics import f1_score
    f1  = f1_score(y_cov, yhat) if covered.any() else 0.0
    coverage = covered.mean()
    # Expected risk with abstain cost (lower is better)
    risk = (1-acc)*coverage + args.abstain_cost*(1-coverage)
    return {"band":band,"coverage":coverage,"acc_cov":acc,"f1_cov":f1,"risk":risk}

grid = [round(b,3) for b in np.linspace(0,0.4,41)]
rows = [eval_band(b) for b in grid]
df = pd.DataFrame(rows).sort_values("risk")
best = df.iloc[0]
df.to_csv(args.out, index=False)
print("Wrote", args.out)
print("Best:", best.to_dict())
