import pandas as pd, numpy as np, matplotlib.pyplot as plt, argparse
ap=argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--probs", required=True)
ap.add_argument("--out",   required=True)
ap.add_argument("--bins", type=int, default=15)
args=ap.parse_args()

y = pd.read_csv(args.labels)["label"].astype(int).to_numpy()
pdf = pd.read_csv(args.probs)
pcol = "pred_cal" if "pred_cal" in pdf.columns else ("pred" if "pred" in pdf.columns else pdf.select_dtypes(float).columns[0])
p = pdf[pcol].astype(float).to_numpy()

bins = np.linspace(0,1,args.bins+1)
ids = np.digitize(p, bins)-1
acc, conf, count = [], [], []
for b in range(args.bins):
    mask = ids==b
    if mask.sum()==0: continue
    conf.append(p[mask].mean())
    acc.append(y[mask].mean())
    count.append(mask.sum())

plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--',lw=1)
plt.plot(conf, acc, marker='o')
plt.xlabel("Confidence"); plt.ylabel("Empirical accuracy")
plt.title("Reliability diagram (external)")
plt.tight_layout(); plt.savefig(args.out, dpi=200)
print(f"Saved {args.out}")
