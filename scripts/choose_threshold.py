import argparse, json, numpy as np, os
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--use-calibrated", action="store_true")
args = ap.parse_args()

rd = args.run_dir
y = np.load(os.path.join(rd, "val_labels.npy"))
pfile = os.path.join(rd, "val_probs_cal.npy" if args.use_calibrated else "val_probs.npy")
if not os.path.exists(pfile):
    # fallback from logits
    logits = np.load(os.path.join(rd, "val_logits.npy"))
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = (exps[:,1] / exps.sum(axis=1))
else:
    p = np.load(pfile)
    if p.ndim == 2:
        p = p[:,1]

def f1_at(t):
    yhat = (p >= t).astype(int)
    tp = ((yhat==1) & (y==1)).sum()
    fp = ((yhat==1) & (y==0)).sum()
    fn = ((yhat==0) & (y==1)).sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    acc = (yhat==y).mean()
    return f1, prec, rec, acc

ths = np.linspace(0.01, 0.99, 99)
scores = [f1_at(t) for t in ths]
best_i = int(np.argmax([s[0] for s in scores]))
best_t = float(ths[best_i])
best = {"threshold": best_t, "f1": scores[best_i][0], "precision": scores[best_i][1],
        "recall": scores[best_i][2], "accuracy": scores[best_i][3],
        "calibrated": args.use_calibrated}

os.makedirs(os.path.join(rd, "audit"), exist_ok=True)
with open(os.path.join(rd, "audit", "threshold.json"), "w") as f:
    json.dump(best, f, indent=2)
with open(os.path.join(rd, "threshold.txt"), "w") as f:
    f.write(str(best_t))

print("Best threshold:", best_t)
print(best)
