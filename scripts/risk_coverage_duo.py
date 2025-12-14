import argparse, numpy as np, matplotlib.pyplot as plt
ap=argparse.ArgumentParser()
ap.add_argument("--y", required=True)
ap.add_argument("--a", required=True)   # plain.npy
ap.add_argument("--b", required=True)   # ctx.npy
ap.add_argument("--name-a", default="plain"); ap.add_argument("--name-b", default="context")
ap.add_argument("--out", required=True); ap.add_argument("--bins", type=int, default=40)
args=ap.parse_args()

y = np.load(args.y).astype(int); pa = np.load(args.a).astype(float); pb = np.load(args.b).astype(float)
def rc(p, y):
    s = np.abs(p-0.5)
    order = np.argsort(s)               # lowest confidence first (to abstain)
    cov = []; f1s=[]
    for k in range(1, len(y)+1):
        keep = order[k-1:]              # keep most confident tail
        yy = y[keep]; pp = p[keep] >= 0.5
        tp=(pp & (yy==1)).sum(); fp=(pp & (yy==0)).sum(); fn=((yy==1) & (~pp)).sum()
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        cov.append(len(keep)/len(y)); f1s.append(f1)
    return np.array(cov), np.array(f1s)

cov_a, f1_a = rc(pa,y); cov_b, f1_b = rc(pb,y)
aurc_a = np.trapz(f1_a, cov_a); aurc_b = np.trapz(f1_b, cov_b)

plt.figure(figsize=(6,5), dpi=140)
plt.plot(cov_a, f1_a, label=f"{args.name_a} (AURC={aurc_a:.3f})")
plt.plot(cov_b, f1_b, label=f"{args.name_b} (AURC={aurc_b:.3f})")
plt.xlabel("Coverage (kept %)"); plt.ylabel("F1")
plt.title("Risk–Coverage (external)"); plt.legend(); plt.tight_layout(); plt.savefig(args.out)
print(f"Wrote {args.out}\nAURC: {args.name_a}={aurc_a:.6f}, {args.name_b}={aurc_b:.6f}")
