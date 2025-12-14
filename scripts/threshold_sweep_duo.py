import argparse, numpy as np, matplotlib.pyplot as plt
ap=argparse.ArgumentParser()
ap.add_argument("--y", required=True)
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--name-a", default="plain")
ap.add_argument("--name-b", default="context")
ap.add_argument("--out", required=True)
args=ap.parse_args()

y = np.load(args.y).astype(int)
pa = np.load(args.a).astype(float)
pb = np.load(args.b).astype(float)
ths = np.linspace(0.01,0.99,99)

def f1_at(p, t):
    yhat = (p>=t).astype(int)
    tp = ((yhat==1)&(y==1)).sum()
    fp = ((yhat==1)&(y==0)).sum()
    fn = ((yhat==0)&(y==1)).sum()
    return 0. if (2*tp+fp+fn)==0 else (2*tp)/(2*tp+fp+fn)

fa = [f1_at(pa,t) for t in ths]
fb = [f1_at(pb,t) for t in ths]

plt.figure(figsize=(6,5), dpi=140)
plt.plot(ths, fa, label=args.name_a)
plt.plot(ths, fb, label=args.name_b)
plt.xlabel("Threshold"); plt.ylabel("F1")
plt.title("F1 vs Threshold (external)")
plt.legend(); plt.tight_layout(); plt.savefig(args.out)

best_a = ths[int(np.argmax(fa))]; best_b = ths[int(np.argmax(fb))]
print(f"Best F1: {args.name_a} {max(fa):.4f} @ {best_a:.2f} | {args.name_b} {max(fb):.4f} @ {best_b:.2f}")
