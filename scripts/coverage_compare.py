import sys, numpy as np, matplotlib.pyplot as plt
# usage: python scripts/coverage_compare.py external/out/y_ext.npy name1.npy label1 name2.npy label2 ...
y = np.load(sys.argv[1])
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
def curve(p):
    # simple coverage sweep: abstain below descending thresholds
    thr = np.linspace(0,1,201)
    cov, f1 = [], []
    for t in thr:
        keep = p>=t
        if keep.sum()==0: continue
        yk, pk = y[keep], (p[keep]>=0.5).astype(int)
        tp = ((pk==1)&(yk==1)).sum(); fp = ((pk==1)&(yk==0)).sum(); fn = ((pk==0)&(yk==1)).sum()
        f1.append(0 if tp==0 else 2*tp/(2*tp+fp+fn))
        cov.append(keep.mean())
    return np.array(cov), np.array(f1)
plt.figure(figsize=(6,4))
for pth, lab in pairs:
    p = np.load(pth)
    c,f = curve(p)
    plt.plot(c, f, label=lab)
plt.xlabel("Coverage"); plt.ylabel("F1"); plt.title("Coverage–F1 (external)"); plt.legend(); plt.tight_layout()
plt.savefig("external/out/rc_compare.png"); print("Wrote external/out/rc_compare.png")
