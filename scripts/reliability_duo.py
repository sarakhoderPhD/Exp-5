import numpy as np, matplotlib.pyplot as plt, argparse
ap=argparse.ArgumentParser()
ap.add_argument("--y", required=True)
ap.add_argument("--a", required=True)    # plain.npy
ap.add_argument("--b", required=True)    # ctx.npy
ap.add_argument("--name-a", default="plain")
ap.add_argument("--name-b", default="context")
ap.add_argument("--out", required=True)
ap.add_argument("--bins", type=int, default=10)
args=ap.parse_args()

y = np.load(args.y).astype(int)
pa = np.load(args.a).astype(float)
pb = np.load(args.b).astype(float)

def curve(p, y, k):
    edges = np.linspace(0,1,k+1); mids=[]; mean_p=[]; frac_pos=[]
    for i in range(k):
        lo,hi=edges[i],edges[i+1]
        idx = (p>=lo)&(p<hi) if i<k-1 else (p>=lo)&(p<=hi)
        if idx.sum()==0: continue
        mids.append((lo+hi)/2)
        mean_p.append(p[idx].mean())
        frac_pos.append(y[idx].mean())
    return np.array(mids), np.array(mean_p), np.array(frac_pos)

import matplotlib.pyplot as plt
plt.figure(figsize=(6,5), dpi=140)
for p, name in [(pa,args.name_a),(pb,args.name_b)]:
    x, m, f = curve(p,y,args.bins)
    plt.plot(m, f, marker='o', label=name)

plt.plot([0,1],[0,1],'--',linewidth=1)
plt.xlabel('Mean predicted probability (bin)')
plt.ylabel('Empirical positive rate')
plt.title('Reliability (external)')
plt.legend()
plt.tight_layout()
plt.savefig(args.out)
print("Wrote", args.out)
