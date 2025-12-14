import argparse, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
ap=argparse.ArgumentParser()
ap.add_argument("--y", required=True)
ap.add_argument("--a", required=True)  # plain.npy
ap.add_argument("--b", required=True)  # ctx.npy
ap.add_argument("--name-a", default="plain")
ap.add_argument("--name-b", default="context")
ap.add_argument("--out-roc", required=True)
ap.add_argument("--out-pr", required=True)
args=ap.parse_args()

y = np.load(args.y).astype(int)
pa = np.load(args.a).astype(float)
pb = np.load(args.b).astype(float)

def do_roc(p, name):
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr, auc(fpr, tpr)

def do_pr(p, name):
    prec, rec, _ = precision_recall_curve(y, p)
    return rec, prec, auc(rec, prec)

# ROC
plt.figure(figsize=(6,5), dpi=140)
for p,name in [(pa,args.name_a),(pb,args.name_b)]:
    fpr,tpr,A = do_roc(p,name)
    plt.plot(fpr,tpr,label=f"{name} (AUC={A:.3f})")
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC (external)"); plt.legend(); plt.tight_layout()
plt.savefig(args.out_roc)

# PR
plt.figure(figsize=(6,5), dpi=140)
for p,name in [(pa,args.name_a),(pb,args.name_b)]:
    r,pr,A = do_pr(p,name)
    plt.plot(r,pr,label=f"{name} (AP={A:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall (external)"); plt.legend(); plt.tight_layout()
plt.savefig(args.out_pr)

print("Wrote", args.out_roc, "and", args.out_pr)
