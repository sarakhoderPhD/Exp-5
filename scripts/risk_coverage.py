#!/usr/bin/env python
import argparse, numpy as np
from cyberhate_lab.training.utils import metrics_binary
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True, help="npy probs for class 1")
ap.add_argument("--labels", required=True, help="npy labels")
ap.add_argument("--out", default="risk_coverage.png")
args = ap.parse_args()

p = np.load(args.probs)
y = np.load(args.labels)

conf = np.maximum(p, 1-p)
order = np.argsort(-conf)
p_sorted, y_sorted = p[order], y[order]

coverages = np.linspace(0.1, 1.0, 10)
risks = []
for c in coverages:
    k = max(1, int(len(p_sorted)*c))
    from cyberhate_lab.training.utils import metrics_binary as _mb
    m = _mb(y_sorted[:k], p_sorted[:k])
    risks.append(1 - m["f1"])

plt.figure()
plt.plot(coverages, risks, marker="o")
plt.xlabel("Coverage")
plt.ylabel("Risk (1 - F1)")
plt.title("Risk–Coverage Curve")
plt.grid(True)
plt.savefig(args.out, dpi=150)
print("Saved", args.out)
