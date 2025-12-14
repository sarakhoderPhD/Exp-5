import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
ap=argparse.ArgumentParser()
ap.add_argument("--plain", required=True)
ap.add_argument("--ctx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--metric", default=None, help="Metric column to use (e.g., f1@0.5). Auto-detect if omitted.")
args=ap.parse_args()

sp = pd.read_csv(args.plain)
sc = pd.read_csv(args.ctx)

for k in ["slice", "group", "identity", "category"]:
    if k in sp.columns and k in sc.columns:
        key = k
        break
else:
    commons = list(set(sp.columns) & set(sc.columns))
    for drop in ["n", "pos_rate"]:
        if drop in commons:
            commons.remove(drop)
    if not commons:
        raise SystemExit("No common join key between slices files.")
    key = commons[0]

if args.metric and (args.metric in sp.columns) and (args.metric in sc.columns):
    mcol = args.metric
else:
    cand = [c for c in sp.columns if "f1" in c.lower()]
    if not cand:
        raise SystemExit("No F1-like metric column found in plain slices.")
    pref = [c for c in cand if "best" in c.lower()]
    mcol = pref[0] if pref else cand[0]

a = sp[[key, mcol]].rename(columns={mcol: "f1_plain"})
b = sc[[key, mcol]].rename(columns={mcol: "f1_ctx"})
m = a.merge(b, on=key, how="inner")
m["delta"] = m["f1_ctx"] - m["f1_plain"]
m = m.sort_values("delta", ascending=True)

plt.figure(figsize=(8,6), dpi=140)
ypos = np.arange(len(m))
plt.barh(ypos, m["delta"])
plt.yticks(ypos, m[key])
plt.axvline(0, color="k", lw=1)
plt.xlabel("ΔF1 (context − plain)")
plt.title("Per-identity F1 change with context")
plt.tight_layout()
plt.savefig(args.out)
print("Wrote", args.out)
