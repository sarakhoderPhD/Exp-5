import pandas as pd, matplotlib.pyplot as plt
import numpy as np, sys

plain = pd.read_csv("external/out/slices_plain.csv")
ctx   = pd.read_csv("external/out/slices_ctx.csv")

keep = ["slice","n","roc_auc","pr_auc","ece_10","f1@best_f1","acc@best_f1","precision@best_f1","recall@best_f1"]
plain = plain[keep].rename(columns={c:f"{c}_plain" for c in keep if c!="slice"})
ctx   = ctx[keep].rename(columns={c:f"{c}_ctx"   for c in keep if c!="slice"})

m = plain.merge(ctx, left_on="slice", right_on="slice", how="inner")
m["d_f1"]  = m["f1@best_f1_ctx"]  - m["f1@best_f1_plain"]
m["d_ece"] = m["ece_10_ctx"]      - m["ece_10_plain"]

m.to_csv("external/out/slices_compare.csv", index=False)
print("Wrote external/out/slices_compare.csv")

def barplot(df, col, fname, title):
    df = df.copy()
    df = df[df["slice"]!="ALL"].sort_values(col)
    plt.figure(figsize=(10,6))
    plt.barh(df["slice"], df[col])
    plt.axvline(0, ls="--", lw=1, color="k")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    print(f"Saved {fname}")

barplot(m, "d_f1",  "external/out/slices_delta_f1.png",  "ΔF1 (context − plain)")
barplot(m, "d_ece", "external/out/slices_delta_ece.png", "ΔECE (context − plain)")
