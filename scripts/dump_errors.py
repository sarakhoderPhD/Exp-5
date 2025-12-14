import pandas as pd, numpy as np, sys

ctx_csv, plain_csv, test_csv, out_csv = sys.argv[1:]

pc = pd.read_csv(ctx_csv)    # must contain 'pred_cal'
pp = pd.read_csv(plain_csv)  # must contain 'pred_cal'
df = pd.read_csv(test_csv)   # must contain 'ctx','text','label'

p_ctx   = pc["pred_cal"].astype(float)
p_plain = pp["pred_cal"].astype(float)
y       = df["label"].astype(int)

pred_ctx   = (p_ctx >= 0.5).astype(int)
pred_plain = (p_plain >= 0.5).astype(int)

base = pd.DataFrame({
    "label": y,
    "p_plain": p_plain, "p_ctx": p_ctx,
    "pred_plain": pred_plain, "pred_ctx": pred_ctx,
    "ctx": df.get("ctx",""),
    "text": df.get("text",""),
})

flips  = base[base.pred_ctx != base.pred_plain].copy()
fix_fp = base[(pred_plain==1) & (pred_ctx==0) & (y==0)].copy().nlargest(50, "p_plain")
fix_fn = base[(pred_plain==0) & (pred_ctx==1) & (y==1)].copy().nlargest(50, "p_ctx")

flips["section"]  = "flips"
fix_fp["section"] = "fix_fp"
fix_fn["section"] = "fix_fn"

out = pd.concat([flips.head(200), fix_fp, fix_fn], axis=0)
out.to_csv(out_csv, index=False)
print(f"Wrote {out_csv} with {len(out)} rows")
