import sys, pandas as pd
_, in_csv, text_col, ctx_col, out_csv = sys.argv
df = pd.read_csv(in_csv)
base = df[text_col].astype(str)
if ctx_col in df.columns:
    ctx = df[ctx_col].fillna("").astype(str).str.strip()
    df["__text_for_model"] = ctx.mask(ctx.eq(""), base, other=ctx + " </s> " + base)
else:
    df["__text_for_model"] = base
df.to_csv(out_csv, index=False)
