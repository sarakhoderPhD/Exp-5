import sys, pandas as pd, numpy as np
in_csv, text_col, ctx_col, variant, out_csv = sys.argv[1:]
df = pd.read_csv(in_csv)
ctx = df[ctx_col].fillna("").astype(str)

if variant=="drop":
    mix = df[text_col].astype(str)
elif variant=="shuffle":
    ctx2 = ctx.sample(frac=1, random_state=42).reset_index(drop=True)
    mix = (ctx2 + " </s> " + df[text_col].astype(str)).str.strip()
elif variant.startswith("truncate"):
    # truncate:<N_words>
    n = int(variant.split(":")[1])
    ctx2 = ctx.apply(lambda s: " ".join(s.split()[:n]))
    mix = (ctx2 + " </s> " + df[text_col].astype(str)).str.strip()
else:
    raise SystemExit("variant must be drop | shuffle | truncate:<N>")

out = df.copy()
out["__text_for_model"] = mix
out.to_csv(out_csv, index=False)
print("Wrote", out_csv)
