import sys, pandas as pd
in_csv, composed_col, out_csv = sys.argv[1:]
df = pd.read_csv(in_csv)
sep = "</s>"
ctx_out, txt_out = [], []
# fall back to original text if no sep
orig_txt = df.get("text", "").astype(str)
for s, t in zip(df.get(composed_col, "").astype(str), orig_txt):
    if sep in s:
        left, right = s.split(sep, 1)
        ctx_out.append(left.strip())
        txt_out.append(right.strip())
    else:
        ctx_out.append("")
        txt_out.append(t)
df["ctx"] = ctx_out
df["text"] = txt_out
df.to_csv(out_csv, index=False)
print(f"Wrote {out_csv} with columns: {list(df.columns)}")
