import argparse, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in-csv", required=True)
ap.add_argument("--out-csv", required=True)
ap.add_argument("--text-col", default="text")
ap.add_argument("--context-cols", default="parent_text,quoted_text,in_reply_to_text,source_caption")
ap.add_argument("--max_ctx_len", type=int, default=384)
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
cand = [c.strip() for c in args.context_cols.split(",") if c.strip() in df.columns]

def build_ctx(row):
    parts = []
    for c in cand:
        v = str(row[c]).strip()
        if v and v.lower()!="nan":
            parts.append(v)
    if not parts: return ""
    ctx = " [CTX] " + " [CTX_SEP] ".join(parts)
    return ctx[:args.max_ctx_len*2]  # cheap guard; tokenizer still truncates

df["ctx"] = df.apply(build_ctx, axis=1)
df.to_csv(args.out_csv, index=False)
print("Wrote:", args.out_csv, " | ctx coverage:", (df["ctx"].str.len()>0).mean())
