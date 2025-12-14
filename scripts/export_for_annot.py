import argparse, pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pred-csv", required=True)     # e.g., eval/preds_cal.csv
ap.add_argument("--text-col", default="text")
ap.add_argument("--prob-col", default="pred_cal")
ap.add_argument("--k", type=int, default=500)
ap.add_argument("--out", required=True)
args = ap.parse_args()

df = pd.read_csv(args.pred_csv)
pcol = args.prob_col if args.prob_col in df.columns else "pred"
if pcol not in df.columns: raise SystemExit("No probability column found.")

df["uncert"] = (df[pcol] - 0.5).abs()
uncertain = df.nsmallest(args.k, "uncert")

# optional QC tails
pos_strong = df.nlargest(50, pcol)
neg_strong = df.nsmallest(50, pcol)

batch = pd.concat([uncertain, pos_strong, neg_strong]).drop_duplicates()
keep_cols = [c for c in [args.text_col, pcol, "uncert"] if c in batch.columns]
aux_cols  = [c for c in batch.columns if c not in keep_cols]
out = batch[keep_cols + aux_cols]
out.to_csv(args.out, index=False)
print("Wrote:", args.out, " rows:", len(out))
