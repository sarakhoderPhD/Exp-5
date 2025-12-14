import argparse, os, json, numpy as np, pandas as pd

def infer_group_cols(df):
    cand = []
    for c in df.columns:
        if c.lower() in {"text","pred","pred_cal","label_hat"}: continue
        if pd.api.types.is_bool_dtype(df[c]) or (
            pd.api.types.is_integer_dtype(df[c]) and set(df[c].dropna().unique()) <= {0,1}
        ):
            cand.append(c)
    return cand

ap = argparse.ArgumentParser()
ap.add_argument("--eval-csv", required=True)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--group-cols", default="")
ap.add_argument("--prob-col", default="pred_cal")  # falls back to "pred" if missing
args = ap.parse_args()

thr_path = os.path.join(args.run_dir, "threshold.txt")
thr = float(open(thr_path).read().strip()) if os.path.exists(thr_path) else 0.5

df = pd.read_csv(args.eval_csv)
pcol = args.prob_col if args.prob_col in df.columns else ("pred" if "pred" in df.columns else None)
if pcol is None: raise SystemExit("No prob column found (pred_cal or pred).")

if "label_hat" not in df.columns:
    df["label_hat"] = (df[pcol] >= thr).astype(int)

groups = [g for g in args.group_cols.split(",") if g] or infer_group_cols(df)
if not groups:
    print("No group columns detected; exiting.")
    raise SystemExit(0)

overall = {
    "N": int(len(df)),
    "mean_score": float(df[pcol].mean()),
    "pos_rate": float((df[pcol] >= thr).mean())
}

rows = []
for g in groups:
    sub = df[df[g]==1]
    if len(sub)==0: continue
    r = {
        "group": g,
        "N": int(len(sub)),
        "mean_score": float(sub[pcol].mean()),
        "pos_rate": float((sub[pcol] >= thr).mean()),
    }
    r["mean_score_diff"] = r["mean_score"] - overall["mean_score"]
    r["pos_rate_diff"]   = r["pos_rate"]   - overall["pos_rate"]
    rows.append(r)

out_dir = os.path.join(args.run_dir, "audit")
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(out_dir, "fairness_unlabeled.csv"), index=False)
with open(os.path.join(out_dir, "fairness_unlabeled_overall.json"), "w") as f:
    json.dump({"overall": overall, "threshold": thr, "prob_col": pcol, "groups": groups}, f, indent=2)

print("Wrote:", os.path.join(out_dir, "fairness_unlabeled.csv"))
