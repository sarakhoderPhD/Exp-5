import argparse, re, subprocess, pandas as pd, numpy as np, os, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--texts", required=True)
ap.add_argument("--text-col", default="text")
ap.add_argument("--ctx-col", default="ctx")
ap.add_argument("--labels", required=True)
ap.add_argument("--probs", required=True)
ap.add_argument("--name", default="model")
ap.add_argument("--out", required=True)
args = ap.parse_args()

df_txt = pd.read_csv(args.texts)

# Combine "<ctx> <text>" robustly (ctx optional)
if args.ctx_col and args.ctx_col in df_txt.columns:
    ctx_series = df_txt[args.ctx_col].fillna("").astype(str)
else:
    ctx_series = pd.Series([""] * len(df_txt), index=df_txt.index)

text_series = df_txt[args.text_col].fillna("").astype(str)
full = (ctx_series + " " + text_series).str.strip().str.lower()

# Load labels & probs and decide which prob column to use
labels = pd.read_csv(args.labels)
probs  = pd.read_csv(args.probs)
if "pred_cal" in probs.columns:
    pcol = "pred_cal"
elif "pred" in probs.columns:
    pcol = "pred"
else:
    # last resort: first numeric column
    num_cols = probs.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise SystemExit("No numeric probability column found in probs CSV.")
    pcol = num_cols[0]

groups = {
    "muslim":    [r"\bmuslim", r"\bislam", r"\bhalal"],
    "jewish":    [r"\bjew", r"\bjewish", r"\bisrael(i)?\b"],
    "black":     [r"\bblack\b", r"\bafrican"],
    "asian":     [r"\basian\b", r"\bpakistani\b", r"\bbangla", r"\bindian\b", r"\bchinese\b"],
    "lgbtq":     [r"\blgbt", r"\bgay\b", r"\blesbian\b", r"\btrans"],
    "women":     [r"\bwoman\b", r"\bwomen\b", r"\bfemale\b", r"\bgirl\b"],
    "men":       [r"\bman\b", r"\bmen\b", r"\bmale\b", r"\bboy\b"],
    "immigrant": [r"\bimmigra", r"\brefugee", r"\bmigrant"],
}
def mask_for(patterns, s):
    reg = re.compile("|".join(patterns), flags=re.I)
    return s.str.contains(reg, na=False)

def run_metrics(y_slice, p_slice_vals):
    # Wrap slices into temp CSVs with expected column names
    y_df = pd.DataFrame({"label": y_slice})
    p_df = pd.DataFrame({pcol: pd.to_numeric(p_slice_vals, errors="coerce")})

    ytmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv"); ytmp.close()
    ptmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv"); ptmp.close()
    outtmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv"); outtmp.close()

    y_df.to_csv(ytmp.name, index=False)
    p_df.to_csv(ptmp.name, index=False)

    subprocess.check_call([
        "python", "scripts/metrics_simple.py",
        "--labels", ytmp.name, "--probs", ptmp.name, "--out", outtmp.name
    ])
    m = pd.read_csv(outtmp.name).iloc[0].to_dict()
    return m

rows = []
# Overall slice
idx_all = full.index == full.index
m_all = run_metrics(labels.loc[idx_all, "label"], probs.loc[idx_all, pcol])
m_all["slice"] = "ALL"; m_all["n"] = int(idx_all.sum())
rows.append(m_all)

# Identity slices
for g, pats in groups.items():
    idx = mask_for(pats, full)
    if idx.sum() == 0:
        continue
    m = run_metrics(labels.loc[idx, "label"], probs.loc[idx, pcol])
    m["slice"] = g; m["n"] = int(idx.sum())
    rows.append(m)

pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"Wrote {args.out}")