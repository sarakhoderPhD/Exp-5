# abstract_numbers.py (v2)
from pathlib import Path
import pandas as pd

root = Path("experiments/exp4_context")

def pick_col(cols, candidates):
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(cols_l):
            if cand in c:
                return cols[i]
    return None

def read_csv_safely(path):
    try:
        df = pd.read_csv(path)
        print(f"[ok] Loaded: {path}")
        print("     Columns:", list(df.columns))
        return df
    except Exception as e:
        print(f"[warn] Could not load {path}: {e}")
        return None

# 1) ΔAUROC / ΔAP
onepager = read_csv_safely(root / "exp4_onepager.csv")
d_auroc = d_ap = None
if onepager is not None:
    cols = list(onepager.columns)
    model_col = pick_col(cols, ["model"])
    if model_col:
        m = onepager.copy()
        m[model_col] = m[model_col].astype(str).str.lower().replace({"ctx":"context"})
        auroc_col = pick_col(cols, ["auroc", "roc_auc"])
        ap_col    = pick_col(cols, ["ap", "avg_precision", "average_precision", "pr_auc", "auprc"])
        if auroc_col and ap_col and {"plain","context"}.issubset(set(m[model_col])):
            d_auroc = m.loc[m[model_col]=="context", auroc_col].iloc[0] - m.loc[m[model_col]=="plain", auroc_col].iloc[0]
            d_ap    = m.loc[m[model_col]=="context", ap_col   ].iloc[0] - m.loc[m[model_col]=="plain", ap_col   ].iloc[0]

# 2) ECE deltas (post-calibration): context − plain
ece = read_csv_safely(root / "ece_external.csv")
ece10_delta = ece20_delta = None
if ece is not None:
    model_col = pick_col(ece.columns, ["model"])
    e10_col   = pick_col(ece.columns, ["ece_10"])
    e20_col   = pick_col(ece.columns, ["ece_20"])
    if model_col and e10_col:
        ex = ece.copy()
        ex[model_col] = ex[model_col].astype(str).str.lower().replace({"ctx":"context"})
        if {"plain","context"}.issubset(set(ex[model_col])):
            ece10_delta = ex.loc[ex[model_col]=="context", e10_col].iloc[0] - ex.loc[ex[model_col]=="plain", e10_col].iloc[0]
            if e20_col:
                ece20_delta = ex.loc[ex[model_col]=="context", e20_col].iloc[0] - ex.loc[ex[model_col]=="plain", e20_col].iloc[0]

# 3) Risk–coverage (optional; prints only if file exists)
def nearest_coverage(df, model_name, target_err):
    def pick(cols, cs): 
        return next((c for c in cols if any(k in c.lower() for k in cs)), None)
    model_col = pick(df.columns, ["model"])
    err_col   = pick(df.columns, ["selective_error","sel_error","error","risk"])
    cov_col   = pick(df.columns, ["coverage","cov"])
    if not all([model_col, err_col, cov_col]): 
        return None
    sub = df.copy()
    sub[model_col] = sub[model_col].astype(str).str.lower().replace({"ctx":"context"})
    sub = sub[sub[model_col]==model_name]
    if sub.empty: return None
    idx = (sub[err_col] - target_err).abs().argsort()
    return float(sub.iloc[idx.iloc[0]][cov_col])

coverage_improvements = []
rc = read_csv_safely(root / "risk_coverage_external.csv")
if rc is not None:
    for t in (0.01, 0.02):
        c_ctx   = nearest_coverage(rc, "context", t)
        c_plain = nearest_coverage(rc, "plain", t)
        if c_ctx is not None and c_plain is not None:
            coverage_improvements.append((t, (c_ctx - c_plain)*100.0))

# --- Print ready-to-paste lines ---
if d_auroc is not None: print(f"ΔAUROC: {d_auroc:+.3f}")
else:                  print("ΔAUROC: (not found)")

if d_ap is not None:   print(f"ΔAP: {d_ap:+.3f}")
else:                  print("ΔAP: (not found)")

if ece10_delta is not None:
    sign = "lower" if ece10_delta < 0 else "higher"
    print(f"ECE@10 delta (context − plain): {ece10_delta:+.3f} ({sign} is better if negative)")
else:
    print("ECE@10 delta: (not found)")

if ece20_delta is not None:
    sign = "lower" if ece20_delta < 0 else "higher"
    print(f"ECE@20 delta (context − plain): {ece20_delta:+.3f} ({sign} is better if negative)")

if coverage_improvements:
    for t, imp_pp in coverage_improvements:
        print(f"Coverage improvement at {int(t*100)}% selective error: {imp_pp:+.1f} pp")
else:
    print("Coverage improvements: (risk_coverage_external.csv not present)")