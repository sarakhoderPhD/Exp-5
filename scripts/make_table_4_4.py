#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ---- Helpers ---------------------------------------------------------------

LABEL_CANDIDATES = ["label", "y", "target", "gold", "is_hate", "is_toxic", "toxic", "abusive"]
PROB_CANDIDATES  = ["p", "prob", "proba", "score", "pred_prob", "p1", "prob_1", "hate_prob", "toxic_prob"]

def guess_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def load_labels(y_path, label_col=None):
    df = pd.read_csv(y_path)
    col = label_col or guess_column(df, LABEL_CANDIDATES)
    if col is None:
        raise ValueError(
            f"Could not find label column in {y_path}. "
            f"Please pass --label-col. Available columns: {list(df.columns)}"
        )
    y = df[col].astype(int).to_numpy()
    return y, df

def load_probs(p_path, prob_col=None):
    df = pd.read_csv(p_path)
    col = prob_col or guess_column(df, PROB_CANDIDATES)
    if col is None:
        # if only one numeric column, take it
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 1:
            col = numeric_cols[0]
        else:
            raise ValueError(
                f"Could not find probability column in {p_path}. "
                f"Please pass --prob-col-a/--prob-col-b. "
                f"Available columns: {list(df.columns)}"
            )
    p = df[col].astype(float).to_numpy()
    return p, df

def risk_coverage(y, p, deltas, tau=0.5):
    rows = []
    for d in deltas:
        accept = np.abs(p - tau) >= d
        cov = accept.mean() if len(accept) else 0.0
        if accept.sum() == 0:
            risk = np.nan
        else:
            yhat = (p >= tau).astype(int)
            risk = (yhat[accept] != y[accept]).mean()
        rows.append({"delta": d, "coverage": cov, "risk": risk})
    return rows

# ---- Main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create Table 4.4 operating points from external calibrated probabilities."
    )
    parser.add_argument("--y", required=True, help="CSV containing ground-truth labels (or full external set).")
    parser.add_argument("--a", required=True, help="CSV with external calibrated probs for sentence-only model.")
    parser.add_argument("--b", required=True, help="CSV with external calibrated probs for context model.")
    parser.add_argument("--out", default="outputs/table_4_4.csv", help="Output CSV path.")
    parser.add_argument("--label-col", default=None, help="Label column name (if auto-detect fails).")
    parser.add_argument("--prob-col-a", default=None, help="Probability column name for model A (if auto-detect fails).")
    parser.add_argument("--prob-col-b", default=None, help="Probability column name for model B (if auto-detect fails).")
    parser.add_argument("--deltas", default="0,0.05,0.10,0.15",
                        help="Comma-separated delta values for abstention band.")
    parser.add_argument("--tau", type=float, default=0.5, help="Decision threshold.")
    args = parser.parse_args()

    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]

    y, _ = load_labels(args.y, args.label_col)
    p_a, _ = load_probs(args.a, args.prob_col_a)
    p_b, _ = load_probs(args.b, args.prob_col_b)

    if not (len(y) == len(p_a) == len(p_b)):
        raise ValueError(
            f"Length mismatch: len(y)={len(y)}, len(a)={len(p_a)}, len(b)={len(p_b)}. "
            "Make sure these files are for the same external split and same ordering."
        )

    rows_a = risk_coverage(y, p_a, deltas, tau=args.tau)
    rows_b = risk_coverage(y, p_b, deltas, tau=args.tau)

    out_rows = []
    for r in rows_a:
        out_rows.append({"model": "Sentence-only", **r})
    for r in rows_b:
        out_rows.append({"model": "Context", **r})

    out_df = pd.DataFrame(out_rows)

    # Save CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Print a neat console view you can copy
    print("\nTable 4.4 operating points (external)\n")
    # pivot to look table-like
    show = out_df.copy()
    show["coverage"] = (show["coverage"] * 100).round(1)  # percent
    show["risk"] = (show["risk"] * 100).round(2)          # percent
    print(show.to_string(index=False))

    print(f"\nSaved: {out_path.resolve()}\n")

if __name__ == "__main__":
    main()