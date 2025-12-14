#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def avg_words(s):
    s = s.fillna("").astype(str)
    return s.apply(lambda x: len(x.split())).mean()

def trunc_rate(text_series, max_tokens=256):
    """
    Approximation if you don't have tokenizer logs:
    treat long word-count as proxy.
    If you *do* have token-level trunc logs elsewhere, use those instead.
    """
    s = text_series.fillna("").astype(str)
    # rough heuristic: ~1.3 words per token in English-ish social text
    approx_tokens = s.apply(lambda x: len(x.split()) / 1.3)
    return (approx_tokens > max_tokens).mean()

def compute_block(df, label_col, child_col, ctx_col=None):
    out = {}
    out["N"] = len(df)
    out["Pos Rate"] = df[label_col].mean()

    out["Avg Words (Child)"] = avg_words(df[child_col])

    if ctx_col is None:
        out["Ctx Coverage"] = np.nan
        out["Avg Words (Ctx)"] = np.nan
        out["Avg Words (Composed)"] = out["Avg Words (Child)"]
        out["Trunc Rate"] = trunc_rate(df[child_col])
    else:
        ctx = df[ctx_col].fillna("").astype(str)
        child = df[child_col].fillna("").astype(str)

        out["Ctx Coverage"] = (ctx.str.len() > 0).mean()
        out["Avg Words (Ctx)"] = avg_words(df[ctx_col])

        composed = ("CONTEXT: " + ctx + "\n---\nCHILD: " + child)
        out["Avg Words (Composed)"] = avg_words(composed)

        out["Trunc Rate"] = trunc_rate(composed)

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-train", required=True)
    parser.add_argument("--internal-val", required=True)
    parser.add_argument("--external", required=True)

    parser.add_argument("--label-col", default="label")
    parser.add_argument("--child-col", default="child")
    parser.add_argument("--ctx-col", default="parent")

    parser.add_argument("--out", default="outputs/table_3_1.csv")
    args = parser.parse_args()

    rows = []

    # Load
    tr = pd.read_csv(args.internal_train)
    va = pd.read_csv(args.internal_val)
    ex = pd.read_csv(args.external)

    # Plain vs context blocks
    # Train
    rows.append({
        "Split": "Train (plain)", "Source": "internal",
        **compute_block(tr, args.label_col, args.child_col, ctx_col=None)
    })
    rows.append({
        "Split": "Train (context)", "Source": "internal",
        **compute_block(tr, args.label_col, args.child_col, ctx_col=args.ctx_col)
    })

    # Validation
    rows.append({
        "Split": "Validation (plain)", "Source": "internal",
        **compute_block(va, args.label_col, args.child_col, ctx_col=None)
    })
    rows.append({
        "Split": "Validation (context)", "Source": "internal",
        **compute_block(va, args.label_col, args.child_col, ctx_col=args.ctx_col)
    })

    # External
    rows.append({
        "Split": "External (plain)", "Source": "external",
        **compute_block(ex, args.label_col, args.child_col, ctx_col=None)
    })
    rows.append({
        "Split": "External (context)", "Source": "external",
        **compute_block(ex, args.label_col, args.child_col, ctx_col=args.ctx_col)
    })

    out_df = pd.DataFrame(rows)

    # Format helpful rounding
    for c in ["Pos Rate", "Ctx Coverage", "Trunc Rate"]:
        out_df[c] = out_df[c].astype(float).round(4)
    for c in ["Avg Words (Child)", "Avg Words (Ctx)", "Avg Words (Composed)"]:
        out_df[c] = out_df[c].astype(float).round(2)

    # Save
    import os
    os.makedirs("outputs", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(out_df.to_string(index=False))
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()