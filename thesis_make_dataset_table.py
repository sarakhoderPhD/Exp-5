#!/usr/bin/env python3
"""
thesis_make_dataset_table.py

Builds a small, quantitative comparison table for internal vs external datasets
used in the thesis. It looks for these CSVs (use --help for overrides):

  Internal (plain):      data/train.csv, data/val.csv
  Internal (context):    data/train_ctx.csv, data/val_ctx.csv
  External (plain):      external/test.csv
  External (context):    external/test_ctx.csv or external/test_with_ctx.csv

Expected columns:
  - text (child message)          [required]
  - label (0/1)                   [required]
  - ctx (parent turn)             [optional for context splits]
  - __text_for_model (composed)   [optional; created by your preprocess script]

Outputs (created if missing):
  experiments/exp4_context/table_dataset_diffs.csv   # numeric, machine-friendly
  experiments/exp4_context/table_dataset_diffs.md    # formatted, thesis-ready

Example:
  python3 thesis_make_dataset_table.py
  python3 thesis_make_dataset_table.py --train data/train.csv --ext_ctx external/test_ctx.csv
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


# --------------------------- utilities ---------------------------

def _exists_any(paths):
    """Return the first existing path from a list (or None)."""
    for p in paths:
        if p and Path(p).is_file():
            return p
    return None


def _word_count(s) -> int:
    if s is None:
        return 0
    try:
        txt = str(s)
    except Exception:
        return 0
    # normalise whitespace
    return len(txt.strip().split()) if txt.strip() else 0


def _compose(ctx, text) -> str:
    if ctx is None or (isinstance(ctx, float) and np.isnan(ctx)) or str(ctx).strip() == "":
        return str(text) if text is not None else ""
    return f"CONTEXT: {ctx}\n---\nCHILD: {text}"


def _summarise_split(df: pd.DataFrame, split_name: str, source: str) -> dict:
    """Compute summary stats for one split."""
    out = {
        "split": split_name,
        "source": source,  # internal / external
        "n": int(len(df)) if df is not None else 0,
        "pos_rate": np.nan,
        "avg_words_child": np.nan,
        "ctx_coverage": np.nan,
        "avg_words_ctx": np.nan,
        "avg_words_composed": np.nan,
    }
    if df is None or len(df) == 0:
        return out

    # Label
    if "label" in df.columns:
        try:
            out["pos_rate"] = float(pd.to_numeric(df["label"], errors="coerce").mean())
        except Exception:
            out["pos_rate"] = np.nan

    # Child text
    if "text" in df.columns:
        out["avg_words_child"] = float(np.mean([_word_count(t) for t in df["text"]]))
    else:
        out["avg_words_child"] = np.nan

    # Context
    if "ctx" in df.columns:
        ctx_series = df["ctx"].fillna("")
        non_empty = ctx_series.apply(lambda x: 1 if str(x).strip() != "" else 0)
        cov = non_empty.mean() if len(non_empty) else np.nan
        out["ctx_coverage"] = float(cov) if cov == cov else np.nan  # keep NaN if NaN
        # avg words in ctx where present
        if cov and cov > 0:
            out["avg_words_ctx"] = float(
                np.mean([_word_count(x) for x in ctx_series[non_empty == 1]])
            )
        else:
            out["avg_words_ctx"] = np.nan

    # Composed
    if "__text_for_model" in df.columns:
        out["avg_words_composed"] = float(
            np.mean([_word_count(t) for t in df["__text_for_model"]])
        )
    else:
        # if we have ctx + text, compose on the fly; else fall back to child
        if "text" in df.columns and "ctx" in df.columns:
            out["avg_words_composed"] = float(
                np.mean([_word_count(_compose(c, t)) for c, t in zip(df["ctx"], df["text"])])
            )
        elif "text" in df.columns:
            out["avg_words_composed"] = float(
                np.mean([_word_count(t) for t in df["text"]])
            )
        else:
            out["avg_words_composed"] = np.nan

    return out


def _fmt_table_markdown(df: pd.DataFrame, out_md: Path) -> None:
    """Write a thesis-ready Markdown table with robust formatting."""
    df = df.copy()

    # normalise placeholders to NaN
    df = df.replace({"N/A": np.nan, "NA": np.nan, "": np.nan, "—": np.nan, "–": np.nan})

    # classify columns for formatting
    label_cols = ["split", "source"]
    int_cols = [c for c in df.columns if c.lower().startswith("n")]
    pct_cols = [c for c in df.columns if any(k in c.lower() for k in ["rate", "share", "pct", "percent", "coverage"])]
    float_cols = [c for c in df.columns if c not in label_cols + int_cols + pct_cols]

    def _fmt_value(x, kind="float"):
        if x is None:
            return "–"
        # try coerce strings like "0.123"
        try:
            if isinstance(x, str):
                x = float(x)
        except Exception:
            return "–"
        # handle NaN
        if isinstance(x, float) and np.isnan(x):
            return "–"
        if kind == "int":
            return f"{int(round(float(x))):,}"
        if kind == "pct":
            # interpret 0..1 as percentage
            return f"{float(x)*100:.1f}%"
        return f"{float(x):.1f}"

    df_fmt = df.copy()
    for c in int_cols:
        if c in df_fmt:
            df_fmt[c] = df_fmt[c].apply(lambda x: _fmt_value(x, "int"))
    for c in pct_cols:
        if c in df_fmt:
            df_fmt[c] = df_fmt[c].apply(lambda x: _fmt_value(x, "pct"))
    for c in float_cols:
        if c in df_fmt:
            df_fmt[c] = df_fmt[c].apply(lambda x: _fmt_value(x, "float"))

    # write Markdown
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(df_fmt.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df_fmt.columns)) + " |\n")
        for _, row in df_fmt.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.tolist()) + " |\n")


# --------------------------- main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Make dataset comparison table for thesis.")
    parser.add_argument("--train", default=None, help="Path to internal train.csv (plain).")
    parser.add_argument("--train_ctx", default=None, help="Path to internal train_ctx.csv (context).")
    parser.add_argument("--val", default=None, help="Path to internal val.csv (plain).")
    parser.add_argument("--val_ctx", default=None, help="Path to internal val_ctx.csv (context).")
    parser.add_argument("--ext_plain", default=None, help="Path to external/test.csv (plain).")
    parser.add_argument("--ext_ctx", default=None, help="Path to external/test_ctx.csv or test_with_ctx.csv (context).")
    parser.add_argument("--out_dir", default="experiments/exp4_context", help="Directory for outputs.")
    args = parser.parse_args()

    # fallbacks if args not provided
    train_path     = _exists_any([args.train, "data/train.csv"])
    train_ctx_path = _exists_any([args.train_ctx, "data/train_ctx.csv"])
    val_path       = _exists_any([args.val, "data/val.csv"])
    val_ctx_path   = _exists_any([args.val_ctx, "data/val_ctx.csv"])
    ext_plain_path = _exists_any([args.ext_plain, "external/test.csv"])
    ext_ctx_path   = _exists_any([args.ext_ctx, "external/test_ctx.csv", "external/test_with_ctx.csv"])

    # read what exists (silently skip missing files)
    def _read_csv(p):
        if p is None:
            return None
        try:
            return pd.read_csv(p)
        except Exception as e:
            print(f"[warn] Could not read {p}: {e}")
            return None

    dfs = []
    mapping = [
        ("Train (plain)",      "internal", _read_csv(train_path)),
        ("Train (context)",    "internal", _read_csv(train_ctx_path)),
        ("Validation (plain)", "internal", _read_csv(val_path)),
        ("Validation (context)","internal", _read_csv(val_ctx_path)),
        ("External (plain)",   "external", _read_csv(ext_plain_path)),
        ("External (context)", "external", _read_csv(ext_ctx_path)),
    ]

    for split_name, source, df in mapping:
        if df is None:
            continue
        # sanity: ensure expected columns exist
        missing = [c for c in ["text", "label"] if c not in df.columns]
        if missing:
            print(f"[warn] {split_name}: missing columns {missing}. Will compute what we can.")
        dfs.append(_summarise_split(df, split_name, source))

    if not dfs:
        raise SystemExit("No dataset CSVs found. Provide paths with --train/--val/--ext_plain/--ext_ctx.")

    out_df = pd.DataFrame(dfs)

    # Optional: add a quick external vs internal size ratio row
    try:
        ext_row = out_df[out_df["split"] == "External (context)"].iloc[0]
        tr_row_candidates = out_df[out_df["split"] == "Train (context)"]
        base_row = tr_row_candidates.iloc[0] if len(tr_row_candidates) else out_df[out_df["split"] == "Train (plain)"].iloc[0]
        ratio = (ext_row["n"] / base_row["n"]) if base_row["n"] else np.nan
        delta_len = ext_row["avg_words_child"] - base_row["avg_words_child"]
        out_df.loc[len(out_df)] = {
            "split": "Δ External−Train (context preferred)",
            "source": "derived",
            "n": np.nan,
            "pos_rate": np.nan,
            "avg_words_child": delta_len,
            "ctx_coverage": (ext_row["ctx_coverage"] - base_row.get("ctx_coverage", np.nan))
                             if not np.isnan(base_row.get("ctx_coverage", np.nan)) else np.nan,
            "avg_words_ctx": ext_row["avg_words_ctx"] - base_row.get("avg_words_ctx", np.nan) if
                             not np.isnan(base_row.get("avg_words_ctx", np.nan)) else np.nan,
            "avg_words_composed": ext_row["avg_words_composed"] - base_row["avg_words_composed"],
        }
        # also store ratio as a separate column for quick reading
        out_df["n_ratio_ext_over_train"] = ""
        out_df.iloc[-1, out_df.columns.get_loc("n_ratio_ext_over_train")] = f"{ratio:.2f}×" if ratio == ratio else "–"
    except Exception:
        pass  # derive row is optional

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "table_dataset_diffs.csv"
    out_md  = out_dir / "table_dataset_diffs.md"

    # write machine-friendly CSV
    out_df.to_csv(out_csv, index=False)
    # write nicely formatted Markdown
    _fmt_table_markdown(out_df, out_md)

    print(f"[ok] Wrote:\n  - {out_csv}\n  - {out_md}")


if __name__ == "__main__":
    main()