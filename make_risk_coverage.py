# make_risk_coverage.py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

SEARCH_ROOTS = [Path("external/out"), Path("experiments/exp4_context"), Path("experiments")]

# Acceptable column names
PROB_COLS = ["pred_cal", "p_cal", "prob", "proba", "probability", "score", "y_pred", "p", "prob_cal"]
LABEL_COLS = ["label", "y_true", "y", "target", "labels", "truth"]

# Filenames we actively prefer
PREFERRED_PLAIN = ["preds_plain.csv"]
PREFERRED_CTX   = ["preds_ctx.csv", "preds_context.csv"]

# Things to skip during auto-discovery
SKIP_PREFIXES = ("slices_", "metrics_", "rc_", "risk_coverage", "reliability_", "compare_", "_mA", "_mB")
SKIP_CONTAINS = ("slices", "metrics", "reliability")

def choose_file(preferred_names, fallbacks_contains, model_hint):
    """Pick a file by preferring exact names, then 'pred' CSVs mentioning the hint, then fall back to p_*.npy."""
    # 1) exact preferred filenames
    for root in SEARCH_ROOTS:
        for name in preferred_names:
            p = root / name
            if p.exists():
                return p

    # 2) any CSV/Parquet/Feather with 'pred' and the model hint in name, skipping slice/metric/rc files
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if any(name.startswith(pref) for pref in SKIP_PREFIXES): 
                continue
            if any(s in name for s in SKIP_CONTAINS):
                continue
            if "pred" in name and model_hint in name and p.suffix in {".csv", ".parquet", ".feather"}:
                return p

    # 3) fallback to p_plain.npy / p_ctx.npy
    for root in SEARCH_ROOTS:
        candidate = root / f"p_{model_hint}.npy"
        if candidate.exists():
            return candidate

    return None

def load_labels_fallback():
    """Find labels if the chosen file lacks them."""
    # Prefer numpy labels, else CSV
    for root in SEARCH_ROOTS:
        p = root / "y_ext.npy"
        if p.exists():
            arr = np.load(p)
            return arr.ravel()
    for root in SEARCH_ROOTS:
        p = root / "y_ext.csv"
        if p.exists():
            df = pd.read_csv(p)
            # first numeric-ish column
            for c in df.columns:
                try:
                    vals = pd.to_numeric(df[c])
                    return vals.to_numpy().ravel()
                except Exception:
                    continue
    raise FileNotFoundError("Could not find labels fallback (y_ext.npy or y_ext.csv).")

def load_preds(path):
    """Return (prob, label or None)."""
    if path.suffix == ".npy":
        # Could be raw probabilities only (p_plain.npy / p_ctx.npy) or dict-like
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
            # dict-like object
            d = arr.item()
            prob = None
            for c in PROB_COLS:
                if c in d:
                    prob = np.asarray(d[c]).ravel()
                    break
            lab = None
            for c in LABEL_COLS:
                if c in d:
                    lab = np.asarray(d[c]).ravel()
                    break
            if prob is None:
                raise ValueError(f"{path} had no recognised prob key. Keys: {list(d.keys())}")
            return prob, lab
        else:
            # assume this is just a vector of probabilities
            return np.asarray(arr).ravel(), None

    # tabular
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".feather":
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)

    prob = None
    for c in PROB_COLS:
        if c in df.columns:
            prob = df[c].to_numpy().ravel()
            break
    if prob is None:
        raise ValueError(f"{path} missing probability column. Have: {list(df.columns)}")

    lab = None
    for c in LABEL_COLS:
        if c in df.columns:
            lab = df[c].to_numpy().ravel()
            break

    return prob, lab

def risk_coverage(prob, labels, thr=0.5, grid=201):
    ws = np.linspace(0.0, 0.5, grid)
    yhat = (prob >= thr).astype(int)
    rows = []
    for w in ws:
        act = np.abs(prob - thr) >= w
        cov = act.mean()
        if cov == 0:
            sel_err = np.nan
        else:
            sel_err = (yhat[act] != labels[act]).mean()
        rows.append((cov, sel_err))
    return pd.DataFrame(rows, columns=["coverage", "selective_error"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", help="Path to plain predictions (CSV/Parquet/Feather with prob+label, or .npy).")
    ap.add_argument("--context", help="Path to context predictions (CSV/Parquet/Feather with prob+label, or .npy).")
    ap.add_argument("--labels", help="Optional labels file (npy/csv) if the chosen preds lack labels.")
    args = ap.parse_args()

    out_dir = Path("experiments/exp4_context")
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = {}

    # Select files (explicit > autodiscover)
    if args.plain:
        chosen["plain"] = Path(args.plain)
    else:
        chosen["plain"] = choose_file(PREFERRED_PLAIN, ("pred",), "plain")

    if args.context:
        chosen["context"] = Path(args.context)
    else:
        chosen["context"] = choose_file(PREFERRED_CTX, ("pred",), "ctx")

    if not chosen["plain"] and not chosen["context"]:
        raise SystemExit("No predictions found. Provide --plain/--context or place preds_plain.csv / preds_ctx.csv in external/out.")

    rows = []
    for model, path in chosen.items():
        if path is None:
            print(f"[info] No file for {model}; skipping.")
            continue
        try:
            prob, lab = load_preds(path)
            if lab is None:
                # need labels from fallback or user-provided
                if args.labels:
                    lab_path = Path(args.labels)
                    if lab_path.suffix == ".npy":
                        lab = np.load(lab_path).ravel()
                    else:
                        df = pd.read_csv(lab_path)
                        # pick first numeric column
                        for c in df.columns:
                            try:
                                vals = pd.to_numeric(df[c])
                                lab = vals.to_numpy().ravel()
                                break
                            except Exception:
                                continue
                        if lab is None:
                            raise ValueError(f"{lab_path} had no numeric column for labels.")
                else:
                    lab = load_labels_fallback()
            df_rc = risk_coverage(prob, lab, thr=0.5, grid=201)
            df_rc.insert(0, "model", model)
            rows.append(df_rc)
            print(f"[ok] Using {model} predictions: {path}  (n={len(prob)})")
        except Exception as e:
            print(f"[warn] {model}: {e}")

    if not rows:
        raise SystemExit("No risk–coverage data produced. Ensure your files have a probability column and labels are available.")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_dir / "risk_coverage_external.csv", index=False)
    print(f"[ok] Wrote {out_dir / 'risk_coverage_external.csv'} with {len(out)} rows.")

if __name__ == "__main__":
    main()