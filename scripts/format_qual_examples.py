import argparse, pandas as pd, numpy as np, re, textwrap, os

def redact(text: str) -> str:
    if not isinstance(text, str): return ""
    # basic slur/obscenity redaction patterns; expand as needed
    patterns = [
        r'\b(nigg\w+)\b', r'\b(fagg\w+)\b', r'\b(spic\w*)\b', r'\b(kik\w*)\b',
        r'\b(retard\w*)\b', r'\b(slut\w*)\b', r'\b(whore\w*)\b', r'\b(cunt\w*)\b'
    ]
    red = text
    for pat in patterns:
        red = re.sub(pat, '▇▇▇', red, flags=re.IGNORECASE)
    return red

def clip(s: str, max_chars=480):
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars-1].rstrip() + "…"

def fmt_prob(x): 
    try: return f"{float(x):.3f}"
    except: return "NA"

def label_name(y):
    return "Hate" if int(y)==1 else "Non-hate"

def make_block(tag, row):
    ctx = redact(str(row.get("ctx","")).strip())
    txt = redact(str(row.get("text","")).strip())
    y   = row.get("label", 0)
    p0, p1 = row.get("p_plain", np.nan), row.get("p_ctx", np.nan)
    yhat0, yhat1 = row.get("pred_plain", np.nan), row.get("pred_ctx", np.nan)

    merged = []
    if ctx: merged.append(f"CONTEXT: {clip(ctx)}")
    if txt: merged.append(f"CHILD: {clip(txt)}")
    body = "\n—\n".join(merged) if merged else "(no text)"

    note_parts = []
    if pd.notna(p0) and pd.notna(p1):
        note_parts.append(f"p_plain={fmt_prob(p0)}, p_ctx={fmt_prob(p1)}")
    if pd.notna(yhat0) and pd.notna(yhat1):
        note_parts.append(f"pred_plain={int(yhat0)}, pred_ctx={int(yhat1)}")
    note = "; ".join(note_parts)

    return (
        f"**{tag}**  \n"
        f"*Gold:* {label_name(y)}  \n"
        f"*Scores:* {note}  \n"
        f"{body}\n"
    )

def pick_top(df, section, key, n):
    sub = df[df["section"] == section].copy()
    if sub.empty: return []
    sub = sub.sort_values(key, ascending=False).head(n)
    return sub.to_dict("records")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="external/out/error_examples.csv")
    ap.add_argument("--out", default="experiments/exp4_context/qual_examples.md")
    ap.add_argument("--max_per_bucket", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.csv)

    # Ensure required columns exist
    needed = {"section","label","p_plain","p_ctx","pred_plain","pred_ctx","ctx","text"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {args.csv}: {sorted(missing)}")

    # Features for ranking
    df["abs_delta"] = (df["p_ctx"] - df["p_plain"]).abs()

    blocks = []

    # 1) FP fixed by context: plain=1, ctx=0, y=0 (already prefiltered in your generator)
    for r in pick_top(df, "fix_fp", key="p_plain", n=args.max_per_bucket):
        blocks.append(make_block("FP fixed by context", r))

    # 2) FN recovered by context: plain=0, ctx=1, y=1 (prefiltered)
    for r in pick_top(df, "fix_fn", key="p_ctx", n=args.max_per_bucket):
        blocks.append(make_block("FN recovered by context", r))

    # 3) High-contrast flips (any direction)
    flips = df[df["section"]=="flips"].sort_values("abs_delta", ascending=False).head(args.max_per_bucket)
    for _, r in flips.iterrows():
        tag = "Flip (directional)"
        # Give a clearer tag if we can infer direction
        if r.get("pred_plain")==1 and r.get("pred_ctx")==0:
            tag = "Flip: plain→Hate, ctx→Non-hate"
        elif r.get("pred_plain")==0 and r.get("pred_ctx")==1:
            tag = "Flip: plain→Non-hate, ctx→Hate"
        blocks.append(make_block(tag, r))

    header = (
        "### 4.12 Qualitative Examples: Where Context Changes Decisions\n\n"
        "The following cases illustrate how conversation context affects predictions. "
        "Examples are grouped by (i) false positives corrected by context, (ii) false negatives recovered by context, "
        "and (iii) high-contrast flips. Scores are calibrated probabilities.\n\n"
    )

    text = header + "\n".join(blocks) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote {args.out} with {len(blocks)} exemplars.")

if __name__ == "__main__":
    main()