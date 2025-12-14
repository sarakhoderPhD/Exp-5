import argparse, pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument("--plain", required=True)
ap.add_argument("--ctx", required=True)
ap.add_argument("--out", required=True)
args=ap.parse_args()

sp = pd.read_csv(args.plain)
sc = pd.read_csv(args.ctx)

def pick_col(df, prefixes):
    cols = [c for c in df.columns]
    for pref in prefixes:
        for c in cols:
            if c.lower().startswith(pref):
                return c
    return None

key = "slice" if "slice" in sp.columns else ("group" if "group" in sp.columns else sp.columns[0])

r_plain = pick_col(sp, ["recall@best_f1","recall@0.5","recall"])
r_ctx   = pick_col(sc, ["recall@best_f1","recall@0.5","recall"])
if r_plain is None or r_ctx is None:
    raise SystemExit("Could not find a recall column in one of the slices files.")

sp_ = sp[[key, r_plain]].rename(columns={r_plain: "recall_plain"})
sc_ = sc[[key, r_ctx  ]].rename(columns={r_ctx:   "recall_ctx"})

m = sp_.merge(sc_, on=key, how="inner")
m["tpr_gap"] = m["recall_ctx"] - m["recall_plain"]
m = m.sort_values("tpr_gap")
m.to_csv(args.out, index=False)

print(f"Joined on '{key}' using plain='{r_plain}', ctx='{r_ctx}'")
print("Wrote", args.out)
