import argparse, pandas as pd, subprocess

ap=argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--probs_a", required=True)
ap.add_argument("--name_a", default="baseline")
ap.add_argument("--probs_b", required=True)
ap.add_argument("--name_b", default="context")
ap.add_argument("--out", required=True)
args=ap.parse_args()

def run_metrics(labels, probs, outcsv):
    subprocess.check_call(
        ["python","scripts/metrics_simple.py","--labels",labels,"--probs",probs,"--out",outcsv]
    )
    return pd.read_csv(outcsv)

tmpA = "external/out/_mA.csv"
tmpB = "external/out/_mB.csv"
ma_raw = run_metrics(args.labels, args.probs_a, tmpA)
mb_raw = run_metrics(args.labels, args.probs_b, tmpB)

# Prefix columns so we can concat side-by-side without clashes
ma = ma_raw.add_prefix(args.name_a + "_")
mb = mb_raw.add_prefix(args.name_b + "_")

# Pull a single shared 'n' and 'pos_rate' from side A (any side is fine)
n_col_a   = [c for c in ma.columns if c.endswith("_n")][0]
pos_col_a = [c for c in ma.columns if c.endswith("_pos_rate")][0]
base = pd.DataFrame({
    "n": ma[n_col_a],
    "pos_rate": ma[pos_col_a]
})

# Drop the duplicated n/pos_rate from each side separately
drop_a = [n_col_a, pos_col_a]
drop_b = [c for c in mb.columns if c.endswith("_n") or c.endswith("_pos_rate")]

out = pd.concat([base, ma.drop(columns=drop_a), mb.drop(columns=drop_b)], axis=1)
out.to_csv(args.out, index=False)
print(f"Wrote {args.out}\n{out.to_string(index=False)}")
