import argparse, os, pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SWAPS = {
  "muslim":"christian", "christian":"muslim",
  "jew":"christian", "christian":"jew",
  "black":"white", "white":"black",
  "gay":"straight", "straight":"gay",
  "woman":"man", "women":"men", "man":"woman", "men":"women"
}

def swap_once(text):
    t = " " + text + " "
    for a,b in SWAPS.items():
        t = t.replace(f" {a} ", f" {b} ")
    return t.strip()

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--in-csv", required=True)
ap.add_argument("--text-col", default="text")
ap.add_argument("--max-length", type=int, default=256)
ap.add_argument("--out", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.run_dir)
clf = AutoModelForSequenceClassification.from_pretrained(args.run_dir).eval()

df = pd.read_csv(args.in_csv)
texts = df[args.text_col].astype(str)
cf_texts = texts.apply(swap_once)

def batched_probs(txts):
    probs=[]; 
    with torch.inference_mode():
        for i in range(0, len(txts), 32):
            enc = tok(list(txts[i:i+32]), padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt")
            p = clf(**enc).logits.softmax(dim=-1)[:,1].cpu().numpy()
            probs.extend(p)
    return np.array(probs)

p_orig = batched_probs(texts)
p_cf   = batched_probs(cf_texts)
delta  = p_cf - p_orig

out = pd.DataFrame({
    "text": texts,
    "text_cf": cf_texts,
    "p_orig": p_orig,
    "p_cf": p_cf,
    "delta": delta
})
os.makedirs(os.path.dirname(args.out), exist_ok=True)
out.to_csv(args.out, index=False)

summary = out["delta"].describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).to_dict()
print("Δ summary:", summary)
