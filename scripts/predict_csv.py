import sys, torch, pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
run_dir, in_csv, text_col, max_len, out_csv = sys.argv[1:]; max_len=int(max_len)
tok = AutoTokenizer.from_pretrained(run_dir)
clf = AutoModelForSequenceClassification.from_pretrained(run_dir).eval()
df = pd.read_csv(in_csv); texts = df[text_col].astype(str).tolist(); preds=[]
with torch.inference_mode():
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        p = clf(**enc).logits.softmax(dim=-1)[:,1].cpu().numpy(); preds.extend(p)
df["pred"] = np.array(preds); df.to_csv(out_csv, index=False)
