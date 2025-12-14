
import argparse, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@torch.inference_mode()
def predict_texts(model, tok, texts, device="cpu", batch_size=32):
    outs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i+batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:,1].cpu().numpy().tolist()
        outs.extend(probs)
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HuggingFace model path or local dir")
    ap.add_argument("--hatecheck", required=True, help="CSV with columns: text,label,functionality")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(args.device).eval()
    df = pd.read_csv(args.hatecheck)
    df["pred"] = predict_texts(model, tok, df["text"].tolist(), device=args.device)
    df["yhat"] = (df["pred"] >= 0.5).astype(int)

    acc = (df["yhat"] == df["label"]).mean()
    print(f"Overall accuracy: {acc:.4f}")
    grp = df.groupby("functionality").apply(lambda g: (g["yhat"]==g["label"]).mean()).reset_index(name="acc")
    print("\n=== Per-functionality accuracy ===")
    print(grp.to_string(index=False))
    df.to_csv("hatecheck_scored.csv", index=False)
    print("\nWrote: hatecheck_scored.csv")

if __name__ == "__main__":
    main()
