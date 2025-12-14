
#!/usr/bin/env python
import argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def mask_tokens(text, tok, token_mask, max_length=256):
    enc = tok(text, return_offsets_mapping=True, truncation=True, padding='max_length', max_length=max_length)
    ids = enc['input_ids']
    attn = enc['attention_mask']
    for i in range(min(len(ids), len(token_mask))):
        if token_mask[i] == 1 and attn[i] == 1 and tok.mask_token_id is not None:
            ids[i] = tok.mask_token_id
    return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([attn])}

def keep_only_tokens(text, tok, token_mask, max_length=256):
    enc = tok(text, return_offsets_mapping=True, truncation=True, padding='max_length', max_length=max_length)
    ids = enc['input_ids']
    attn = enc['attention_mask']
    for i in range(len(ids)):
        if attn[i]==1 and token_mask[i]==0:
            ids[i] = tok.pad_token_id
            attn[i] = 0
    return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([attn])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--val-csv', required=True)  # needs columns: text,label
    ap.add_argument('--rationales-npy', required=True)
    ap.add_argument('--max-length', type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    clf = AutoModelForSequenceClassification.from_pretrained(args.model_path).eval()

    df = pd.read_csv(args.val_csv)
    R = np.load(args.rationales_npy)  # shape [N, seq_len]

    @torch.inference_mode()
    def prob_positive(enc):
        logits = clf(**{k:v for k,v in enc.items()}).logits
        return torch.softmax(logits, dim=-1)[:,1].cpu().numpy()

    comp_deltas, suff_deltas = [], []
    for i, row in df.iterrows():
        text = str(row['text'])
        mask = R[i]
        base = tok(text, truncation=True, padding='max_length', max_length=args.max_length, return_tensors='pt')
        p_base = prob_positive(base)[0]
        masked = mask_tokens(text, tok, mask, max_length=args.max_length)
        p_masked = prob_positive(masked)[0]
        kept = keep_only_tokens(text, tok, mask, max_length=args.max_length)
        p_kept = prob_positive(kept)[0]
        comp_deltas.append(p_base - p_masked)
        suff_deltas.append(p_base - p_kept)

    print(f"Comprehensiveness (mean prob drop): {np.mean(comp_deltas):.4f}")
    print(f"Sufficiency (mean prob drop): {np.mean(suff_deltas):.4f}")
    np.save('faithfulness_comp.npy', np.array(comp_deltas))
    np.save('faithfulness_suff.npy', np.array(suff_deltas))
    print('Saved: faithfulness_comp.npy , faithfulness_suff.npy')

if __name__ == '__main__':
    main()
