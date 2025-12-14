
import argparse, os, numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, set_seed
from typing import List, Tuple

def parse_spans(span_str: str) -> List[Tuple[int,int]]:
    if span_str is None or (isinstance(span_str, float) and np.isnan(span_str)):
        return []
    s = str(span_str).strip()
    if not s:
        return []
    out = []
    for part in s.split(';'):
        part = part.strip()
        if not part: 
            continue
        a,b = part.split(':')
        out.append((int(a), int(b)))
    return out

def char_to_token_labels(text: str, spans: List[Tuple[int,int]], tok, max_length: int=256):
    enc = tok(text, return_offsets_mapping=True, truncation=True, padding='max_length', max_length=max_length)
    offsets = enc['offset_mapping']
    labels = [0]*len(offsets)
    for i,(a,b) in enumerate(offsets):
        # offsets for special tokens may be (0,0); keep 0 but will be ignored via attention_mask
        if a==b:
            labels[i] = 0
            continue
        lab = 0
        for (s,e) in spans:
            if not (b<=s or a>=e):
                lab = 1; break
        labels[i] = lab
    # ignore padding tokens in loss
    for i,m in enumerate(enc['attention_mask']):
        if m == 0:
            labels[i] = -100
    enc['labels'] = labels
    enc.pop('offset_mapping', None)
    return enc

def build_dataset(csv_path: str, tok, text_col: str='text', span_col: str='spans', max_length: int=256):
    df = pd.read_csv(csv_path)
    records = []
    for _,row in df.iterrows():
        text = str(row[text_col])
        spans = parse_spans(row.get(span_col, ''))
        rec = char_to_token_labels(text, spans, tok, max_length=max_length)
        records.append(rec)
    # unify keys
    keys = records[0].keys()
    data = {k:[r[k] for r in records] for k in keys}
    ds = Dataset.from_dict(data)
    ds.set_format(type='torch')
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--model', default='roberta-base')
    ap.add_argument('--text-col', default='text')
    ap.add_argument('--span-col', default='spans')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='runs/span_model')
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_ds = build_dataset(args.train, tok, text_col=args.text_col, span_col=args.span_col, max_length=args.max_length)
    val_ds = build_dataset(args.val, tok, text_col=args.text_col, span_col=args.span_col, max_length=args.max_length)

    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=2)

    collator = DataCollatorForTokenClassification(tokenizer=tok)
    args_tr = TrainingArguments(
        output_dir=args.out, learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='f1', report_to='none', fp16=torch.cuda.is_available()
    )

    from sklearn.metrics import f1_score
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = (logits[...,1] >= logits[...,0]).astype(int)
        mask = labels != -100
        y_true = labels[mask].flatten()
        y_pred = preds[mask].flatten()
        return {'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)}

    trainer = Trainer(model=model, args=args_tr, data_collator=collator, tokenizer=tok,
                      train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(args.out)

    logits = trainer.predict(val_ds).predictions
    preds = (logits[...,1] >= logits[...,0]).astype(int)
    np.save(f"{args.out}/val_token_rationales.npy", preds)
    print('Saved token-level rationale predictions to', f"{args.out}/val_token_rationales.npy")

if __name__ == '__main__':
    main()
