#!/usr/bin/env python
import argparse, pandas as pd, json
from cyberhate_lab.data.counterfactuals import generate_counterfactuals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-csv', required=True)
    ap.add_argument('--text-col', default='text')
    ap.add_argument('--out-jsonl', required=True)
    ap.add_argument('--lm-name', default='roberta-base')
    ap.add_argument('--topk', type=int, default=50)
    ap.add_argument('--pll-delta', type=float, default=5.0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    cands = generate_counterfactuals(df[args.text_col].astype(str).tolist(), lm_name=args.lm_name, topk=args.topk, pll_delta_thresh=args.pll_delta)
    with open(args.out_jsonl, 'w') as f:
        for i,opts in enumerate(cands):
            rec = {'idx': int(i), 'text': df.iloc[i][args.text_col], 'counterfactuals': opts}
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')
    print('Wrote', args.out_jsonl)

if __name__ == '__main__':
    main()
