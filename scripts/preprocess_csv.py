#!/usr/bin/env python
import argparse, pandas as pd
from cyberhate_lab.data.preprocess import normalize_text, segment_hashtags, replace_emojis_with_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-csv', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--text-col', default='text')
    ap.add_argument('--segment-hashtags', action='store_true')
    ap.add_argument('--replace-emojis', action='store_true')
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    texts = []
    for t in df[args.text_col].astype(str).tolist():
        x = normalize_text(t, lower=True, strip_urls=True, strip_users=True)
        if args.segment_hashtags:
            x = segment_hashtags(x)
        if args.replace_emojis:
            x = replace_emojis_with_names(x)
        texts.append(x)
    df[args.text_col] = texts
    df.to_csv(args.out_csv, index=False)
    print('Wrote', args.out_csv)

if __name__ == '__main__':
    main()
