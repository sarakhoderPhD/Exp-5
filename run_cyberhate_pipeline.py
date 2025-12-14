#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_cyberhate_pipeline.py

End-to-end driver for your PhD cyber-hate detection pipeline.

What it does (in order):
1) (Optional) Train a binary classifier (text) with scripts/train_text.py
2) Calibrate probabilities (temperature scaling) and keep ECE-friendly outputs
3) Score an evaluation CSV to create fairness input (preds_with_identities.csv)
4) Compute fairness metrics (Subgroup/BPSN/BNSP AUC)
5) Generate diagnostic figures (reliability diagrams + per-identity bias chart)
6) Build a thesis-ready report (Markdown + LaTeX + BibTeX)
7) (Optional) HateCheck functional evaluation
8) (Optional) Robustness sweep (OCR/ASR noise) — ΔF1 (via the all-in-one driver)
9) (Optional) Span rationales + faithfulness (comprehensiveness/sufficiency) (via the all-in-one driver)

USAGE (minimal):
    python run_cyberhate_pipeline.py \
      --run-dir runs/roberta_bin \
      --train-csv data/train.csv --val-csv data/val.csv \
      --text-col text --label-col label \
      --eval-texts-csv eval/eval_texts.csv \
      --id-cols identity_women,identity_men

If you have already trained a model (e.g., in Experiment_3.ipynb), add --skip-train
and ensure RUN_DIR contains:
    - val_logits.npy
    - val_labels.npy
(Optionally val_probs.npy)
"""

from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

def shell(cmd, check=False):
    print('>>', ' '.join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd)
    if check and r.returncode != 0:
        raise SystemExit(f"Command failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return r.returncode

def main():
    ap = argparse.ArgumentParser(description='Run SK PhD cyberhate pipeline end-to-end.')
    # Core outputs & data
    ap.add_argument('--run-dir', default='runs/roberta_bin', help='Model + audit output directory')
    ap.add_argument('--skip-train', action='store_true', help='Skip training and reuse an existing run')

    ap.add_argument('--train-csv', help='CSV with text,label for training')
    ap.add_argument('--val-csv',   help='CSV with text,label for validation')
    ap.add_argument('--text-col',  default='text')
    ap.add_argument('--label-col', default='label')

    # Training hyperparameters
    ap.add_argument('--model', default='roberta-base')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--class-weights', action='store_true')

    # Fairness inputs
    ap.add_argument('--eval-texts-csv', help='CSV with id,text,label,(identity_* columns) to score')
    ap.add_argument('--id-cols', default='', help='Comma-separated identity columns, e.g. "identity_women,identity_men"')

    # HateCheck
    ap.add_argument('--hatecheck-csv', help='CSV with text,label,functionality (optional)')

    # Spans / faithfulness (optional)
    ap.add_argument('--spans-train-csv', help='CSV with text,spans ("start:end;start:end")')
    ap.add_argument('--spans-val-csv',   help='CSV with text,spans ("start:end;start:end")')
    ap.add_argument('--span-model-out',  default='runs/span_model')

    # Charts
    ap.add_argument('--use-calibrated', action='store_true', help='Plot post-calibration reliability if available')

    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) TRAIN (optional) ----
    if not args.skip_train:
        if not args.train_csv or not args.val_csv:
            raise SystemExit('Provide --train-csv and --val-csv, or use --skip-train to reuse an existing run.')
        cmd = [
            sys.executable, 'scripts/train_text.py',
            '--train', args.train_csv, '--val', args.val_csv,
            '--text-col', args.text_col, '--label-col', args.label_col,
            '--model', args.model, '--epochs', str(args.epochs),
            '--lr', str(args.lr), '--batch-size', str(args.batch_size),
            '--max-length', str(args.max_length), '--out', str(run_dir)
        ]
        if args.class_weights:
            cmd.append('--class-weights')
        shell(cmd, check=True)

    # ---- 2) CALIBRATION ----
    logits = run_dir/'val_logits.npy'
    labels = run_dir/'val_labels.npy'
    if not logits.exists() or not labels.exists():
        raise SystemExit(f'Missing validation outputs in {run_dir}. Expect val_logits.npy and val_labels.npy.')
    shell([sys.executable, 'scripts/calibrate.py',
           '--logits', str(logits), '--labels', str(labels),
           '--out', str(run_dir/'val_probs_cal.npy')])

    # ---- 3) SCORE EVAL → preds_with_identities.csv ----
    if args.eval_texts_csv:
        scorer = (
            "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; "
            "import torch, pandas as pd, numpy as np, sys; "
            "run_dir=sys.argv[1]; in_csv=sys.argv[2]; text_col=sys.argv[3]; max_len=int(sys.argv[4]); out_csv=sys.argv[5]; "
            "tok=__import__('transformers').AutoTokenizer.from_pretrained(run_dir); "
            "clf=__import__('transformers').AutoModelForSequenceClassification.from_pretrained(run_dir).eval(); "
            "df=pd.read_csv(in_csv); texts=df[text_col].astype(str).tolist(); preds=[]; "
            "with torch.inference_mode(): "
            "  \n"
            "  import itertools\n"
            "  for i in range(0,len(texts),32): "
            "    enc=tok(texts[i:i+32],padding=True,truncation=True,max_length=max_len,return_tensors='pt'); "
            "    p=torch.softmax(clf(**enc).logits,dim=-1)[:,1].cpu().numpy(); preds+=list(p); "
            "df['pred']=np.array(preds); df.to_csv(out_csv,index=False)"
        )
        out_preds = Path('eval/preds_with_identities.csv')
        out_preds.parent.mkdir(parents=True, exist_ok=True)
        shell([sys.executable, '-c', scorer,
               str(run_dir), args.eval_texts_csv, args.text_col, str(args.max_length), str(out_preds)])

    # ---- 4) FAIRNESS ----
    fairness_input = Path('eval/preds_with_identities.csv')
    if fairness_input.exists() and args.id_cols.strip():
        shell([sys.executable, 'scripts/eval.py',
               '--preds', str(fairness_input), '--id-cols', args.id_cols])

    # ---- 5) CHARTS ----
    charts_cmd = [sys.executable, 'scripts/make_charts.py', '--run-dir', str(run_dir)]
    if args.use_calibrated:
        charts_cmd.append('--use-calibrated')
    shell(charts_cmd)

    # ---- 6) REPORT ----
    shell([sys.executable, 'scripts/make_thesis_report.py', '--run-dir', str(run_dir)])

    # ---- 7) HATECHECK (optional) ----
    if args.hatecheck_csv and Path(args.hatecheck_csv).exists():
        shell([sys.executable, 'scripts/run_hatecheck.py',
               '--model-path', str(run_dir), '--hatecheck', args.hatecheck_csv])

    # ---- 8 & 9) ROBUSTNESS + FAITHFULNESS + CONSOLIDATED SUMMARY (via driver) ----
    driver_cmd = [sys.executable, 'scripts/thesis_audit.py', '--run-dir', str(run_dir), '--skip-train']
    if fairness_input.exists() and args.id_cols.strip():
        driver_cmd += ['--preds-csv', str(fairness_input), '--identity-cols', args.id_cols]
    if args.eval_texts_csv and Path(args.eval_texts_csv).exists():
        driver_cmd += ['--eval-texts-csv', str(args.eval_texts_csv)]
    if args.hatecheck_csv and Path(args.hatecheck_csv).exists():
        driver_cmd += ['--hatecheck-csv', str(args.hatecheck_csv)]
    if args.spans_train_csv and args.spans_val_csv and Path(args.spans_train_csv).exists() and Path(args.spans_val_csv).exists():
        driver_cmd += ['--spans-train-csv', str(args.spans_train_csv),
                       '--spans-val-csv', str(args.spans_val_csv),
                       '--span-model-out', args.span_model_out]
    shell(driver_cmd)

    print('\nAll done. See:')
    print(f' - Aggregate & audit: {run_dir}/audit/summary.json')
    print(f' - Report (MD/TeX/Bib + figures): {run_dir}/audit/report/')

if __name__ == '__main__':
    main()