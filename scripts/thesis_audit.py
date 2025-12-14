#!/usr/bin/env python

import argparse, os, sys, json, subprocess, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from cyberhate_lab.evaluation.calibration import expected_calibration_error
from cyberhate_lab.data.fairness_metrics import compute_bias_table, counterfactual_token_fairness
from pathlib import Path
import numpy as np

def shell(cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[WARN] Command failed: {' '.join(cmd)}", file=sys.stderr)
    return r.returncode

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def load_val_outputs(run_dir):
    # Try probs first; if missing, derive from logits
    p_probs  = Path(run_dir)/"val_probs.npy"
    p_logits = Path(run_dir)/"val_logits.npy"
    p_labels = Path(run_dir)/"val_labels.npy"
    p_cal    = Path(run_dir)/"val_probs_cal.npy"   # <— NEW

    if not p_labels.exists():
        raise SystemExit(f"Missing {p_labels} (train with scripts/train_text.py first).")

    y = np.load(p_labels)

    # Prefer calibrated probabilities if present  <— NEW
    if p_cal.exists():
        p = np.load(p_cal)
        if p.ndim == 2:
            p = p[:, 1]
        return y, p

    if p_probs.exists():
        p = np.load(p_probs)
        if p.ndim == 2:
            p = p[:,1]
    elif p_logits.exists():
        logits = np.load(p_logits)
        if logits.ndim == 1:
            # binary logit; convert to prob via sigmoid
            p = 1/(1+np.exp(-logits))
        else:
            # softmax over last dim
            exps = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = (exps[:,1] / exps.sum(axis=1))
    else:
        raise SystemExit("Need val_probs_cal.npy or val_probs.npy or val_logits.npy in run_dir.")

    return y, p

def basic_metrics(y, p, name):
    yhat = (p >= 0.5).astype(int)
    f1 = f1_score(y, yhat, average="binary", zero_division=0)
    auc = roc_auc_score(y, p) if len(set(y.tolist()))>1 else float('nan')
    ece = expected_calibration_error(p, y, n_bins=15)
    return {"name": name, "f1": f1, "roc_auc": auc, "ece_15": ece}

def predict_texts(model_path, texts, device="cpu", batch_size=32, max_length=256):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    clf = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    outs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = clf(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:,1].cpu().numpy().tolist()
            outs.extend(probs)
    return np.array(outs)

def main():
    ap = argparse.ArgumentParser(description="End-to-end thesis audit: Train → Calibrate → Fairness → HateCheck → Robustness → Faithfulness → Risk–Coverage")
    # Training / run_dir
    ap.add_argument("--run-dir", default="runs/roberta_bin")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--train-csv"); ap.add_argument("--val-csv")
    ap.add_argument("--text-col", default="text"); ap.add_argument("--label-col", default="label")
    ap.add_argument("--multilabel-cols", default="")
    ap.add_argument("--sarcasm-col", default="")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--class-weights", action="store_true")

    # Fairness
    ap.add_argument("--preds-csv", help="CSV with id, pred, label, identity_* columns")
    ap.add_argument("--identity-cols", default="")
    ap.add_argument("--ctf-pairs-csv", help="Optional CSV with pair_id,text,label for CTF scoring")

    # HateCheck
    ap.add_argument("--hatecheck-csv", help="CSV with text,label,functionality")

    # Robustness
    ap.add_argument("--eval-texts-csv", help="CSV with id,text,label for robustness sweep")
    ap.add_argument("--p-del", type=float, default=0.03)
    ap.add_argument("--p-sub", type=float, default=0.03)
    ap.add_argument("--p-ins", type=float, default=0.03)

    # Faithfulness (rationales)
    ap.add_argument("--spans-train-csv")
    ap.add_argument("--spans-val-csv")
    ap.add_argument("--span-model-out", default="runs/span_model")

    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    audit_dir = Path(run_dir)/"audit"
    ensure_dir(audit_dir)

    # 0) TRAIN (optional)
    if not args.skip_train:
        assert args.train_csv and args.val_csv, "Provide --train-csv and --val-csv or use --skip-train"
        cmd = [
            "python","scripts/train_text.py",
            "--train", args.train_csv, "--val", args.val_csv,
            "--text-col", args.text_col, "--label-col", args.label_col,
            "--model", args.model, "--epochs", str(args.epochs),
            "--lr", str(args.lr), "--batch-size", str(args.batch_size),
            "--max-length", str(args.max_length),
            "--out", str(run_dir),
        ]
        if args.multilabel_cols:
            cmd.extend(["--multilabel-cols", args.multilabel_cols])
        if args.sarcasm_col:
            cmd.extend(["--aux-sarcasm-col", args.sarcasm_col])
        if args.class_weights:
            cmd.append("--class-weights")
        shell(cmd)

    # 1) Aggregate metrics (pre-calibration)
    y, p = load_val_outputs(run_dir)
    agg_pre = basic_metrics(y, p, "pre_calibration")

    # 2) Calibration
    calib_path = run_dir/"val_probs_cal.npy"
    shell(["python","scripts/calibrate.py","--logits",str(run_dir/'val_logits.npy'),"--labels",str(run_dir/'val_labels.npy'),"--out",str(calib_path)])
    p_cal = np.load(calib_path) if calib_path.exists() else p
    agg_post = basic_metrics(y, p_cal, "post_calibration")

    # 3) Fairness (if provided)
    fairness_csv = None; fairness_tbl = None
    if args.preds_csv and args.identity_cols:
        df = pd.read_csv(args.preds_csv)
        id_cols = [c.strip() for c in args.identity_cols.split(",") if c.strip()]
        from cyberhate_lab.data.fairness_metrics import compute_bias_table
        fairness_tbl = compute_bias_table(df, id_cols, label_col="label", prob_col="pred")
        fairness_csv = audit_dir/"fairness.csv"
        fairness_tbl.to_csv(fairness_csv, index=False)
        print("Wrote", fairness_csv)

    # Optional: CTF scoring from raw text pairs
    ctf_json = None
    if args.ctf_pairs_csv:
        pairs = pd.read_csv(args.ctf_pairs_csv)  # expects: pair_id, text, label
        probs = predict_texts(str(run_dir), pairs["text"].astype(str).tolist())
        pairs["pred"] = probs
        from cyberhate_lab.data.fairness_metrics import counterfactual_token_fairness
        ctf_gap = counterfactual_token_fairness(pairs, group_col="pair_id", label_col="label", prob_col="pred")
        ctf_json = {"ctf_gap": float(ctf_gap)}
        with open(audit_dir/"ctf.json","w") as f: json.dump(ctf_json, f, indent=2)
        pairs.to_csv(audit_dir/"ctf_scored_pairs.csv", index=False)
        print("CTF gap:", ctf_json["ctf_gap"])

    # 4) HateCheck
    hatecheck_csv_out = None
    if args.hatecheck_csv:
        shell(["python","scripts/run_hatecheck.py","--model-path",str(run_dir),"--hatecheck",args.hatecheck_csv])
        if Path("hatecheck_scored.csv").exists():
            hatecheck_csv_out = audit_dir/"hatecheck_scored.csv"
            os.replace("hatecheck_scored.csv", hatecheck_csv_out)
            print("Wrote", hatecheck_csv_out)

    # 5) Robustness (OCR/ASR noise)
    robust_json = None
    if args.eval_texts_csv:
        from cyberhate_lab.ocr_asr.noise_simulation import batch_corrupt
        eval_df = pd.read_csv(args.eval_texts_csv)  # id,text,label
        eval_df["pred"] = predict_texts(str(run_dir), eval_df["text"].astype(str).tolist())
        # Noisy
        text_noisy = batch_corrupt(eval_df["text"].astype(str).tolist(), p_del=args.p_del, p_sub=args.p_sub, p_ins=args.p_ins)
        eval_df["text_noisy"] = text_noisy
        eval_df["pred_noisy"] = predict_texts(str(run_dir), eval_df["text_noisy"].astype(str).tolist())
        # Metrics
        y_eval = eval_df["label"].values
        f1_clean = f1_score(y_eval, (eval_df["pred"]>=0.5).astype(int), zero_division=0)
        f1_noisy = f1_score(y_eval, (eval_df["pred_noisy"]>=0.5).astype(int), zero_division=0)
        robust_json = {"f1_clean": float(f1_clean), "f1_noisy": float(f1_noisy), "delta_f1": float(f1_noisy - f1_clean),
                       "p_del": args.p_del, "p_sub": args.p_sub, "p_ins": args.p_ins}
        with open(audit_dir/"robustness.json","w") as f: json.dump(robust_json, f, indent=2)
        eval_df.to_csv(audit_dir/"robustness_scored.csv", index=False)
        print("Robustness ΔF1:", robust_json["delta_f1"])

    # 6) Faithfulness (span training + comprehensiveness/sufficiency)
    faith_json = None
    if args.spans_train_csv and args.spans_val_csv:
        # Train spans
        cmd = ["python","-m","cyberhate_lab.xai.train_spans","--train",args.spans_train_csv,"--val",args.spans_val_csv,"--model",args.model,"--out",args.span_model_out]
        shell(cmd)
        # Evaluate
        shell(["python","scripts/eval_rationales.py","--model-path",str(run_dir),"--val-csv",args.spans_val_csv,"--rationales-npy",str(Path(args.span_model_out)/"val_token_rationales.npy")])
        # If eval script saved arrays, compute means and store JSON
        if Path("faithfulness_comp.npy").exists() and Path("faithfulness_suff.npy").exists():
            comp = float(np.mean(np.load("faithfulness_comp.npy")))
            suff = float(np.mean(np.load("faithfulness_suff.npy")))
            faith_json = {"comprehensiveness_drop": comp, "sufficiency_drop": suff}
            with open(audit_dir/"faithfulness.json","w") as f: json.dump(faith_json, f, indent=2)
            os.replace("faithfulness_comp.npy", audit_dir/"faithfulness_comp.npy")
            os.replace("faithfulness_suff.npy", audit_dir/"faithfulness_suff.npy")
            print("Faithfulness:", faith_json)

    # 7) Risk–coverage plot
    shell(["python","scripts/risk_coverage.py","--probs",str(run_dir/'val_probs.npy'),"--labels",str(run_dir/'val_labels.npy'),"--out",str(audit_dir/'risk_coverage.png')])

    # FINAL SUMMARY
    summary = {
        "aggregate_pre": agg_pre,
        "aggregate_post": agg_post,
        "fairness_csv": str(fairness_csv) if fairness_csv else None,
        "ctf": ctf_json,
        "hatecheck_csv": str(hatecheck_csv_out) if hatecheck_csv_out else None,
        "robustness": robust_json,
        "faithfulness": faith_json,
        "risk_coverage_png": str(audit_dir/'risk_coverage.png')
    }
    with open(audit_dir/"summary.json","w") as f:
        json.dump(summary, f, indent=2)
    # Also write a small CSV with core numbers
    core = {
        "f1_pre": agg_pre["f1"], "auc_pre": agg_pre["roc_auc"], "ece_pre": agg_pre["ece_15"],
        "f1_post": agg_post["f1"], "auc_post": agg_post["roc_auc"], "ece_post": agg_post["ece_15"],
        "ctf_gap": (ctf_json or {}).get("ctf_gap") if ctf_json else None,
        "robust_delta_f1": (robust_json or {}).get("delta_f1") if robust_json else None,
        "faith_comp_drop": (faith_json or {}).get("comprehensiveness_drop") if faith_json else None,
        "faith_suff_drop": (faith_json or {}).get("sufficiency_drop") if faith_json else None,
    }
    pd.DataFrame([core]).to_csv(audit_dir/"summary_core.csv", index=False)
    print("Wrote summary to", audit_dir)

if __name__ == "__main__":
    main()
