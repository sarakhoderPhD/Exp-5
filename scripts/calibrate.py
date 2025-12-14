#!/usr/bin/env python
import argparse, numpy as np, pandas as pd
from cyberhate_lab.evaluation.calibration import temperature_scaling, expected_calibration_error

ap = argparse.ArgumentParser()
ap.add_argument("--logits", help="Numpy .npy of validation logits shape [N,2]", required=False)
ap.add_argument("--probs", help="Numpy .npy of validation probabilities shape [N] or [N,2]", required=False)
ap.add_argument("--labels", help="Numpy .npy or CSV with labels", required=True)
ap.add_argument("--out", help="Path to write calibrated probs (.npy)", required=True)
args = ap.parse_args()

y = np.load(args.labels) if args.labels.endswith(".npy") else pd.read_csv(args.labels)["label"].values

if args.logits:
    logits = np.load(args.logits)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    T, calib = temperature_scaling(logits, y)
    print(f"Fitted temperature T={T:.3f}")
    print(f"ECE (before): {expected_calibration_error((logits[:,1]).copy(), y):.4f}")
    print(f"ECE (after):  {expected_calibration_error(calib.copy(), y):.4f}")
    np.save(args.out, calib)
elif args.probs:
    p = np.load(args.probs)
    if p.ndim == 2:
        p = p[:,1]
    eps = 1e-6
    p_clip = np.clip(p, eps, 1-eps)
    logits = np.stack([np.log(1-p_clip), np.log(p_clip)], axis=1)
    T, calib = temperature_scaling(logits, y)
    print(f"Fitted temperature T={T:.3f}")
    print(f"ECE (before): {expected_calibration_error(p, y):.4f}")
    print(f"ECE (after):  {expected_calibration_error(calib, y):.4f}")
    np.save(args.out, calib)
else:
    raise SystemExit("Provide --logits or --probs")
