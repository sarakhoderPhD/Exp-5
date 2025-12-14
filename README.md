# cyberhate_lab

A modular toolkit aligned to your PhD plan for **cyber hate detection** with rigorous evaluation, fairness diagnostics, calibration, and robustness.
It plugs into your existing BERT/RoBERTa baseline (e.g., `Experiment_3.ipynb`) and supports multilingual adapter training.

## Features
- **Evaluation**: ROC-AUC/F1, **ECE calibration**, **fairness** (Subgroup AUC, **BPSN/BNSP**, **CTF gap**), **HateCheck** functional testing.
- **Robustness**: OCR/ASR **noise simulation**, emoji/hashtag preprocessing, counterfactual identity swaps.
- **Training**: Single-task, **multi-label**, and **multi-task** (sarcasm aux) trainers; **Optuna** HP search; **PEFT/LoRA** multilingual adapters.
- **Human-in-the-loop**: entropy **active learning**; **risk–coverage** curves; abstention hook.

## Quick commands
```bash
pip install -r requirements.txt

# Train (binary or multi-label)
python scripts/train_text.py --train data/train.csv --val data/val.csv --text-col text --label-col label --model roberta-base --epochs 3 --out runs/roberta_bin

# Hyperparameter search
python scripts/hparam_search.py --train data/train.csv --val data/val.csv --model roberta-base --trials 20 --out runs/hparam_search

# LoRA adapters for multilingual
python scripts/train_adapters.py --train data/train.csv --val data/val.csv --model xlm-roberta-base --out runs/xlmr_lora

# Calibrate & evaluate fairness
python scripts/calibrate.py --logits runs/roberta_bin/val_logits.npy --labels runs/roberta_bin/val_labels.npy --out runs/roberta_bin/val_probs_cal.npy
python scripts/eval.py --preds eval/preds_with_identities.csv --id-cols identity_women,identity_muslim,identity_black

# HateCheck functional testing
python scripts/run_hatecheck.py --model-path runs/roberta_bin --hatecheck data/hatecheck.csv --device cuda

# Risk–coverage (abstention planning)
python scripts/risk_coverage.py --probs runs/roberta_bin/val_probs.npy --labels runs/roberta_bin/val_labels.npy --out runs/roberta_bin/risk_coverage.png
```
