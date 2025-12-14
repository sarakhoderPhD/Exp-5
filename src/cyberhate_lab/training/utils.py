
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    if labels.ndim == 1:
        pos = labels.mean()
        neg = 1 - pos
        w_pos = 1.0 / max(pos, 1e-6)
        w_neg = 1.0 / max(neg, 1e-6)
        w = torch.tensor([w_neg, w_pos], dtype=torch.float32)
        w = w / w.mean()
        return w
    else:
        prev = labels.mean(axis=0)
        w = 1.0 / np.clip(prev, 1e-6, 1.0)
        w = w / w.mean()
        return torch.tensor(w, dtype=torch.float32)

def metrics_binary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_hat = (y_prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat, average="binary", zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out

def metrics_multilabel(y_true: np.ndarray, y_prob: np.ndarray, threshold: float=0.5) -> Dict[str, float]:
    y_hat = (y_prob >= threshold).astype(int)
    return {
        "f1_micro": f1_score(y_true, y_hat, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_hat, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_hat, average="weighted", zero_division=0),
    }
