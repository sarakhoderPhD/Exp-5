
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int=15):
    bins = np.linspace(0, 1, n_bins+1)
    binned = np.digitize(probs, bins) - 1
    accs, confs, counts = [], [], []
    for i in range(n_bins):
        mask = binned == i
        if mask.sum() == 0:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
            continue
        accs.append(labels[mask].mean())
        confs.append(probs[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(accs), np.array(confs), np.array(counts), bins

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int=15) -> float:
    accs, confs, counts, bins = reliability_bins(probs, labels, n_bins)
    weights = counts / max(1, counts.sum())
    diffs = np.abs(accs - confs)
    diffs[np.isnan(diffs)] = 0.0
    return float((weights * diffs).sum())

class _TempScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))  # T=1
    def forward(self, logits):
        return logits / torch.exp(self.log_t)

def temperature_scaling(logits: np.ndarray, labels: np.ndarray, max_iter: int=200) -> Tuple[float, np.ndarray]:
    device = "cpu"
    model = _TempScale().to(device)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")
    nll = nn.CrossEntropyLoss()

    def _closure():
        opt.zero_grad()
        loss = nll(model(logits_t), labels_t)
        loss.backward()
        return loss

    opt.step(_closure)
    T = float(torch.exp(model.log_t).detach().cpu().numpy())
    with torch.no_grad():
        calibrated = torch.softmax(model(logits_t), dim=-1)[:,1].cpu().numpy()
    return T, calibrated
