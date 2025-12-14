
import torch
import torch.nn as nn

class MultiLabelHead(nn.Module):
    def __init__(self, in_dim: int, n_labels: int, p_drop: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p_drop)
        self.out = nn.Linear(in_dim, n_labels)

    def forward(self, x):
        x = self.dropout(x)
        logits = self.out(x)
        return logits

class SelectiveAbstain(nn.Module):
    def __init__(self, threshold: float=0.75):
        super().__init__()
        self.threshold = threshold

    def forward(self, probs: torch.Tensor):
        conf = torch.maximum(probs, 1.0 - probs)
        auto_decide = conf >= self.threshold
        return auto_decide
