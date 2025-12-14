
import numpy as np

def entropy(p: np.ndarray, eps: float=1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1-eps)
    return - (p*np.log2(p) + (1-p)*np.log2(1-p))

def select_top_k_uncertain(probs: np.ndarray, k: int) -> np.ndarray:
    H = entropy(probs)
    return np.argsort(-H)[:k]
