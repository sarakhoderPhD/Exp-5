
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def _safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan

def subgroup_auc(df: pd.DataFrame, ident_col: str, label_col: str, prob_col: str) -> float:
    mask = df[ident_col].astype(bool)
    return _safe_auc(df.loc[mask, label_col], df.loc[mask, prob_col])

def bpsn_auc(df: pd.DataFrame, ident_col: str, label_col: str, prob_col: str) -> float:
    s = df[ident_col].astype(bool)
    bg_pos = df[~s & (df[label_col] == 1)]
    sg_neg = df[s & (df[label_col] == 0)]
    sub = pd.concat([bg_pos, sg_neg], axis=0)
    return _safe_auc(sub[label_col], sub[prob_col])

def bnsp_auc(df: pd.DataFrame, ident_col: str, label_col: str, prob_col: str) -> float:
    s = df[ident_col].astype(bool)
    bg_neg = df[~s & (df[label_col] == 0)]
    sg_pos = df[s & (df[label_col] == 1)]
    sub = pd.concat([bg_neg, sg_pos], axis=0)
    return _safe_auc(sub[label_col], sub[prob_col])

def compute_bias_table(df: pd.DataFrame, identity_cols: List[str], label_col: str="label", prob_col: str="pred") -> pd.DataFrame:
    rows = []
    for ident in identity_cols:
        rows.append({
            "identity": ident,
            "subgroup_auc": subgroup_auc(df, ident, label_col, prob_col),
            "bpsn_auc": bpsn_auc(df, ident, label_col, prob_col),
            "bnsp_auc": bnsp_auc(df, ident, label_col, prob_col),
            "prevalence": float(df[ident].mean())
        })
    out = pd.DataFrame(rows)
    out["macro_bias_auc"] = out[["subgroup_auc","bpsn_auc","bnsp_auc"]].mean(axis=1)
    return out

def counterfactual_token_fairness(pairs: pd.DataFrame, group_col: str="pair_id", label_col: str="label", prob_col: str="pred") -> float:
    abs_diffs = []
    for pair_id, group in pairs.groupby(group_col):
        if len(group) != 2: 
            continue
        probs = group[prob_col].values
        abs_diffs.append(abs(probs[0] - probs[1]))
    return float(np.nanmean(abs_diffs)) if abs_diffs else np.nan
