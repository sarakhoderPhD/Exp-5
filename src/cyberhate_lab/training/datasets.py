
from typing import Dict, List, Optional
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def load_text_dataset(
    train_csv: str,
    val_csv: str,
    text_col: str = "text",
    label_col: str = "label",
    multilabel_cols: Optional[List[str]] = None,
    aux_cols: Optional[Dict[str, str]] = None,
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    schema = {"problem_type": "binary", "n_labels": 2}
    if multilabel_cols:
        schema["problem_type"] = "multilabel"
        schema["n_labels"] = len(multilabel_cols)

    def _build_labels(df: pd.DataFrame):
        if multilabel_cols:
            y = df[multilabel_cols].astype("float32").values
        else:
            y = df[label_col].astype("int64").values
        return y

    def _tokenize(batch):
        return tok(batch[text_col], padding="max_length", truncation=True, max_length=max_length)

    def _to_dataset(df: pd.DataFrame):
        ds = Dataset.from_pandas(df.reset_index(drop=True))
        ds = ds.map(_tokenize, batched=True, remove_columns=[c for c in df.columns if c not in [text_col]])
        y = _build_labels(df)
        ds = ds.add_column("labels", y.tolist())
        if aux_cols:
            for k, col in aux_cols.items():
                if col in df.columns:
                    ds = ds.add_column(f"{k}_labels", df[col].astype("int64").tolist())
        ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"] + [f"{k}_labels" for k in (aux_cols or {}).keys()])
        return ds

    train_ds = _to_dataset(train_df)
    val_ds = _to_dataset(val_df)
    return train_ds, val_ds, tok, schema
