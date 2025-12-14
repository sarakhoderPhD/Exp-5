import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path


def load_labels(y_path, label_col="label"):
    df = pd.read_csv(y_path)
    if label_col not in df.columns:
        raise ValueError(
            "Label column '{0}' not found in {1}. Available columns: {2}".format(
                label_col, y_path, df.columns.tolist()
            )
        )
    y = df[label_col].values
    return df, y


def load_probs(pred_path, id_col=None, prob_col=None):
    df = pd.read_csv(pred_path)

    # If you know the prob column name explicitly, use it
    if prob_col is not None:
        if prob_col not in df.columns:
            raise ValueError(
                "Requested prob_col '{0}' not found in {1}. Available: {2}".format(
                    prob_col, pred_path, df.columns.tolist()
                )
            )
        return df, df[prob_col].values

    # Otherwise, infer: take the last numeric column that is not clearly an id/label
    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c.lower() not in ("label", "y", "target")
    ]
    if not numeric_cols:
        raise ValueError(
            "No numeric probability column inferred in {0}. Columns: {1}".format(
                pred_path, df.columns.tolist()
            )
        )

    col = numeric_cols[-1]
    return df, df[col].values


def compute_metrics(y_true, y_scores):
    roc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return roc, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--y", default="external/test_with_ctx.csv")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--ctx", default="external/out/preds_ctx.csv")
    parser.add_argument("--drop", default="external/out/preds_drop.csv")
    parser.add_argument("--shuffle", default="external/out/preds_shuffle.csv")
    parser.add_argument("--trunc", default="external/out/preds_tr64.csv")
    parser.add_argument(
        "--out",
        default="outputs/table_robustness.csv",
        help="Output CSV path for robustness metrics.",
    )
    parser.add_argument(
        "--prob-col",
        default=None,
        help="Optional explicit probability column name (e.g. 'prob_1').",
    )
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 1) Load labels
    _, y = load_labels(args.y, args.label_col)

    rows = []
    for condition, path in [
        ("True", args.ctx),
        ("Drop", args.drop),
        ("Shuffle", args.shuffle),
        ("Truncate", args.trunc),
    ]:
        _, probs = load_probs(path, prob_col=args.prob_col)

        # Assuming row order in preds matches external/test_with_ctx.csv
        roc, ap = compute_metrics(y, probs)
        rows.append(
            {
                "condition": condition,
                "roc_auc": roc,
                "avg_precision": ap,
            }
        )

    out_df = pd.DataFrame(rows)
    print("\nRobustness metrics (external set):")
    print(out_df.to_string(index=False))

    out_df.to_csv(args.out, index=False)
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()