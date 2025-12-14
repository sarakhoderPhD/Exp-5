import argparse, os, numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, set_seed
from .datasets import load_text_dataset
from .modeling import MultiTaskSequenceModel
from .utils import metrics_binary, metrics_multilabel, compute_class_weights

# ---------- helper to merge context ----------
def _compose_text_column(df, text_col, ctx_col, sep="</s>"):
    """
    Return strings where context (if present) is prepended:
    "<ctx> </s> <text>"
    """
    base = df[text_col].astype(str)
    if ctx_col and ctx_col in df.columns:
        ctx = df[ctx_col].fillna("").astype(str).str.strip()
        mixed = ctx.mask(ctx.eq(""), base, other=ctx + f" {sep} " + base)
        return mixed.tolist()
    return base.tolist()

def build_model(model_name: str, schema, use_multitask: bool, aux_tasks):
    if use_multitask or schema["problem_type"] == "multilabel":
        return MultiTaskSequenceModel(
            model_name,
            n_labels=schema["n_labels"],
            problem_type=schema["problem_type"],
            aux_tasks=aux_tasks,
        )
    else:
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Column name we’ll create for the composed text
MODEL_TEXT_COL = "__text_for_model"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--multilabel-cols", default="")
    ap.add_argument("--aux-sarcasm-col", default="")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/text_model")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--ctx-col", default=None, help="Optional column with conversation context to prepend")
    args = ap.parse_args()

    set_seed(args.seed)

    # ---------- context handling (writes sidecar CSVs & rewires args) ----------
    if args.ctx_col:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(args.train)
        val_df   = pd.read_csv(args.val)

        train_df[MODEL_TEXT_COL] = _compose_text_column(train_df, args.text_col, args.ctx_col)
        val_df[MODEL_TEXT_COL]   = _compose_text_column(val_df,   args.text_col, args.ctx_col)

        train_ctx = out_dir / "train_ctx.csv"
        val_ctx   = out_dir / "val_ctx.csv"
        train_df.to_csv(train_ctx, index=False)
        val_df.to_csv(val_ctx, index=False)

        # Re-point downstream to the context-augmented files/column
        args.train    = str(train_ctx)
        args.val      = str(val_ctx)
        args.text_col = MODEL_TEXT_COL
    # ---------- end context handling ----------
    
    multilabel_cols = [c.strip() for c in args.multilabel_cols.split(",") if c.strip()] or None
    aux_cols = {}
    aux_tasks = {}
    if args.aux_sarcasm_col:
        aux_cols["sarcasm"] = args.aux_sarcasm_col
        aux_tasks["sarcasm"] = 2

    train_ds, val_ds, tok, schema = load_text_dataset(
        args.train, args.val, text_col=args.text_col, label_col=args.label_col,
        multilabel_cols=multilabel_cols, aux_cols=aux_cols, model_name=args.model, max_length=args.max_length
    )
    os.makedirs(args.out, exist_ok=True)

    model = build_model(args.model, schema, use_multitask=bool(aux_tasks), aux_tasks=aux_tasks)

    ce_weights = None
    if args.class_weights:
        y = np.stack([x.numpy() for x in train_ds["labels"]])
        w = compute_class_weights(y if y.ndim > 1 else y.reshape(-1))
        if schema["problem_type"] == "binary" and not aux_tasks:
            ce_weights = w

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if schema["problem_type"] == "multilabel":
            probs = 1/(1+np.exp(-logits))
            return metrics_multilabel(labels, probs)
        else:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:,1]
            return metrics_binary(labels, probs)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = outputs.get("loss", None)
            if loss is None:
                if schema["problem_type"] == "multilabel":
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels.float())
                else:
                    if ce_weights is not None:
                        loss_fct = torch.nn.CrossEntropyLoss(weight=ce_weights.to(logits.device))
                    else:
                        loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    args_tr = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1" if schema["problem_type"]=="binary" else "f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(args.out)
    preds = trainer.predict(val_ds)
    logits = preds.predictions
    labels = preds.label_ids
    if schema["problem_type"] == "multilabel":
        probs = 1/(1+np.exp(-logits))
    else:
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:,1]

    import numpy as np
    np.save(f"{args.out}/val_logits.npy", logits)
    np.save(f"{args.out}/val_probs.npy", probs)
    np.save(f"{args.out}/val_labels.npy", labels)
    print(f"Saved validation tensors to {args.out}")
    print("Training complete. Proceed to calibration & fairness evaluation.")

if __name__ == "__main__":
    main()
