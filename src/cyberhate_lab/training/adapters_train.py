
import argparse, os, torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, TaskType
from ..training.datasets import load_text_dataset
from ..training.utils import metrics_binary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="runs/lora")
    args = ap.parse_args()
    set_seed(42)

    train_ds, val_ds, tok, schema = load_text_dataset(args.train, args.val, text_col=args.text_col, label_col=args.label_col, model_name=args.model)

    base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=["query","value","key"])
    model = get_peft_model(base, config)
    model.print_trainable_parameters()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:,1]
        from ..training.utils import metrics_binary as _mb
        m = _mb(labels, probs)
        return {"f1": m["f1"], "roc_auc": m["roc_auc"]}

    args_tr = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=args_tr, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tok, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(args.out)
    print("Saved LoRA-adapted model to", args.out)

if __name__ == "__main__":
    main()
