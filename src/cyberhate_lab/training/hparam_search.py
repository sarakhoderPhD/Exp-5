
import argparse, os, numpy as np, optuna, torch
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, set_seed
from ..training.datasets import load_text_dataset
from ..training.utils import metrics_binary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="runs/hparam_search")
    args = ap.parse_args()
    set_seed(42)

    train_ds, val_ds, tok, schema = load_text_dataset(args.train, args.val, text_col=args.text_col, label_col=args.label_col, model_name=args.model)

    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
        bs = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs = trial.suggest_int("epochs", 2, 6)
        wd = trial.suggest_float("weight_decay", 0.0, 0.1)

        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
        args_tr = TrainingArguments(
            output_dir=f"{args.out}/trial_{trial.number}",
            learning_rate=lr,
            weight_decay=wd,
            num_train_epochs=epochs,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:,1]
            m = metrics_binary(labels, probs)
            return {"f1": m["f1"], "roc_auc": m["roc_auc"]}

        trainer = Trainer(model=model, args=args_tr, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tok, compute_metrics=compute_metrics)
        trainer.train()
        res = trainer.evaluate()
        return res["eval_f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    os.makedirs(args.out, exist_ok=True)
    study.trials_dataframe().to_csv(f"{args.out}/optuna_trials.csv", index=False)
    print("Best params:", study.best_trial.params)

if __name__ == "__main__":
    main()
