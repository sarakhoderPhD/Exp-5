
from typing import Optional, Dict
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MultiTaskSequenceModel(nn.Module):
    def __init__(self, model_name: str, n_labels: int = 2, problem_type: str = "binary", p_drop: float = 0.1, aux_tasks: Optional[Dict[str,int]] = None):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.cfg)
        hidden = self.cfg.hidden_size
        self.dropout = nn.Dropout(p_drop)
        self.problem_type = problem_type
        self.main_head = nn.Linear(hidden, n_labels)
        self.aux_heads = nn.ModuleDict()
        self.aux_tasks = aux_tasks or {}
        for name, k in self.aux_tasks.items():
            self.aux_heads[name] = nn.Linear(hidden, k)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0]
        h = self.dropout(cls)
        logits = self.main_head(h)
        loss = None
        if labels is not None:
            if self.problem_type == "multilabel":
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels.long())
        aux_logits = {}
        for name in self.aux_tasks.keys():
            aux_logit = self.aux_heads[name](h)
            aux_logits[name] = aux_logit
            aux_labels = kwargs.get(f"{name}_labels", None)
            if aux_labels is not None:
                aux_loss_fn = nn.CrossEntropyLoss()
                loss = loss + 0.2 * aux_loss_fn(aux_logit, aux_labels.long()) if loss is not None else 0.2 * aux_loss_fn(aux_logit, aux_labels.long())
        return {"loss": loss, "logits": logits, **{f"logits_{k}":v for k,v in aux_logits.items()}}
