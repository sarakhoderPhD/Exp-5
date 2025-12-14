
import re
from typing import List, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .preprocess import IDENTITY_MAP

def _pseudo_log_likelihood(text: str, tok, mlm, max_length=256):
    enc = tok(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    input_ids = enc['input_ids'][0]
    attn = enc['attention_mask'][0]
    logp = 0.0
    with torch.inference_mode():
        for i in range(len(input_ids)):
            if attn[i] == 0:
                continue
            orig = input_ids[i].item()
            if orig in tok.all_special_ids:
                continue
            masked = input_ids.clone()
            masked[i] = tok.mask_token_id
            logits = mlm(input_ids=masked.unsqueeze(0), attention_mask=attn.unsqueeze(0)).logits[0, i]
            logp += torch.log_softmax(logits, dim=-1)[orig].item()
    return logp

def generate_counterfactuals(texts: List[str], mapping: Dict[str,str]=None, lm_name: str='roberta-base', topk: int=50, pll_delta_thresh: float=5.0, max_length:int=256):
    mapping = mapping or IDENTITY_MAP
    tok = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
    mlm = AutoModelForMaskedLM.from_pretrained(lm_name).eval()
    results = []
    with torch.inference_mode():
        for text in texts:
            cand_texts = []
            base_pll = _pseudo_log_likelihood(text, tok, mlm, max_length=max_length)
            low = text.lower()
            for src, tgt in mapping.items():
                if src in low:
                    pattern = re.compile(rf'(?<!\w){re.escape(src)}(?!\w)', re.IGNORECASE)
                    for m in pattern.finditer(low):
                        s,e = m.span()
                        masked_text = text[:s] + tok.mask_token + text[e:]
                        enc = tok(masked_text, return_tensors='pt')
                        pos = (enc['input_ids'][0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
                        if len(pos)==0: 
                            continue
                        logits = mlm(**enc).logits[0, pos[0]]
                        topk_ids = torch.topk(logits, k=topk).indices.tolist()
                        tgt_id = tok.convert_tokens_to_ids(tgt)
                        if tgt_id in topk_ids:
                            swapped = text[:s] + tgt + text[e:]
                            pll = _pseudo_log_likelihood(swapped, tok, mlm, max_length=max_length)
                            if (base_pll - pll) <= pll_delta_thresh:
                                cand_texts.append(swapped)
            results.append(list(dict.fromkeys(cand_texts)))
    return results
