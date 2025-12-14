
import random, numpy as np
from typing import List

def corrupt_text(s: str, p_del=0.02, p_sub=0.02, p_ins=0.02, charset="abcdefghijklmnopqrstuvwxyz0123456789") -> str:
    out = []
    for ch in s:
        r = random.random()
        if ch.isspace():
            out.append(ch); continue
        if r < p_del:
            continue
        elif r < p_del + p_sub:
            out.append(random.choice(charset))
        else:
            out.append(ch)
            if random.random() < p_ins:
                out.append(random.choice(charset))
    return "".join(out)

def batch_corrupt(texts: List[str], p_del=0.02, p_sub=0.02, p_ins=0.02) -> List[str]:
    return [corrupt_text(t, p_del, p_sub, p_ins) for t in texts]
