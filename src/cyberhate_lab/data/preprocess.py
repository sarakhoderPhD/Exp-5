
import re
from typing import List, Tuple
import emoji as emoji_lib
from wordsegment import load as ws_load, segment as ws_segment

ws_load()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
MULTISPACE_RE = re.compile(r"\s+")

def normalize_text(text: str, lower: bool=True, strip_urls: bool=True, strip_users: bool=True) -> str:
    """Basic text normalisation suitable for Twitter/Reddit/TikTok captions."""
    if strip_urls:
        text = URL_RE.sub(" ", text)
    if strip_users:
        text = USER_RE.sub("@USER", text)
    text = text.replace("\u200d","")  # zero-width joiner
    if lower:
        text = text.lower()
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

def segment_hashtags(text: str) -> str:
    """Replace hashtags with a segmented form: '#StopAsianHate' -> '<hashtag> stop asian hate </hashtag>'"""
    def repl(m):
        tag = m.group(0)[1:]
        segmented = " ".join(ws_segment(tag))
        return f"<hashtag> {segmented} </hashtag>"
    return HASHTAG_RE.sub(repl, text)

def extract_emojis(text: str) -> List[str]:
    return [ch for ch in text if ch in emoji_lib.EMOJI_DATA]

def replace_emojis_with_names(text: str) -> str:
    """Map emoji to CLDR names, useful when models lack native emoji embeddings."""
    out = []
    for ch in text:
        if ch in emoji_lib.EMOJI_DATA:
            name = emoji_lib.EMOJI_DATA[ch].get('en', '').replace(":", "").replace("_"," ")
            out.append(f" <emoji_{name}> ")
        else:
            out.append(ch)
    return "".join(out)

IDENTITY_MAP = {
    "women": "men", "woman":"man", "girl":"boy", "girls":"boys",
    "men":"women", "man":"woman", "boy":"girl", "boys":"girls",
    "muslims":"christians", "muslim":"christian",
    "christians":"muslims", "christian":"muslim",
    "black":"white", "blacks":"whites", "white":"black", "whites":"blacks",
    "jews":"christians", "jew":"christian",
    "gay":"straight", "gays":"straights", "lesbian":"straight",
    "immigrants":"citizens", "immigrant":"citizen",
}

def swap_identity_terms(text: str, mapping: dict = None) -> str:
    """Simple token-level identity swapper (context-agnostic). Use with plausibility filtering upstream."""
    mapping = mapping or IDENTITY_MAP
    toks = re.findall(r"\w+|\W+", text.lower())
    out = []
    for t in toks:
        if t.isalpha() and t in mapping:
            out.append(mapping[t])
        else:
            out.append(t)
    return "".join(out)
