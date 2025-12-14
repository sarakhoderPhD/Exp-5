#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Reddit datasets (train/val CSVs) that faithfully replicate Experiment_3.ipynb:
- Fetch from the exact same subreddits you used
- Apply the same cleaning (URLs -> removed, 'u/...' -> 'u_user', whitespace normalize)
- Construct the same conversation context (parent text + child)
- Auto-label with 'cardiffnlp/twitter-roberta-base-offensive'
- Export CSVs for the end-to-end pipeline (text,label)

Usage (typical):
  export REDDIT_CLIENT_ID=...
  export REDDIT_CLIENT_SECRET=...
  export REDDIT_USER_AGENT="script by Mediocre_Decision_28 for PhD experiment"

  python scripts/build_reddit_autolabeled_csv.py \
    --items-per-sub 500 \
    --fetch hot \
    --sleep 1.0 \
    --label-on combined_text \
    --train-text-field combined_text \
    --test-size 0.2

Notes:
- Labels are from the classifier (no subreddit-based labels).
- 'text' column in CSV defaults to combined_text to mirror your baseline training.
"""

import os, re, time, argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import prawcore
from tqdm.auto import tqdm

# -------------------------
# 1) Experiment_3 constants
# -------------------------

# --- Subreddit groups you defined/expanded ---
BASELINE_SUBS = [
    "AskReddit", "ask", "notstupidquestions", "explainlikeimfive",
    "todayilearned", "casualconversation", "science", "mildlyinteresting",
    "MadeMeSmile", "wholesomememes", "HumansBeingBros", "eyebleach",
    "baking", "gardening", "crafts", "books", "movies", "gaming"
]
MIXED_CONFLICT_SUBS = [
    "amitheasshole", "unpopularopinion", "politics", "Conservative",
    "relationship_advice", "TwoXChromosomes", "TrueOffMyChest",
    "facepalm", "Cringe"
]
HIGH_PREVALENCE_SUBS = [
    "MensRights", "ForeverAlone", "KotakuInAction",
    "aznidentity", "AsianMasculinity", "PoliticalCompassMemes",
    "Russia", "Sino"
]
# NOTE: r/TheRedPill causes quarantine/auth problems. We omit it by default.
SNARK_SUBS = [
    "DuggarsSnark", "NYCinfluencersnark", "LAinfluencersnark",
    "Blogsnark", "tiktokgossip", "parentsnark", "KUWTKSnarkUncensored"
]
META_SUBS = [
    "AgainstHateSubreddits", "IncelTears", "TheBluePill"
]

EXPERIMENT3_SUBS = BASELINE_SUBS + MIXED_CONFLICT_SUBS + HIGH_PREVALENCE_SUBS + SNARK_SUBS + META_SUBS

#EXPERIMENT3_SUBS = [
#    "AskReddit", "amitheasshole", "ask",
#    "notstupidquestions", "unpopularopinion", "mildlyinteresting"
#]
DEFAULT_AUTOLABEL_MODEL = "cardiffnlp/twitter-roberta-base-offensive"

def require_env(k: str) -> str:
    v = os.getenv(k)
    if not v:
        raise SystemExit(f"Please set environment variable: {k}")
    return v

#def get_reddit():
#    import praw
#    cid = require_env("REDDIT_CLIENT_ID")
#    csc = require_env("REDDIT_CLIENT_SECRET")
#    ua  = os.getenv("REDDIT_USER_AGENT", "cyberhate_phd_pipeline")
#    return praw.Reddit(client_id=cid, client_secret=csc, user_agent=ua, check_for_async=False)

def get_reddit():
    import praw
    cid = require_env("REDDIT_CLIENT_ID")
    csc = require_env("REDDIT_CLIENT_SECRET")
    ua  = os.getenv("REDDIT_USER_AGENT", "cyberhate_phd_pipeline")
    reddit = praw.Reddit(client_id=cid, client_secret=csc, user_agent=ua, check_for_async=False)
    reddit.read_only = True 
    return reddit

# Cleaning exactly as in Experiment_3
def basic_cleaning(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"u/\S+", "u_user", text)  # anonymize Reddit mentions
    text = re.sub(r"\s+", " ", text).strip() # whitespace normalize
    return text

def parse_parent_id(pid: str) -> str:
    if not isinstance(pid, str) or not pid:
        return ""
    if pid.startswith("t1_") or pid.startswith("t3_"):
        return pid.split("_", 1)[1]
    return ""

def fetch_sr(reddit, sub_name: str, items: int, fetch: str, sleep: float, include_comments: bool = True):
    """
    Yield dict rows for submissions + (optionally) comments, with a per-subreddit progress bar.
    - Progress bar tracks #submissions processed; comment count shown in the bar's postfix.
    - Forbidden/private/quarantined subs are cleanly skipped (no crash).
    """
    try:
        sr = reddit.subreddit(sub_name)
        if fetch == "hot":
            iterator = sr.hot(limit=items)
        elif fetch == "new":
            iterator = sr.new(limit=items)
        elif fetch == "top":
            iterator = sr.top(limit=items)
        else:
            raise SystemExit(f"--fetch must be one of hot/new/top, got: {fetch}")
    except Exception as e:
        tqdm.write(f"[SKIP] r/{sub_name}: cannot create listing iterator ({type(e).__name__}: {e})")
        return

    # progress bar per subreddit; 'total' is the requested #submissions
    bar = tqdm(
        iterator,
        total=items,
        desc=f"r/{sub_name} • {fetch}",
        unit="post",
        dynamic_ncols=True,
        leave=False
    )

    n_submissions = 0
    n_comments = 0
    try:
        for submission in bar:
            # submission row
            submission_text = (submission.title or "") + "\n" + (submission.selftext or "")
            yield {
                "type": "submission",
                "id": submission.id,
                "parent_id": None,
                "author": str(submission.author),
                "created_utc": getattr(submission, "created_utc", None),
                "body": submission_text,
                "source_sub": sub_name,
            }
            n_submissions += 1

            # comments (optional): show a compact inner progress bar, but don't spam the console
            if include_comments:
                try:
                    submission.comments.replace_more(limit=0)
                    comments = submission.comments.list()  # list so we know total
                    if comments:
                        cbar = tqdm(
                            comments,
                            desc="  ↳ comments",
                            unit="cmt",
                            dynamic_ncols=True,
                            leave=False
                        )
                        for c in cbar:
                            yield {
                                "type": "comment",
                                "id": c.id,
                                "parent_id": c.parent_id,
                                "author": str(c.author),
                                "created_utc": getattr(c, "created_utc", None),
                                "body": getattr(c, "body", "") or "",
                                "source_sub": sub_name,
                            }
                            n_comments += 1
                        cbar.close()
                except Exception:
                    # swallow transient comment/replace_more issues
                    pass

            # update postfix with running counts
            bar.set_postfix_str(f"subs={n_submissions}, cmts={n_comments}")
            time.sleep(sleep)

    except (prawcore.exceptions.Forbidden,
            prawcore.exceptions.NotFound,
            prawcore.exceptions.OAuthException,
            prawcore.exceptions.ResponseException,
            prawcore.exceptions.RequestException) as e:
        tqdm.write(f"[SKIP] r/{sub_name}: {type(e).__name__} ({e})")
        return
    except Exception as e:
        tqdm.write(f"[SKIP] r/{sub_name}: unexpected {type(e).__name__} ({e})")
        return
    finally:
        bar.close()
        tqdm.write(f"[DONE] r/{sub_name}: submissions={n_submissions}, comments={n_comments}")



def auto_label_series(texts, model_name=DEFAULT_AUTOLABEL_MODEL, device_preference="auto",
                      batch_size=32, max_length=256, positive_threshold=0.50):
    """
    Label texts using a HF pipeline with return_all_scores=True and a probability threshold
    for the 'offensive' class. This is more stable than top-1 on Reddit domain.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    import numpy as np
    import pandas as pd

    # device
    if device_preference == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif device_preference == "cpu":
        device = -1
    else:
        device = int(device_preference)

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)

    clf = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        device=device,
        truncation=True,
        max_length=max_length,
        return_all_scores=True
    )

    # Figure out which label is 'offensive'
    # Prefer config.id2label if present; else infer by name
    id2label = getattr(mdl.config, "id2label", None)
    offence_idx = None
    label_names = []
    if isinstance(id2label, dict) and id2label:
        # id2label may have int keys or str keys; normalize to int ordering if possible
        try:
            max_k = max(int(k) for k in id2label.keys())
            label_names = [id2label[i] for i in range(max_k+1)]
        except Exception:
            # fall back: sort by key string
            label_names = [v for _, v in sorted(id2label.items(), key=lambda x: str(x[0]))]
    # Heuristic fallback if names missing
    if not label_names:
        # common with some HF models
        label_names = ["LABEL_0", "LABEL_1"]

    # find index containing 'offen' or 'tox' or 'hate'
    lname = [str(x).lower() for x in label_names]
    for i, n in enumerate(lname):
        if ("offen" in n) or ("tox" in n) or ("abuse" in n) or ("hate" in n):
            offence_idx = i
            break
    # If still None, guess class 1 is offensive (typical for many binaries)
    if offence_idx is None and len(label_names) >= 2:
        offence_idx = 1
    if offence_idx is None:
        # worst-case fallback
        offence_idx = 1

    labels = []
    tlist = texts.tolist()
    for i in range(0, len(tlist), batch_size):
        batch = tlist[i:i+batch_size]
        outs = clf(batch)  # each item: [{'label': '...', 'score': p}, {'label': '...', 'score': p}]
        for scores in outs:
            # normalize ordering if needed: map by label_names
            # Build a small map name->score
            name2p = {d["label"]: float(d["score"]) for d in scores}
            # Try to access by the label_names order; if not present, use any available order
            if all(name in name2p for name in label_names):
                p_off = name2p[label_names[offence_idx]]
            else:
                # If labels are 'LABEL_0'/'LABEL_1' or 'OFFENSE'/'NOT_OFFENSE' etc., pick the one matching heuristics
                # Try to find any key containing 'offen|tox|abuse|hate'
                p_off = None
                for k, v in name2p.items():
                    lk = k.lower()
                    if ("offen" in lk) or ("tox" in lk) or ("abuse" in lk) or ("hate" in lk):
                        p_off = v
                        break
                if p_off is None:
                    # Fallback: assume index 1 is offensive
                    # Need a stable order; pick the second score if present
                    try:
                        p_off = float(scores[1]["score"])
                    except Exception:
                        p_off = 0.0
            labels.append(1 if p_off >= positive_threshold else 0)

    return pd.Series(labels, index=texts.index, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default=",".join(EXPERIMENT3_SUBS),
                    help="Comma-separated subreddit names (defaults to the Experiment_3 list).")
    ap.add_argument("--items-per-sub", type=int, default=500)
    ap.add_argument("--fetch", choices=["hot","new","top"], default="hot")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between requests to be gentle on API.")
    ap.add_argument("--include-comments", action="store_true", default=True)
    ap.add_argument("--autolabel-model", default=DEFAULT_AUTOLABEL_MODEL)
    ap.add_argument("--label-on", choices=["clean_text","combined_text"], default="combined_text",
                    help="Which field to pass into the auto-labeller (notebook demo used clean_text; training used combined_text).")
    ap.add_argument("--train-text-field", choices=["clean_text","combined_text"], default="combined_text",
                    help="Which field to export as 'text' in CSV. Default mirrors your baseline training (combined_text).")
    ap.add_argument("--device", default="auto", help="'auto', 'cpu', or CUDA device id (e.g., 0)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-train", default="data/train.csv")
    ap.add_argument("--out-val", default="data/val.csv")
    ap.add_argument("--out-eval", default="eval/eval_texts.csv")
    ap.add_argument("--positive-threshold", type=float, default=0.50,
                    help="Probability threshold for offensive/abusive label when using return_all_scores.")
    ap.add_argument("--min-chars", type=int, default=12,
                    help="Minimum character length for a sample to be sent to the auto-labeller.")
    args = ap.parse_args()

    subs = [s.strip() for s in args.subs.split(",") if s.strip()]
    if not subs:
        raise SystemExit("No subreddits provided.")

    # IO
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("eval").mkdir(parents=True, exist_ok=True)

    # 1) Reddit fetch
    reddit = get_reddit()
    rows = []
    for s in subs:
        for row in fetch_sr(reddit, s, args.items_per_sub, args.fetch, args.sleep, include_comments=args.include_comments):
            rows.append(row)
    if not rows:
        raise SystemExit("Fetched 0 rows; check credentials or subreddit availability.")

    df = pd.DataFrame(rows)

    # 2) Cleaning from Experiment_3
    df["clean_text"] = df["body"].apply(basic_cleaning)

    # 3) Conversation context (parent mapping) exactly as in Experiment_3
    df["parsed_parent_id"] = df["parent_id"].fillna("").apply(parse_parent_id)
    id2text = dict(zip(df["id"], df["clean_text"]))
    df["parent_text"] = df["parsed_parent_id"].map(id2text).fillna("")
    df["combined_text"] = "CONTEXT: " + df["parent_text"] + "\n---\nCHILD: " + df["clean_text"]

    # Filter short texts (avoid 'thanks', 'lol', etc.) to reduce trivial non-offensive majority
    field_for_label = args.label_on
    df = df[df[field_for_label].astype(str).str.len() >= args.min_chars].reset_index(drop=True)

    # 4) Auto-label (thresholded, probability-based)
    labels = auto_label_series(
        df[field_for_label],
        model_name=args.autolabel_model,
        device_preference=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        positive_threshold=args.positive_threshold
    )
    df["label"] = labels
    print("Label distribution:", df["label"].value_counts(normalize=True).to_dict())

    # 5) Build train/val with the training text you actually used in the notebook (combined_text)
    train_field = args.train_text_field
    out_df = df[[train_field, "label"]].rename(columns={train_field: "text"}).copy()

    # Remove accidental empties
    out_df = out_df[(out_df["text"].astype(str).str.len() >= 3)]

    # Stratified split
    if out_df["label"].nunique() < 2:
        raise SystemExit("Auto-labeller produced a single class only; increase data or check the model.")
    train, val = train_test_split(out_df, test_size=args.test_size, random_state=args.seed, stratify=out_df["label"])

    train.to_csv(args.out_train, index=False)
    val.to_csv(args.out_val, index=False)

    # Minimal eval file (no identity flags yet)
    val.assign(id=range(1, len(val)+1)).to_csv(args.out_eval, index=False)

    print(f"Wrote {args.out_train} ({len(train)})")
    print(f"Wrote {args.out_val} ({len(val)})")
    print(f"Wrote {args.out_eval} ({len(val)})")

if __name__ == "__main__":
    main()
