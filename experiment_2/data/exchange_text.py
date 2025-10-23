# build_dengue_text_dataset_en_nameless.py
# -*- coding: utf-8 -*-
"""
Convert a binary-symptom CSV into English natural text + label dataset
without including the 'name' column in the output.
"""

import argparse
import hashlib
import random
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------
# English name lists
# --------------------------
FIRST_NAMES = [
    "John","Emily","Michael","Sarah","David","Laura","James","Olivia","Daniel","Sophia",
    "Robert","Emma","William","Ava","Joseph","Mia","Charles","Isabella","Thomas","Grace"
]
LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Miller","Davis","Garcia","Rodriguez","Wilson",
    "Martinez","Anderson","Taylor","Thomas","Hernandez","Moore","Martin","Jackson","Thompson","White"
]

# --------------------------
# English verbs for symptoms
# --------------------------
POS_VERBS = [
    "has", "shows", "complains of", "reports", "presents with", "experiences", "exhibits"
]
NEG_VERBS = [
    "denies", "does not have", "shows no sign of", "reports no", "has no", "lacks", "no evidence of"
]

SYMPTOM_NAME_MAP = {}  # optional mapping if you want to rename symptoms


# --------------------------
# Helper functions
# --------------------------
def stable_hash_int(s: str, algo: str = "md5") -> int:
    if algo == "md5":
        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    elif algo == "sha256":
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)
    raise ValueError("Unsupported hash algorithm.")


def pretty_symptom(col: str) -> str:
    if col in SYMPTOM_NAME_MAP:
        return SYMPTOM_NAME_MAP[col]
    return col.replace("_", " ").replace("-", " ")


def choose_pos_verb(symptom: str) -> str:
    h = stable_hash_int(symptom, "sha256")
    return POS_VERBS[h % len(POS_VERBS)]


def choose_neg_verb(symptom: str) -> str:
    h = stable_hash_int("neg-" + symptom, "sha256")
    return NEG_VERBS[h % len(NEG_VERBS)]


def make_random_name(index: int, raw_id) -> str:
    base = f"{index}-{raw_id}"
    h = stable_hash_int(base, "md5")
    first = FIRST_NAMES[h % len(FIRST_NAMES)]
    last = LAST_NAMES[(h // 101) % len(LAST_NAMES)]
    return f"{first} {last}"


def coerce_binary(v) -> int:
    try:
        iv = int(v)
        return 1 if iv != 0 else 0
    except Exception:
        return 0


def build_text(row, i, id_col, label_col, symptom_cols, max_neg: int) -> tuple:
    """Build one English sentence; exclude name column from output."""
    name = make_random_name(i, row[id_col])
    positives, negatives = [], []

    for col in symptom_cols:
        v = coerce_binary(row[col])
        sym = pretty_symptom(col)
        if v == 1:
            positives.append(f"{choose_pos_verb(sym)} {sym}")
        else:
            negatives.append(f"{choose_neg_verb(sym)} {sym}")

    if positives:
        main = f"{name} " + ", ".join(positives) + "."
    else:
        main = f"{name} has no specific symptoms."

    if negatives:
        k = min(max_neg, len(negatives))
        sampled = random.sample(negatives, k)
        neg = " However, " + ", ".join(sampled) + "."
    else:
        neg = ""

    text = main + neg
    label = coerce_binary(row[label_col])
    return text, label


def transform(input_csv: Path, output_csv: Path, seed: int = 42, max_neg: int = 5):
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(input_csv)
    df.columns = [str(c).strip() for c in df.columns]

    id_col = df.columns[0]
    label_col = "dengue"
    if label_col not in df.columns:
        candidates = [c for c in df.columns if c.lower().strip() in {"dengue","label","target"}]
        if not candidates:
            raise ValueError("No dengue column found.")
        label_col = candidates[0]

    symptom_cols = [c for c in df.columns if c not in {id_col, label_col}]
    if not symptom_cols:
        raise ValueError("No symptom columns detected.")

    texts, labels = [], []
    for i, row in df.iterrows():
        text, label = build_text(row, i, id_col, label_col, symptom_cols, max_neg)
        texts.append(text)
        labels.append(label)

    out_df = pd.DataFrame({"text": texts, "label": labels})
    out_df.to_csv(output_csv, index=False)
    print(f"[OK] Saved dataset: {output_csv} (rows={len(out_df)})")
    print(f"[Info] Symptom columns: {symptom_cols}")


def parse_args():
    p = argparse.ArgumentParser(description="Build an English dengue text dataset (no name column).")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_neg", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    transform(args.input, args.output, seed=args.seed, max_neg=args.max_neg)
