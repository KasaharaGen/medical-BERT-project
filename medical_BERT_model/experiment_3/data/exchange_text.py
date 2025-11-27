import argparse
from pathlib import Path
import pandas as pd


def coerce_binary(v) -> int:
    try:
        iv = int(v)
        return 1 if iv != 0 else 0
    except Exception:
        return 0


def pretty_symptom(col: str) -> str:
    # Make column names human-readable: fever_high -> "fever high"
    return str(col).replace("_", " ").replace("-", " ").strip()


def build_text_for_row(row, name: str, id_col: str, label_col: str, symptom_cols: list[str]) -> tuple[str, int]:
    positives, negatives = [], []
    for col in symptom_cols:
        val = coerce_binary(row[col])
        sym = pretty_symptom(col)
        if val == 1:
            positives.append(sym)
        else:
            negatives.append(sym)

    segments = []
    if positives:
        segments.append(f"{name} has " + ", ".join(positives) + ".")
    else:
        segments.append(f"{name} doesn't have any listed symptoms.")

    if negatives:
        segments.append(f"{name} doesn't have " + ", ".join(negatives) + ".")

    text = " ".join(segments)
    label = coerce_binary(row[label_col])
    return text, label


def transform(input_csv: Path, output_csv: Path, unified_name: str = "Alex"):
    df = pd.read_csv(input_csv)
    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    # Assume first column is an ID-like field
    if len(df.columns) < 2:
        raise ValueError("Input CSV must have at least an ID column, a label column, and one symptom column.")

    id_col = df.columns[0]

    # Detect label column (prioritize 'dengue', fallback to common aliases)
    label_col = None
    if "dengue" in df.columns:
        label_col = "dengue"
    else:
        for cand in df.columns:
            if cand.lower() in {"label", "target"}:
                label_col = cand
                break
    if label_col is None:
        raise ValueError("Label column not found. Please include 'dengue' or 'label' or 'target'.")

    # Symptom columns = all except id and label
    symptom_cols = [c for c in df.columns if c not in {id_col, label_col}]
    if not symptom_cols:
        raise ValueError("No symptom columns detected. Ensure columns other than ID/label exist.")

    texts, labels = [], []
    for _, row in df.iterrows():
        text, label = build_text_for_row(row, unified_name, id_col, label_col, symptom_cols)
        texts.append(text)
        labels.append(label)

    out_df = pd.DataFrame({"text": texts, "label": labels})
    out_df.to_csv(output_csv, index=False)
    print(f"[OK] Saved: {output_csv}  rows={len(out_df)}")
    print(f"[Info] Name used for all rows: {unified_name}")
    print(f"[Info] Symptom columns ({len(symptom_cols)}): {symptom_cols}")


def parse_args():
    p = argparse.ArgumentParser(description="Refactor: simple text builder with unified name and has/doesn't have verbs.")
    p.add_argument("--input", type=Path, required=True, help="Path to input CSV (e.g., row_data.csv)")
    p.add_argument("--output", type=Path, required=True, help="Path to output CSV")
    p.add_argument("--name", type=str, default="Alex", help="Unified name used for all samples (default: Alex)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    transform(args.input, args.output, unified_name=args.name)
