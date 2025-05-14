import os
import re
import sys
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

NER_MODEL = "d4data/biomedical-ner-all"
THRESHOLD = 0.5

TERMS = ["dengue", "covid19", "malaria","full", "sars coronavirus", "mars coronavirus"]
    
PATTERN = re.compile(r'\b(' + '|'.join(re.escape(term.lower()) for term in TERMS) + r')\b')

def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ").replace("\r", " ").strip().lower())

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[。．.!?])\s+', text) if s.strip()]

def contains_car_specific_term(text):
    return bool(PATTERN.search(text))

def extract_blocks(ner_pipeline, sentence_lists, threshold=0.5):
    flat_sentences, sentence_map = [], []
    for doc_idx, sentences in enumerate(sentence_lists):
        for i, sent in enumerate(sentences):
            flat_sentences.append(sent)
            sentence_map.append((doc_idx, i))

    ner_results = []
    for i in range(0, len(flat_sentences), 512):
        batch = flat_sentences[i:i+512]
        try:
            ner_results.extend(ner_pipeline(batch))
        except:
            ner_results.extend([[] for _ in batch])

    doc_blocks = [[] for _ in sentence_lists]
    for idx, ((doc_idx, i), ner_output) in enumerate(zip(sentence_map, ner_results)):
        sentence = flat_sentences[idx]
        has_entity = any(ent["score"] > threshold for ent in ner_output)
        has_keyword = contains_car_specific_term(sentence)

        if has_entity or has_keyword:
            prev = sentence_lists[doc_idx][i - 1] if i > 0 else ""
            next_ = sentence_lists[doc_idx][i + 1] if i + 1 < len(sentence_lists[doc_idx]) else ""
            block = " ".join([s for s in [prev, sentence, next_] if s])
            doc_blocks[doc_idx].append(block)

    return [list(set(blocks)) for blocks in doc_blocks]

def main():
    if len(sys.argv) < 3:
        print("Usage: python process_worker.py <input_csv_path> <gpu_id>")
        sys.exit(1)

    input_path = sys.argv[1]
    gpu_id = int(sys.argv[2])

    output_dir = "../data/processed"
    if not os.path.isdir(output_dir):
        raise FileNotFoundError("[ERROR] ../data/processed ディレクトリが存在しません")

    print(f"[INFO] {input_path} を GPU:{gpu_id} で処理中...")
    df = pd.read_csv(input_path)
    df["normalized_text"] = df["sentence"].apply(normalize_text)
    df["sentence_list"] = df["normalized_text"].apply(split_sentences)

    print("[INFO] モデル読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
    ner_pipeline_inst = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=gpu_id)

    print("[INFO] NER抽出中（前後文を含む）...")
    df["candidate_blocks"] = extract_blocks(ner_pipeline_inst, df["sentence_list"].tolist())

    flat_blocks = []
    for _, row in df.iterrows():
        base = row.to_dict()
        for block in row["candidate_blocks"]:
            record = base.copy()
            record["candidate_block"] = block
            flat_blocks.append(record)

    output_path = os.path.join(output_dir, os.path.basename(input_path).replace(".csv", "_blocks.csv"))
    pd.DataFrame(flat_blocks).to_csv(output_path, index=False)
    print(f"出力完了: {output_path} に保存（{len(flat_blocks)} ブロック）")

if __name__ == "__main__":
    main()