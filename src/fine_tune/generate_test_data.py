import pandas as pd
import spacy

# === spaCy 構文モデルの読み込み ===
nlp = spacy.load("en_core_web_sm")

# === データ読み込み ===
df = pd.read_csv("data/learning_df.csv")
SYMPTOMS = [col for col in df.columns if col.lower() not in ("id", "dengue")]

# === 表現辞書（構文・語彙の変換ルール）===
symptom_expression_map = {
    "fever":              ("suffers from", "a fever"),
    "headache":           ("suffers from", "a headache"),
    "muscle pain":        ("experiences", "muscle pain"),
    "joint pain":         ("complains of", "joint pain"),
    "rash":               ("has", "a skin rash"),
    "nausea":             ("feels", "nauseated"),
    "vomiting":           ("has", "vomiting episodes"),
    "eye pain":           ("suffers from", "eye pain"),
    "abdominal pain":     ("complains of", "abdominal pain"),
    "lymphadenopathy":    ("has", "swollen lymph nodes"),
    "chills":             ("feels", "chills"),
    "diarrhea":           ("has", "diarrhea"),
    "fatigue":            ("feels", "fatigued")
}

# === 自然文生成関数（文法構文処理付き）===
def row_to_structured_text(row, subject="He"):
    sents = []

    # ID文
    if pd.notnull(row.get("ID")):
        sents.append(f"{subject} is patient ID {int(row['ID'])}.")

    for symptom, (verb, phrase) in symptom_expression_map.items():
        val = row.get(symptom)
        if pd.isnull(val): continue
        if val == 1:
            sents.append(f"{subject} {verb} {phrase}.")
        else:
            # 否定文：feel → does not feel、has → does not have
            if verb in ["has", "feels"]:
                sents.append(f"{subject} does not {verb} {phrase}.")
            else:
                sents.append(f"{subject} does not {verb} {phrase}.")
    return sents

# === 高度構文チェック：主語重複文の削除・統合など ===
def revise_sentences(sentences):
    revised = []
    prev_root = None

    for sent in sentences:
        doc = nlp(sent)
        root = [token for token in doc if token.head == token][0]
        
        # 主語の繰り返しなどを避けたい場合
        if root.lemma_ == prev_root:
            continue
        prev_root = root.lemma_
        revised.append(sent)

    return " ".join(revised)

# === ラベル付与 ===
df["label"] = df["dengue"].apply(lambda x: 1 if x == 1 else 0)

# === 文生成 + 構文構造に基づく修正 ===
df["sent_list"] = df.apply(lambda row: row_to_structured_text(row, subject="He"), axis=1)
df["cleaned_text"] = df["sent_list"].apply(revise_sentences)

# === 出力保存 ===
df_out = df[["cleaned_text", "label"]]
df_out.to_csv("processed_classification_dataset_structured.csv", index=False)
print("✅ 構文構造を考慮したCSVを保存: processed_classification_dataset_structured.csv")
