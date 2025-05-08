import subprocess

steps = [
    #("📦 論文統合と文単位分割", ["python", "merge_fulltext.py"]),
    #("✂️ 文CSVのチャンク分割", ["python", "split_cunk_csv.py"]),
    #("🔍 NERブロック抽出 (マルチGPU)", ["python", "run_ner.py"]),
    ("🔤 トークナイザ学習 (SentencePiece)", ["python", "train_tokenizer.py"]),
    ("🧠 自作BERT MLM 事前学習", ["python", "bert_pretraining.py"])
]

def main():
    for label, cmd in steps:
        print(f"\n[STEP] {label}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ エラー: {label} に失敗しました\n{e}")
            return
    print("\n✅ パイプラインすべて完了")

if __name__ == "__main__":
    main()
