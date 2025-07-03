import subprocess
import os
import sys

# 実行したいPythonスクリプトのリスト
scripts = [
    "data_run.py",
    "pretraining.py"
]

missing_files = [script for script in scripts if not os.path.exists(script)]
if missing_files:
    print("以下のスクリプトが見つかりません:")
    for file in missing_files:
        print(f" - {file}")
    print("全てのファイルが揃っていないため処理を終了します")
    sys.exit(1)  

print("すべてのスクリプトが存在します。順番に実行を開始します\n")


# エラー記録用リスト
errors = []

for script in scripts:
    print(f"\n=== {script} を実行中 ===")
    try:
        result = subprocess.run(["python", script], check=True, capture_output=True, text=True)
        print(f"{script} 正常終了")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ エラー: {script} の実行に失敗しました")
        print(f"終了コード: {e.returncode}")
        print(f"標準出力:\n{e.stdout}")
        print(f"標準エラー出力:\n{e.stderr}")
        errors.append({
            "script": script,
            "returncode": e.returncode,
            "stdout": e.stdout,
            "stderr": e.stderr
        })
    except FileNotFoundError as e:
        print(f"⚠️ エラー: {script} が見つかりません")
        errors.append({
            "script": script,
            "returncode": None,
            "stdout": "",
            "stderr": str(e)
        })

# ============================
# 実行後にエラー一覧をまとめて表示
# ============================
if errors:
    print("\n===== ⚠️ エラーが発生したスクリプト一覧 =====")
    for err in errors:
        print(f"\n--- {err['script']} ---")
        print(f"終了コード: {err['returncode']}")
        print(f"標準出力:\n{err['stdout']}")
        print(f"標準エラー出力:\n{err['stderr']}")
else:
    print("\n✅ すべてのスクリプトが正常に実行されました")
