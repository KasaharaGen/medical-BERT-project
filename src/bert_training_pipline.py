import subprocess
import os

def run_step(name, command):
    print(f"\n=== Step: {name} ===")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"[ERROR] Failed at step: {name}")
    print(f"✅ Completed: {name}")

def main():
    # 環境前提確認
    required_files = [
        "train_tokenizer.py", "optuna_setup.py",
        "optuna_launcher.py", "optuna_worker.py",
        "pretrain_medical_bert.py", "data/merged_blocks_50%.csv"
    ]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"[ERROR] Required file not found: {f}")
    
    # ステップ実行
    run_step("Tokenizer Training", "python train_tokenizer.py")
    run_step("Optuna Study Initialization", "python optuna_setup.py")
    run_step("Parallel Hparam Search (GPU)", "python optuna_launcher.py")
    run_step("Final Pretraining with Best Params", "python pretrain_medical_bert.py")
    
    print("\n🎉 All pipeline steps completed successfully!")

if __name__ == "__main__":
    main()
