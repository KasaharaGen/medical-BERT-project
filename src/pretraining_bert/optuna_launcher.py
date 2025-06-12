import subprocess

# 使用GPUのID一覧（GTX1080Ti ×2 なら [0, 1]）
gpu_ids = [0, 1 ,2 ,3 ,4]

# 各GPUごとにworker起動
processes = []
for gpu_id in gpu_ids:
    p = subprocess.Popen(["python", "optuna_worker.py", str(gpu_id)])
    processes.append(p)

# 全workerの終了を待機
for p in processes:
    p.wait()

print("✅ All GPU trials completed.")
