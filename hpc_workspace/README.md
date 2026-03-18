# Moondream HPC Workspace

在 iService / SLURM 叢集上跑 Moondream 視覺語言模型。

## 目錄結構

```
moondream_workspace/
├── setup.sh                  ← 第一次執行，建立環境
├── requirements.txt          ← Python 依賴套件
├── configs/
│   ├── test_images.txt       ← 填入你的圖片路徑
│   └── questions.txt         ← 自訂問題清單
├── jobs/
│   ├── run_moondream2.sh     ← SLURM 提交腳本（2B 模型）
│   └── run_moondream3.sh     ← SLURM 提交腳本（3.0 旗艦版）
├── scripts/
│   ├── check_env.py          ← 環境診斷
│   ├── run_moondream2.py     ← Moondream 2 推理
│   └── run_moondream3.py     ← Moondream 3 推理
└── results/                  ← 推理結果輸出到這裡
```

## 使用步驟

### 1. 上傳到 iService
```bash
# 在你的 Mac 上執行
scp -r hpc_workspace/ 帳號@伺服器:~/moondream_workspace/
```

### 2. 登入後：第一次建立環境
```bash
cd ~/moondream_workspace
bash setup.sh
```

### 3. 確認環境正常
```bash
source .venv/bin/activate
python3 scripts/check_env.py
```

### 4. 填入圖片路徑
```bash
# 編輯 configs/test_images.txt，填入你的圖片路徑
nano configs/test_images.txt
```

### 5. 提交 SLURM 任務

**跑 Moondream 2（2B，推薦一般使用）：**
```bash
sbatch jobs/run_moondream2.sh
```

**跑 Moondream 3 旗艦版（需要 H100，~20GB VRAM）：**
```bash
sbatch jobs/run_moondream3.sh
```

### 6. 查看任務狀態
```bash
squeue -u $USER           # 查看排隊狀態
tail -f results/*.log     # 即時看 log
```

### 7. 結果在哪裡？
```
results/output_moondream2_<JobID>.json
results/output_moondream3_<JobID>.json
```

## 注意事項

- 如果學校 HPC 需要先 `module load python` 或 `module load cuda`，請在 `setup.sh` 最上方加上
- `jobs/*.sh` 裡的 `--partition=gpu` 可能需要改成你們學校的 partition 名稱
- 模型第一次跑會從 HuggingFace 下載（2B 約 3.7GB，3.0 約 18GB），之後會快取
