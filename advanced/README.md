# Advanced — Moondream 3 / H100

此資料夾為 Moondream 3 Preview（9B MoE）在 NVIDIA H100 上的實驗性腳本，非主要工具。

| 檔案 | 說明 |
|------|------|
| `moondream3_h100.py` | H100 互動 CLI，結果存 `output/` |
| `run_h100.py` | HPC 批量執行腳本 |

使用前設定 HF Token（Moondream 3 為 Gated Model）：

```bash
export HF_TOKEN=hf_你的token
python advanced/moondream3_h100.py 圖片.jpg
```

硬體需求：VRAM ≥ 20GB（H100 / A100 / RTX 4090）
