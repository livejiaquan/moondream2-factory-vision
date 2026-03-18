#!/usr/bin/env python3
"""
環境診斷腳本 — 登入節點後先跑這個確認環境正常
用法：python3 scripts/check_env.py
"""

import sys, importlib

print("=" * 50)
print("  Moondream HPC 環境診斷")
print("=" * 50)

# Python
print(f"\nPython: {sys.version.split()[0]}")

# 套件版本檢查
packages = {
    "torch":          "PyTorch",
    "transformers":   "Transformers",
    "PIL":            "Pillow",
    "einops":         "Einops",
    "accelerate":     "Accelerate",
}

all_ok = True
for mod, name in packages.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  ✅ {name:<15} {ver}")
    except ImportError:
        print(f"  ❌ {name:<15} 未安裝 → pip install {mod}")
        all_ok = False

# GPU 檢查
try:
    import torch
    print(f"\nCUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本：{torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            vram = p.total_memory / 1e9
            print(f"  GPU {i}: {p.name}  VRAM={vram:.0f}GB")
            if vram >= 20:
                print(f"         → ✅ 可跑 Moondream 3 (需要 ~20GB)")
            elif vram >= 4:
                print(f"         → ✅ 可跑 Moondream 2 (需要 ~4GB)")
            else:
                print(f"         → ⚠️  VRAM 不足，建議用 int4 量化")
    else:
        print("  ⚠️  沒有偵測到 GPU，將使用 CPU（速度較慢）")
except Exception as e:
    print(f"  ❌ GPU 檢查失敗：{e}")

print()
if all_ok:
    print("✅ 環境正常！可以執行 sbatch jobs/run_moondream2.sh")
else:
    print("⚠️  有套件未安裝，請先執行 bash setup.sh")
print("=" * 50)
