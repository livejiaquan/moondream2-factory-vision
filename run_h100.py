#!/usr/bin/env python3
"""
在 H100 / NVIDIA GPU 上跑 Moondream 3 Preview
需要：pip install transformers einops accelerate Pillow pyvips
"""

import torch, time
from transformers import AutoModelForCausalLM
from PIL import Image

# ── 自動偵測最佳裝置 ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🖥  GPU: {gpu}  VRAM: {vram:.0f} GB")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🖥  Apple MPS")
else:
    DEVICE = "cpu"
    print("🖥  CPU")

# ── 根據 VRAM 自動選模型 ──────────────────────────────────────────
if DEVICE == "cuda":
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 20:
        # H100 / A100 / RTX 4090 — 跑最強 Moondream 3
        MODEL_ID = "moondream/moondream3-preview"
        REVISION  = None
        print(f"✅ VRAM {vram_gb:.0f}GB 充足 → 使用 Moondream 3 (MoE 9B)")
    elif vram_gb >= 4:
        # Jetson Orin Nano / RTX 3060 等
        MODEL_ID = "vikhyatk/moondream2"
        REVISION  = "2025-01-09"
        print(f"✅ VRAM {vram_gb:.0f}GB → 使用 Moondream 2 (2B)")
    else:
        MODEL_ID = "vikhyatk/moondream2"
        REVISION  = "2025-01-09"
        print(f"⚠️  VRAM 不足 20GB → 自動降級到 Moondream 2")
else:
    MODEL_ID  = "vikhyatk/moondream2"
    REVISION  = "2025-01-09"

DTYPE = torch.float16

# ── 載入模型 ──────────────────────────────────────────────────────
print(f"\n🌙 載入 {MODEL_ID}...")
t0 = time.time()
kwargs = {"trust_remote_code": True, "torch_dtype": DTYPE}
if REVISION:
    kwargs["revision"] = REVISION

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs).to(DEVICE).eval()
load_time = time.time() - t0
print(f"   ✓ 載入完成 ({load_time:.1f}s)")

if DEVICE == "cuda":
    used = torch.cuda.memory_allocated() / 1e9
    print(f"   VRAM 使用：{used:.1f} GB")

# ── 測試圖像 ──────────────────────────────────────────────────────
import urllib.request, os
img_path = "test_dog.jpg"
if not os.path.exists(img_path):
    url = "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"
    urllib.request.urlretrieve(url, img_path)

image = Image.open(img_path).convert("RGB")

# ── 推理測試 ──────────────────────────────────────────────────────
print("\n" + "="*55)
print("推理速度測試（3次取平均）")
print("="*55)

# warmup
enc = model.encode_image(image)

encode_times, query_times = [], []
for i in range(3):
    t0 = time.time()
    enc = model.encode_image(image)
    encode_times.append(time.time()-t0)

    t0 = time.time()
    ans = model.query(enc, "What animal is in the image?")["answer"]
    query_times.append(time.time()-t0)
    print(f"  Run {i+1}: encode={encode_times[-1]:.2f}s  query={query_times[-1]:.2f}s")

print(f"\n📊 平均結果：")
print(f"  圖像編碼：{sum(encode_times)/len(encode_times):.2f}s")
print(f"  問答推理：{sum(query_times)/len(query_times):.2f}s")
print(f"  答案：{ans}")

cap = model.caption(enc)["caption"]
print(f"\n📝 圖像描述：{cap[:120]}...")
print("\n✅ 完成！")
