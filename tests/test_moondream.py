#!/usr/bin/env python3
"""
Moondream 本地測試腳本
測試：圖像描述、問答、物件偵測、目標定位
裝置：Apple M4 MPS
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw
import urllib.request
import os, time

# ── 裝置選擇 ──────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE != "cpu" else torch.float32
print(f"🖥  裝置：{DEVICE.upper()} | dtype: {DTYPE}")

# ── 下載測試圖片 ──────────────────────────────────────────────────
TEST_IMAGE_PATH = "examples/dog.jpg"
if not os.path.exists(TEST_IMAGE_PATH):
    print("📥 下載測試圖片...")
    url = "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r, open(TEST_IMAGE_PATH, "wb") as f:
            f.write(r.read())
        print(f"   ✓ {TEST_IMAGE_PATH}")
    except Exception:
        img = Image.new("RGB", (640, 480), (135, 180, 100))
        d = ImageDraw.Draw(img)
        d.ellipse([200,150,440,350], fill=(210,160,100))
        d.ellipse([260,100,320,150], fill=(210,160,100))
        d.ellipse([340,100,400,150], fill=(210,160,100))
        img.save(TEST_IMAGE_PATH)
        print(f"   ✓ 生成測試圖 {TEST_IMAGE_PATH}")

# ── 載入模型 ──────────────────────────────────────────────────────
MODEL_ID = "vikhyatk/moondream2"
REVISION  = "2025-01-09"
print(f"\n🌙 載入模型（首次執行會下載 ~3.7 GB，請稍候...）")
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=REVISION,
    trust_remote_code=True,
    torch_dtype=DTYPE,
).to(DEVICE).eval()

print(f"   ✓ 載入完成（{time.time()-t0:.1f} 秒）\n")

# ── 載入圖像 ──────────────────────────────────────────────────────
image = Image.open(TEST_IMAGE_PATH).convert("RGB")
print(f"📷 圖片尺寸：{image.size[0]}×{image.size[1]} px")
print("   （編碼圖像中，M4 MPS 首次較慢...）")
t0 = time.time()
encoded = model.encode_image(image)
print(f"   ✓ 圖像編碼（{time.time()-t0:.1f} 秒）\n")
print("=" * 60)

# ── 測試 1：圖像描述 ──────────────────────────────────────────────
print("【測試 1】圖像描述 (caption)")
t0 = time.time()
result = model.caption(encoded, length="normal")
caption = result["caption"]
print(f"📝 {caption}")
print(f"   ⏱ {time.time()-t0:.1f}s\n")

# ── 測試 2：視覺問答 ──────────────────────────────────────────────
print("【測試 2】視覺問答 (query)")
qs = [
    "What animal is in the image?",
    "What color is the dog?",
    "Is the dog indoors or outdoors?",
]
for q in qs:
    t0 = time.time()
    ans = model.query(encoded, q)["answer"]
    print(f"  Q: {q}")
    print(f"  A: {ans}  ({time.time()-t0:.1f}s)")

print()

# ── 測試 3：物件偵測 ──────────────────────────────────────────────
print("【測試 3】物件偵測 (detect)")
t0 = time.time()
objects = model.detect(encoded, "dog")["objects"]
print(f"🔍 偵測到 {len(objects)} 個 'dog'：")
for i, o in enumerate(objects):
    print(f"   [{i+1}] x_min={o['x_min']:.3f} y_min={o['y_min']:.3f} "
          f"x_max={o['x_max']:.3f} y_max={o['y_max']:.3f}")
print(f"   ⏱ {time.time()-t0:.1f}s\n")

# ── 測試 4：目標定位 ──────────────────────────────────────────────
print("【測試 4】目標定位 (point)")
t0 = time.time()
pts = model.point(encoded, "dog's nose")["points"]
if pts:
    for p in pts:
        print(f"📍 鼻子座標：x={p['x']:.3f} y={p['y']:.3f}  "
              f"（像素: {int(p['x']*image.width)}, {int(p['y']*image.height)}）")
else:
    print("   （未偵測到點位）")
print(f"   ⏱ {time.time()-t0:.1f}s\n")

print("=" * 60)
print("✅ 全部測試完成！Moondream 在你的 M4 MacBook Air 上運行正常。")
