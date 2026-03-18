#!/usr/bin/env python3
"""
Moondream 進階功能展示
測試：票據OCR、圖表理解、多問題問答
"""

import torch
from transformers import AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont
import time, textwrap

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE  = torch.float16

print("🌙 載入已快取的模型...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision="2025-01-09",
    trust_remote_code=True, torch_dtype=DTYPE,
).to(DEVICE).eval()
print(f"   ✓ {time.time()-t0:.1f}s\n")

# ── 建立含文字的假發票圖 ──────────────────────────────────────────
def make_receipt():
    img = Image.new("RGB", (400, 500), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([10,10,390,490], outline="black", width=2)
    lines = [
        ("SUPER MARKET", 30, 20, 22),
        ("Receipt #12345", 60, 50, 14),
        ("-"*35, 80, 80, 12),
        ("Apple x3        $4.50", 100, 100, 13),
        ("Milk  x2        $6.00", 120, 100, 13),
        ("Bread x1        $3.20", 140, 100, 13),
        ("Eggs  x1        $5.80", 160, 100, 13),
        ("-"*35, 180, 80, 12),
        ("Subtotal:      $19.50", 200, 100, 13),
        ("Tax (5%):       $0.98", 220, 100, 13),
        ("TOTAL:         $20.48", 250, 100, 15),
        ("Payment: Credit Card", 280, 100, 12),
        ("Thank you!", 320, 150, 16),
    ]
    for text, y, x, size in lines:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except:
            font = ImageFont.load_default()
        d.text((x, y), text, fill="black", font=font)
    img.save("examples/receipt.jpg")
    return img

# ── 建立假條形圖圖片 ──────────────────────────────────────────────
def make_chart():
    img = Image.new("RGB", (500, 350), "white")
    d = ImageDraw.Draw(img)
    # title
    try:
        font_b = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_s = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font_b = ImageFont.load_default()
        font_s = font_b
    d.text((100, 10), "Monthly Sales (Unit: $1000)", fill="black", font=font_b)
    # bars
    months = ["Jan","Feb","Mar","Apr","May","Jun"]
    values = [42, 58, 35, 71, 63, 85]
    colors = ["#4488FF","#00AACC","#4488FF","#FF6644","#4488FF","#22CC66"]
    base_y = 280
    for i, (m, v, c) in enumerate(zip(months, values, colors)):
        x0 = 60 + i*70
        h = v * 2
        d.rectangle([x0, base_y-h, x0+45, base_y], fill=c)
        d.text((x0+10, base_y-h-18), str(v), fill="black", font=font_s)
        d.text((x0+8,  base_y+5),  m, fill="black", font=font_s)
    d.line([50, 50, 50, base_y+2], fill="black", width=2)
    d.line([50, base_y, 470, base_y], fill="black", width=2)
    img.save("examples/chart.jpg")
    return img

# ─────────────────────────────────────────────────────────────────
def test(title, img, questions):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    t0 = time.time()
    enc = model.encode_image(img)
    print(f"  [編碼: {time.time()-t0:.1f}s]\n")

    cap = model.caption(enc)["caption"]
    print(f"  📝 自動描述：")
    for line in textwrap.wrap(cap, 55):
        print(f"     {line}")
    print()

    for q in questions:
        t0 = time.time()
        ans = model.query(enc, q)["answer"].strip()
        print(f"  Q: {q}")
        print(f"  A: {ans}  ({time.time()-t0:.1f}s)")
    print()

# ── 場景 A：超市發票（OCR + 數字理解） ───────────────────────────
receipt = make_receipt()
test("場景 A：超市發票 — 文字識別 + 數字理解", receipt, [
    "What is the total amount on this receipt?",
    "How many different items are listed?",
    "What payment method was used?",
    "What is the tax amount?",
])

# ── 場景 B：條形圖（圖表理解） ────────────────────────────────────
chart = make_chart()
test("場景 B：銷售圖表 — 圖表數據理解", chart, [
    "Which month had the highest sales?",
    "What is shown in this chart?",
    "Describe the overall sales trend.",
])

# ── 場景 C：已有的狗狗圖（展示 detect + point） ───────────────────
import os
if os.path.exists("examples/dog.jpg"):
    dog = Image.open("examples/dog.jpg").convert("RGB")
    print("=" * 60)
    print("  場景 C：狗狗圖 — detect + point 視覺化")
    print("=" * 60)
    enc = model.encode_image(dog)
    obj = model.detect(enc, "dog")["objects"]
    pts = model.point(enc, "dog's eye")["points"]
    print(f"  🔍 detect('dog')：{len(obj)} 個目標")
    for o in obj:
        w = int((o['x_max']-o['x_min']) * dog.width)
        h = int((o['y_max']-o['y_min']) * dog.height)
        print(f"     BBox 大小：{w}×{h} px")
    print(f"  📍 point('eye')：", end="")
    if pts:
        p = pts[0]
        print(f"x={p['x']:.2f}, y={p['y']:.2f}  → 像素({int(p['x']*dog.width)}, {int(p['y']*dog.height)})")
    # 畫出結果儲存
    vis = dog.copy().convert("RGB")
    d2 = ImageDraw.Draw(vis)
    for o in obj:
        x0,y0 = int(o['x_min']*dog.width), int(o['y_min']*dog.height)
        x1,y1 = int(o['x_max']*dog.width), int(o['y_max']*dog.height)
        d2.rectangle([x0,y0,x1,y1], outline="red", width=3)
    if pts:
        p = pts[0]
        cx,cy = int(p['x']*dog.width), int(p['y']*dog.height)
        d2.ellipse([cx-8,cy-8,cx+8,cy+8], fill="yellow", outline="black", width=2)
    vis.save("examples/dog_annotated.jpg")
    print(f"  💾 已儲存標註結果：examples/dog_annotated.jpg\n")

print("=" * 60)
print("✅ 進階測試完成！")
