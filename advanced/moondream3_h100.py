#!/usr/bin/env python3
"""
Moondream 3 CLI — H100 版本
- 自動安裝相依套件
- terminal chat 介面
- detect / point 結果存 output/ 資料夾（序號命名）
"""

# ── 自動安裝套件 ──────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

print("📦 檢查並安裝相依套件...")
for pkg in ["torch", "torchvision", "accelerate", "einops", "Pillow"]:
    try:
        __import__(pkg.lower().replace("-", "_"))
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ⬇ 安裝 {pkg}...")
        install(pkg)
        print(f"   ✓ {pkg}")

# transformers 需要指定版本，太新的版本與 Moondream 3 不相容
print("   ⬇ 安裝 transformers==4.45.2...")
install("transformers==4.45.2")
print("   ✓ transformers==4.45.2")
print()

# ── 主要 imports ──────────────────────────────────────────────────
import os, sys, time
import torch
from PIL import Image, ImageDraw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16   # H100 原生支援 bfloat16，比 float16 更穩定

if DEVICE == "cuda":
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🖥  GPU: {gpu}  |  VRAM: {vram:.0f} GB")
else:
    print("⚠️  未偵測到 CUDA，將使用 CPU（速度較慢）")

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # 設定環境變數：export HF_TOKEN=hf_xxx

MODEL_ID = "vikhyatk/moondream2"
REVISION = None

if DEVICE == "cuda" and torch.cuda.get_device_properties(0).total_memory / 1e9 >= 20:
    MODEL_ID = "moondream/moondream3-preview"
    REVISION  = None
    print(f"✅ VRAM 充足 → 使用 {MODEL_ID}")
else:
    REVISION = "2025-01-09"
    print(f"✅ 使用 {MODEL_ID} (revision={REVISION})")

# ── 輸出資料夾 ────────────────────────────────────────────────────
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
_counter = [0]   # 用 list 讓 nested function 可修改

def next_path(label):
    _counter[0] += 1
    safe = label.replace(" ", "_").replace("'", "").replace("/", "-")[:30]
    return os.path.join(OUTPUT_DIR, f"{_counter[0]:03d}_{safe}.jpg")

# ── 標註函式 ──────────────────────────────────────────────────────
def draw_detect(image, objects, target):
    vis = image.copy()
    d = ImageDraw.Draw(vis)
    for o in objects:
        x0 = int(o["x_min"] * image.width);  y0 = int(o["y_min"] * image.height)
        x1 = int(o["x_max"] * image.width);  y1 = int(o["y_max"] * image.height)
        d.rectangle([x0, y0, x1, y1], outline="red", width=4)
        d.rectangle([x0, y0, x0 + len(target)*8 + 8, y0 + 18], fill="red")
        d.text((x0 + 4, y0 + 2), target, fill="white")
    return vis

def draw_point(image, points, target):
    vis = image.copy()
    d = ImageDraw.Draw(vis)
    for p in points:
        cx, cy = int(p["x"] * image.width), int(p["y"] * image.height)
        r = 12
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill="yellow", outline="black", width=3)
        d.text((cx + r + 4, cy - 8), target, fill="yellow")
    return vis

# ── 模型載入 ──────────────────────────────────────────────────────
_model  = None
_image  = None
_enc    = None
_img_path = None

def load_model():
    global _model
    if _model is None:
        from transformers import AutoModelForCausalLM
        print(f"\n🌙 載入模型 {MODEL_ID}（首次需要幾分鐘）...")
        t0 = time.time()
        kwargs = dict(trust_remote_code=True, torch_dtype=DTYPE)
        if REVISION:
            kwargs["revision"] = REVISION
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs).to(DEVICE)
        _model.eval()
        if DEVICE == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            print(f"   ✓ 載入完成 ({time.time()-t0:.1f}s)  VRAM 使用：{used:.1f} GB\n")
        else:
            print(f"   ✓ 載入完成 ({time.time()-t0:.1f}s)\n")
    return _model

def load_image(path):
    global _image, _enc, _img_path
    if not os.path.exists(path):
        print(f"❌ 找不到圖片：{path}")
        return False
    print(f"🔄 載入圖片：{path}")
    t0 = time.time()
    _image = Image.open(path).convert("RGB")
    _enc   = load_model().encode_image(_image)
    _img_path = path
    print(f"   ✓ ({_image.width}×{_image.height}px)  編碼：{time.time()-t0:.1f}s")
    return True

# ── 互動主迴圈 ────────────────────────────────────────────────────
HELP = """
指令：
  caption              — 圖像描述
  caption long         — 詳細描述
  query <問題>         — 視覺問答
  detect <物件>        — 物件偵測（存 output/）
  point <部位>         — 目標定位（存 output/）
  load <圖片路徑>      — 換圖片
  outdir               — 顯示輸出資料夾路徑
  quit                 — 離開
"""

def main():
    load_model()   # 先載入模型

    # 啟動時要求指定圖片
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("請輸入圖片路徑（或直接 Enter 跳過）：").strip()

    if path:
        load_image(path)
    else:
        print("（未載入圖片，請用 load <路徑> 載入）")

    print(HELP)
    abs_out = os.path.abspath(OUTPUT_DIR)
    print(f"📁 標註圖輸出位置：{abs_out}/\n")

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break

        if not line:
            continue

        if line in ("quit", "exit", "q"):
            print("再見！")
            break

        if line == "help":
            print(HELP)
            continue

        if line == "outdir":
            print(f"📁 {os.path.abspath(OUTPUT_DIR)}/")
            continue

        if line.startswith("load "):
            load_image(line[5:].strip())
            continue

        # 以下需要已載入圖片
        if _image is None:
            print("❌ 請先用 load <路徑> 載入圖片")
            continue

        model = load_model()

        if line in ("caption", "caption long"):
            length = "long" if "long" in line else "normal"
            t0 = time.time()
            cap = model.caption(_enc, length=length)["caption"]
            print(f"📝 {cap}  ({time.time()-t0:.1f}s)")

        elif line.startswith("query "):
            q = line[6:].strip()
            t0 = time.time()
            ans = model.query(_enc, q)["answer"]
            print(f"A: {ans}  ({time.time()-t0:.1f}s)")

        elif line.startswith("detect "):
            tgt = line[7:].strip()
            t0 = time.time()
            objs = model.detect(_enc, tgt)["objects"]
            print(f"🔍 找到 {len(objs)} 個「{tgt}」  ({time.time()-t0:.1f}s)")
            for i, o in enumerate(objs, 1):
                w = int((o["x_max"]-o["x_min"])*_image.width)
                h = int((o["y_max"]-o["y_min"])*_image.height)
                print(f"   [{i}] ({o['x_min']:.3f},{o['y_min']:.3f})→"
                      f"({o['x_max']:.3f},{o['y_max']:.3f})  {w}×{h}px")
            if objs:
                out = next_path(f"detect_{tgt}")
                draw_detect(_image, objs, tgt).save(out)
                print(f"   💾 {out}")

        elif line.startswith("point "):
            tgt = line[6:].strip()
            t0 = time.time()
            pts = model.point(_enc, tgt)["points"]
            print(f"📍 找到 {len(pts)} 個點「{tgt}」  ({time.time()-t0:.1f}s)")
            for i, p in enumerate(pts, 1):
                px, py = int(p["x"]*_image.width), int(p["y"]*_image.height)
                print(f"   [{i}] ({p['x']:.3f}, {p['y']:.3f}) → 像素({px}, {py})")
            if pts:
                out = next_path(f"point_{tgt}")
                draw_point(_image, pts, tgt).save(out)
                print(f"   💾 {out}")

        else:
            print("❓ 不認識的指令，輸入 help 查看說明")


if __name__ == "__main__":
    main()
