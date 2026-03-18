#!/usr/bin/env python3
"""
Moondream CLI — 互動式視覺語言模型工具
用法見 README.md
"""

import argparse, sys, time, os
import torch
from PIL import Image, ImageDraw

# ── 裝置 ──────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32 if DEVICE == "cpu" else torch.float16

_model = None

def load_model():
    global _model
    if _model is None:
        from transformers import AutoModelForCausalLM
        print(f"🌙 載入模型（裝置：{DEVICE.upper()}）...")
        t0 = time.time()
        _model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", revision="2025-01-09",
            trust_remote_code=True, torch_dtype=DTYPE,
        ).to(DEVICE).eval()
        print(f"   ✓ 完成（{time.time()-t0:.1f}s）\n")
    return _model


def open_image(path):
    if not os.path.exists(path):
        sys.exit(f"❌ 找不到圖片：{path}")
    return Image.open(path).convert("RGB")


# ── 四個功能 ──────────────────────────────────────────────────────

def cmd_caption(args):
    model = load_model()
    image = open_image(args.image)
    enc   = model.encode_image(image)
    length = getattr(args, "length", "normal")
    t0 = time.time()
    cap = model.caption(enc, length=length)["caption"]
    print(f"📝 描述：{cap}")
    print(f"   ⏱ {time.time()-t0:.1f}s")


def cmd_query(args):
    if not args.question:
        sys.exit("❌ 請用 -q 指定問題，例如：-q \"What is in the image?\"")
    model = load_model()
    image = open_image(args.image)
    enc   = model.encode_image(image)
    t0 = time.time()
    ans = model.query(enc, args.question)["answer"]
    print(f"Q: {args.question}")
    print(f"A: {ans}")
    print(f"   ⏱ {time.time()-t0:.1f}s")


def annotate_detect(image, objects, target):
    vis = image.copy()
    d = ImageDraw.Draw(vis)
    for o in objects:
        x0 = int(o["x_min"] * image.width);  y0 = int(o["y_min"] * image.height)
        x1 = int(o["x_max"] * image.width);  y1 = int(o["y_max"] * image.height)
        d.rectangle([x0, y0, x1, y1], outline="red", width=3)
        d.text((x0 + 4, y0 + 2), target, fill="red")
    return vis


def annotate_point(image, points, target):
    vis = image.copy()
    d = ImageDraw.Draw(vis)
    for p in points:
        cx, cy = int(p["x"] * image.width), int(p["y"] * image.height)
        r = 10
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill="yellow", outline="black", width=2)
        d.text((cx + r + 2, cy - 8), target, fill="yellow")
    return vis


def cmd_detect(args):
    if not args.target:
        sys.exit("❌ 請用 -t 指定偵測目標，例如：-t \"person\"")
    model = load_model()
    image = open_image(args.image)
    enc   = model.encode_image(image)
    t0 = time.time()
    objects = model.detect(enc, args.target)["objects"]
    print(f"🔍 偵測「{args.target}」→ 找到 {len(objects)} 個目標（{time.time()-t0:.1f}s）")
    for i, o in enumerate(objects, 1):
        w = int((o["x_max"] - o["x_min"]) * image.width)
        h = int((o["y_max"] - o["y_min"]) * image.height)
        print(f"   [{i}] BBox: ({o['x_min']:.3f}, {o['y_min']:.3f}) → "
              f"({o['x_max']:.3f}, {o['y_max']:.3f})  大小: {w}×{h}px")
    if objects:
        vis = annotate_detect(image, objects, args.target)
        vis.show()
        if args.output:
            vis.save(args.output)
            print(f"   💾 已儲存：{args.output}")


def cmd_point(args):
    if not args.target:
        sys.exit("❌ 請用 -t 指定定位目標，例如：-t \"person's face\"")
    model = load_model()
    image = open_image(args.image)
    enc   = model.encode_image(image)
    t0 = time.time()
    points = model.point(enc, args.target)["points"]
    print(f"📍 定位「{args.target}」→ 找到 {len(points)} 個點（{time.time()-t0:.1f}s）")
    for i, p in enumerate(points, 1):
        px, py = int(p["x"] * image.width), int(p["y"] * image.height)
        print(f"   [{i}] 正規化: ({p['x']:.3f}, {p['y']:.3f})  像素: ({px}, {py})")
    if points:
        vis = annotate_point(image, points, args.target)
        vis.show()
        if args.output:
            vis.save(args.output)
            print(f"   💾 已儲存：{args.output}")


# ── 互動模式 ──────────────────────────────────────────────────────

def cmd_interactive(args):
    model = load_model()
    image_path = args.image
    image = open_image(image_path)
    enc   = model.encode_image(image)
    print(f"📷 已載入：{image_path}  ({image.width}×{image.height}px)")
    print("輸入指令（help 查看說明，quit 離開）\n")

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
            print("  caption              — 描述圖片")
            print("  caption long         — 詳細描述")
            print("  query <問題>         — 問答")
            print("  detect <物件>        — 偵測物件")
            print("  point <部位>         — 定位部位")
            print("  load <圖片路徑>      — 換圖片")
            continue
        if line.startswith("load "):
            path = line[5:].strip()
            image = open_image(path)
            print(f"🔄 重新編碼圖像：{path}")
            enc = model.encode_image(image)
            print(f"   ✓ ({image.width}×{image.height}px)")
            continue
        if line == "caption" or line == "caption long":
            length = "long" if "long" in line else "normal"
            cap = model.caption(enc, length=length)["caption"]
            print(f"📝 {cap}")
        elif line.startswith("query "):
            q = line[6:].strip()
            ans = model.query(enc, q)["answer"]
            print(f"A: {ans}")
        elif line.startswith("detect "):
            tgt = line[7:].strip()
            objs = model.detect(enc, tgt)["objects"]
            print(f"🔍 找到 {len(objs)} 個「{tgt}」")
            for i, o in enumerate(objs, 1):
                w = int((o["x_max"]-o["x_min"])*image.width)
                h = int((o["y_max"]-o["y_min"])*image.height)
                print(f"   [{i}] ({o['x_min']:.3f},{o['y_min']:.3f})→({o['x_max']:.3f},{o['y_max']:.3f}) {w}×{h}px")
            if objs:
                annotate_detect(image, objs, tgt).show()
        elif line.startswith("point "):
            tgt = line[6:].strip()
            pts = model.point(enc, tgt)["points"]
            print(f"📍 找到 {len(pts)} 個點「{tgt}」")
            for i, p in enumerate(pts, 1):
                print(f"   [{i}] ({p['x']:.3f}, {p['y']:.3f}) → 像素({int(p['x']*image.width)}, {int(p['y']*image.height)})")
            if pts:
                annotate_point(image, pts, tgt).show()
        else:
            print("❓ 不認識的指令，輸入 help 查看說明")


# ── 主程式 ────────────────────────────────────────────────────────

def main():
    # 共用參數（給每個子命令繼承）
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-i", "--image", default="test_dog.jpg", help="圖片路徑")
    common.add_argument("-q", "--question", help="問題（query 用）")
    common.add_argument("-t", "--target",   help="目標物件或部位（detect/point 用）")
    common.add_argument("-o", "--output",   help="標註結果儲存路徑")
    common.add_argument("--length", choices=["short", "normal", "long"], default="normal",
                        help="描述長度（caption 用，預設 normal）")

    parser = argparse.ArgumentParser(
        prog="moondream",
        description="Moondream 視覺語言模型 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""範例：
  python moondream.py caption -i photo.jpg
  python moondream.py query   -i photo.jpg -q "What color is the car?"
  python moondream.py detect  -i photo.jpg -t "person" -o result.jpg
  python moondream.py point   -i photo.jpg -t "person's eye" -o result.jpg
  python moondream.py chat    -i photo.jpg
"""
    )

    sub = parser.add_subparsers(dest="cmd", metavar="<功能>")
    sub.add_parser("caption", parents=[common], help="圖像描述")
    sub.add_parser("query",   parents=[common], help="視覺問答  -q \"問題\"")
    sub.add_parser("detect",  parents=[common], help="物件偵測  -t \"物件\"  [-o 輸出圖]")
    sub.add_parser("point",   parents=[common], help="目標定位  -t \"部位\"  [-o 輸出圖]")
    sub.add_parser("chat",    parents=[common], help="互動模式（可連續輸入指令）")

    args = parser.parse_args()

    dispatch = {
        "caption": cmd_caption,
        "query":   cmd_query,
        "detect":  cmd_detect,
        "point":   cmd_point,
        "chat":    cmd_interactive,
    }

    if args.cmd not in dispatch:
        parser.print_help()
        sys.exit(0)

    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
