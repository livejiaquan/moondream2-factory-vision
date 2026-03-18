#!/usr/bin/env python3
"""
Moondream 批量處理腳本 — H100 版本
用法：
  python batch.py detect  -d 圖片資料夾 -t "person" [-o 輸出資料夾]
  python batch.py point   -d 圖片資料夾 -t "person's face" [-o 輸出資料夾]
  python batch.py caption -d 圖片資料夾 [-o 輸出資料夾]
  python batch.py query   -d 圖片資料夾 -q "問題" [-o 輸出資料夾]
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

install("transformers==4.45.2")
print("   ✓ transformers==4.45.2")
print()

# ── Imports ───────────────────────────────────────────────────────
import os, sys, time, argparse, csv
import torch
from PIL import Image, ImageDraw

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # 設定環境變數：export HF_TOKEN=hf_xxx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

MODELS = {
    "2": ("vikhyatk/moondream2", "2024-07-23"),
    "3": ("moondream/moondream3-preview", None),
}

# ── 模型載入 ──────────────────────────────────────────────────────
def load_model(model_ver="3"):
    from transformers import AutoModelForCausalLM

    if DEVICE == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if model_ver == "3" and vram < 20:
            print(f"⚠️  VRAM {vram:.0f}GB 不足，自動降為 Moondream 2")
            model_ver = "2"
        print(f"🖥  {torch.cuda.get_device_name(0)}  VRAM: {vram:.0f}GB")

    model_id, rev = MODELS[model_ver]
    print(f"🌙 載入 Moondream {model_ver}（{model_id}）...")
    t0 = time.time()
    kwargs = dict(trust_remote_code=True, torch_dtype=DTYPE)
    if rev:
        kwargs["revision"] = rev
    if HF_TOKEN and len(HF_TOKEN) > 4:
        kwargs["token"] = HF_TOKEN
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(DEVICE).eval()
    print(f"   ✓ 完成（{time.time()-t0:.1f}s）\n")
    return model


def unload_model(model):
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

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

# ── 取得資料夾內所有圖片 ──────────────────────────────────────────
def get_images(folder):
    paths = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
            paths.append(os.path.join(folder, f))
    return paths

# ── 批量處理主函式 ────────────────────────────────────────────────
def batch_detect(args):
    images = get_images(args.dir)
    if not images:
        sys.exit(f"❌ 資料夾內找不到圖片：{args.dir}")

    os.makedirs(args.output, exist_ok=True)
    model = load_model(args.model if args.model != "both" else "3")

    found_count = 0
    print(f"🔍 detect「{args.target}」— 共 {len(images)} 張圖\n")

    for i, path in enumerate(images, 1):
        name = os.path.basename(path)
        try:
            image = Image.open(path).convert("RGB")
            enc   = model.encode_image(image)
            t0    = time.time()
            objs  = model.detect(enc, args.target)["objects"]
            elapsed = time.time() - t0

            if objs:
                found_count += 1
                out_name = os.path.splitext(name)[0] + "_detect.jpg"
                out_path = os.path.join(args.output, out_name)
                draw_detect(image, objs, args.target).save(out_path)
                print(f"[{i:3d}/{len(images)}] ✅ {name}  → {len(objs)} 個目標  💾 {out_name}  ({elapsed:.1f}s)")
            else:
                print(f"[{i:3d}/{len(images)}] ——  {name}  → 未偵測到  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"[{i:3d}/{len(images)}] ❌ {name}  → 錯誤：{e}")

    print(f"\n✅ 完成！{len(images)} 張中有 {found_count} 張偵測到「{args.target}」")
    print(f"   輸出：{os.path.abspath(args.output)}/")


def batch_point(args):
    images = get_images(args.dir)
    if not images:
        sys.exit(f"❌ 資料夾內找不到圖片：{args.dir}")

    os.makedirs(args.output, exist_ok=True)
    model = load_model(args.model if args.model != "both" else "3")

    found_count = 0
    print(f"📍 point「{args.target}」— 共 {len(images)} 張圖\n")

    for i, path in enumerate(images, 1):
        name = os.path.basename(path)
        try:
            image = Image.open(path).convert("RGB")
            enc   = model.encode_image(image)
            t0    = time.time()
            pts   = model.point(enc, args.target)["points"]
            elapsed = time.time() - t0

            if pts:
                found_count += 1
                out_name = os.path.splitext(name)[0] + "_point.jpg"
                out_path = os.path.join(args.output, out_name)
                draw_point(image, pts, args.target).save(out_path)
                print(f"[{i:3d}/{len(images)}] ✅ {name}  → {len(pts)} 個點  💾 {out_name}  ({elapsed:.1f}s)")
            else:
                print(f"[{i:3d}/{len(images)}] ——  {name}  → 未偵測到  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"[{i:3d}/{len(images)}] ❌ {name}  → 錯誤：{e}")

    print(f"\n✅ 完成！{len(images)} 張中有 {found_count} 張找到「{args.target}」")
    print(f"   輸出：{os.path.abspath(args.output)}/")


def _run_caption_single(model, model_ver, images, output, length):
    csv_path = os.path.join(output, f"captions_md{model_ver}.csv")
    print(f"📝 Moondream {model_ver} caption — 共 {len(images)} 張  (length={length})\n")
    results = {}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", f"caption_md{model_ver}"])
        for i, path in enumerate(images, 1):
            name = os.path.basename(path)
            try:
                image = Image.open(path).convert("RGB")
                enc   = model.encode_image(image)
                t0    = time.time()
                cap   = model.caption(enc, length=length)["caption"]
                elapsed = time.time() - t0
                writer.writerow([name, cap])
                f.flush()
                results[name] = cap
                short = cap[:80] + "..." if len(cap) > 80 else cap
                print(f"[{i:3d}/{len(images)}] {name}")
                print(f"          {short}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"[{i:3d}/{len(images)}] ❌ {name}  → 錯誤：{e}")
                results[name] = f"ERROR: {e}"
    print(f"\n   💾 {csv_path}\n")
    return results


def batch_caption(args):
    images = get_images(args.dir)
    if not images:
        sys.exit(f"❌ 資料夾內找不到圖片：{args.dir}")

    os.makedirs(args.output, exist_ok=True)
    length = getattr(args, "length", "normal")
    versions = ["2", "3"] if args.model == "both" else [args.model]
    all_results = {}

    for ver in versions:
        model = load_model(ver)
        all_results[ver] = _run_caption_single(model, ver, images, args.output, length)
        unload_model(model)

    # 若兩個模型都跑，輸出對比 CSV
    if args.model == "both":
        compare_path = os.path.join(args.output, "captions_compare.csv")
        with open(compare_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "caption_md2", "caption_md3"])
            for path in images:
                name = os.path.basename(path)
                writer.writerow([name,
                                  all_results["2"].get(name, ""),
                                  all_results["3"].get(name, "")])
        print(f"📊 對比結果：{compare_path}")

    print(f"\n✅ 完成！輸出：{os.path.abspath(args.output)}/")


def batch_query(args):
    images = get_images(args.dir)
    if not images:
        sys.exit(f"❌ 資料夾內找不到圖片：{args.dir}")

    os.makedirs(args.output, exist_ok=True)
    model = load_model(args.model if args.model != "both" else "3")

    csv_path = os.path.join(args.output, "query_results.csv")
    print(f"❓ query「{args.question}」— 共 {len(images)} 張圖\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "question", "answer"])

        for i, path in enumerate(images, 1):
            name = os.path.basename(path)
            try:
                image = Image.open(path).convert("RGB")
                enc   = model.encode_image(image)
                t0    = time.time()
                ans   = model.query(enc, args.question)["answer"]
                elapsed = time.time() - t0
                writer.writerow([name, args.question, ans])
                f.flush()
                print(f"[{i:3d}/{len(images)}] {name}  → {ans}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"[{i:3d}/{len(images)}] ❌ {name}  → 錯誤：{e}")

    print(f"\n✅ 完成！結果已存：{csv_path}")


# ── 主程式 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="batch",
        description="Moondream 批量處理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""範例：
  python batch.py detect  -d usig/ -t "person" -o output/
  python batch.py point   -d usig/ -t "person's face" -o output/
  python batch.py caption -d usig/ -o output/
  python batch.py query   -d usig/ -q "Is there a person?" -o output/
"""
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-d", "--dir",    required=True, help="圖片資料夾路徑")
    common.add_argument("-o", "--output", default="output_batch", help="輸出資料夾（預設 output_batch/）")
    common.add_argument("-t", "--target",   help="偵測目標（detect/point 用）")
    common.add_argument("-q", "--question", help="問題（query 用）")
    common.add_argument("--length", choices=["short", "normal", "long"], default="normal",
                        help="描述長度（caption 用）")
    common.add_argument("--model", choices=["2", "3", "both"], default="3",
                        help="使用哪個模型：2=Moondream2, 3=Moondream3, both=兩個都跑（預設 3）")

    sub = parser.add_subparsers(dest="cmd", metavar="<功能>")
    sub.add_parser("detect",  parents=[common], help="批量物件偵測  -t \"物件\"")
    sub.add_parser("point",   parents=[common], help="批量目標定位  -t \"部位\"")
    sub.add_parser("caption", parents=[common], help="批量圖像描述，輸出 captions.csv")
    sub.add_parser("query",   parents=[common], help="批量問答     -q \"問題\"，輸出 query_results.csv")

    args = parser.parse_args()

    if args.cmd == "detect":
        if not args.target:
            sys.exit("❌ detect 需要 -t 指定偵測目標")
        batch_detect(args)
    elif args.cmd == "point":
        if not args.target:
            sys.exit("❌ point 需要 -t 指定定位目標")
        batch_point(args)
    elif args.cmd == "caption":
        batch_caption(args)
    elif args.cmd == "query":
        if not args.question:
            sys.exit("❌ query 需要 -q 指定問題")
        batch_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
