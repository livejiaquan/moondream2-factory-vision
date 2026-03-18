#!/usr/bin/env python3
"""
Moondream 3 Preview HPC 推理腳本
MoE 架構（9B total / 2B active）
需要 ~20GB VRAM — H100 80GB 完全足夠
"""

import argparse, json, os, time, torch
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser(description="Moondream 3 批次推理")
    p.add_argument("--image",  default="configs/test_images.txt")
    p.add_argument("--output", default="results/output_md3.json")
    p.add_argument("--questions", default="configs/questions.txt")
    return p.parse_args()

def load_paths(path):
    p = Path(path)
    if p.suffix in (".jpg",".jpeg",".png",".bmp",".webp"):
        return [str(p)]
    return [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

def load_questions(path):
    p = Path(path)
    if p.exists():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return ["What is in this image?", "Describe the scene in detail."]

def main():
    args = parse_args()
    os.makedirs(Path(args.output).parent, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1e9
        print(f"GPU: {props.name}  VRAM: {vram:.0f}GB")
        if vram < 20:
            print(f"⚠️  警告：VRAM {vram:.0f}GB 可能不足以跑 Moondream 3（需要 ~20GB）")
            print(f"   建議改用 run_moondream2.py")

    print("載入 Moondream 3 Preview（MoE 9B）...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device).eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"✓ 載入完成 ({time.time()-t0:.1f}s)  VRAM 使用：{used:.1f}GB")
    else:
        print(f"✓ 載入完成 ({time.time()-t0:.1f}s)")

    image_paths = load_paths(args.image)
    questions   = load_questions(args.questions)

    if not image_paths:
        print(f"⚠️  找不到圖片：{args.image}")
        return

    print(f"圖片數量：{len(image_paths)} | 問題數量：{len(questions)}")

    results = []
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {img_path}")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ❌ {e}"); continue

        t0 = time.time()
        enc = model.encode_image(image)
        encode_time = time.time() - t0

        t0 = time.time()
        caption = model.caption(enc, length="normal")["caption"]
        cap_time = time.time() - t0
        print(f"  caption ({cap_time:.1f}s): {caption[:100]}...")

        qa_results = []
        for q in questions:
            t0 = time.time()
            ans = model.query(enc, q)["answer"].strip()
            qa_results.append({"q": q, "a": ans, "t": round(time.time()-t0, 2)})
            print(f"  [{time.time()-t0:.1f}s] Q: {q[:50]} → A: {ans[:60]}")

        # Moondream 3 支援更多功能
        try:
            objects = model.detect(enc, "object")["objects"]
        except Exception:
            objects = []

        results.append({
            "image": img_path,
            "size": list(image.size),
            "encode_time": round(encode_time, 2),
            "caption": caption,
            "objects": objects,
            "qa": qa_results,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model": "moondream3-preview", "results": results}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n✅ 結果儲存：{args.output}  ({len(results)} 張圖片)")

if __name__ == "__main__":
    main()
