#!/usr/bin/env python3
"""
Moondream 2 HPC 推理腳本
支援批次處理多張圖片，結果儲存為 JSON
"""

import argparse, json, os, time, torch
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser(description="Moondream 2 批次推理")
    p.add_argument("--image",  default="configs/test_images.txt",
                   help="圖片路徑列表檔案（每行一個路徑），或單一圖片路徑")
    p.add_argument("--output", default="results/output.json",
                   help="結果輸出 JSON 路徑")
    p.add_argument("--questions", default="configs/questions.txt",
                   help="問題列表（每行一個），若無則只做 caption")
    return p.parse_args()

def load_image_paths(path: str) -> list:
    p = Path(path)
    if p.suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return [str(p)]
    if p.exists():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return []

def load_questions(path: str) -> list:
    p = Path(path)
    if p.exists():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return [
        "What is in this image?",
        "Describe the main subject.",
        "What colors are dominant?",
    ]

def main():
    args = parse_args()
    os.makedirs(Path(args.output).parent, exist_ok=True)

    # ── 裝置與模型載入 ──────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu}  VRAM: {vram:.0f}GB")

    print("載入 Moondream 2...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()
    print(f"✓ 載入完成 ({time.time()-t0:.1f}s)")

    # ── 載入輸入 ────────────────────────────────────────────────
    image_paths = load_image_paths(args.image)
    questions   = load_questions(args.questions)

    if not image_paths:
        print(f"⚠️  找不到圖片，請確認 {args.image} 是否存在")
        return

    print(f"圖片數量：{len(image_paths)}")
    print(f"問題數量：{len(questions)}")

    # ── 批次推理 ────────────────────────────────────────────────
    results = []
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {img_path}")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ❌ 無法開啟圖片：{e}")
            continue

        t0 = time.time()
        enc = model.encode_image(image)
        encode_time = time.time() - t0

        # caption
        t0 = time.time()
        caption = model.caption(enc, length="normal")["caption"]
        cap_time = time.time() - t0
        print(f"  caption ({cap_time:.1f}s): {caption[:80]}...")

        # 物件偵測
        objects = model.detect(enc, "object")["objects"]

        # 問答
        qa_results = []
        for q in questions:
            t0 = time.time()
            ans = model.query(enc, q)["answer"].strip()
            qa_results.append({"question": q, "answer": ans, "time": round(time.time()-t0, 2)})
            print(f"  Q: {q[:50]}")
            print(f"  A: {ans[:80]}")

        results.append({
            "image": img_path,
            "size": list(image.size),
            "encode_time": round(encode_time, 2),
            "caption": caption,
            "detected_objects": len(objects),
            "bboxes": objects,
            "qa": qa_results,
        })

    # ── 儲存結果 ────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model": "moondream2-2025-01-09", "results": results}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n✅ 結果已儲存：{args.output}")
    print(f"   處理了 {len(results)} 張圖片")

if __name__ == "__main__":
    main()
