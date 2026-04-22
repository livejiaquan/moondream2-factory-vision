#!/usr/bin/env python3
"""
Factory safety feasibility test — 針對 data/ 影片做安全檢測測試。
輸出至 outputs/<timestamp>/
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM

# ── 設定 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs"

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-01-09"
SAMPLE_EVERY_SEC = 30  # 每幾秒抽一幀

DETECT_TARGETS = ["person", "helmet", "wheel chock"]

QUERY_CHECKS = [
    {
        "name": "worker_visible",
        "prompt": "Is there any visible worker or person in this frame? Answer yes or no first, then explain briefly.",
    },
    {
        "name": "wheel_chock_present",
        "prompt": "Is there a wheel chock or tire stopper visible near a vehicle wheel? Answer yes or no first, then explain briefly.",
    },
    {
        "name": "missing_helmet",
        "prompt": "Is any visible person not wearing a safety helmet or hard hat? Answer yes or no first, then explain briefly.",
    },
    {
        "name": "face_mask_visible",
        "prompt": "Is any visible worker wearing a face mask or respirator mask? Answer yes or no first, then explain briefly.",
    },
    {
        "name": "abnormal_behavior",
        "prompt": "Does this frame show any unsafe, abnormal, or suspicious worker behavior? Answer yes or no first, then explain briefly.",
    },
]

# ── 顏色（每個 detect target 一個顏色）────────────────────────────────
COLORS = {
    "person": "#FF4444",
    "helmet": "#44FF44",
    "wheel chock": "#FFD700",
}
DEFAULT_COLOR = "#FF8800"


# ── 工具函式 ──────────────────────────────────────────────────────────

def setup_device():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_model(device, dtype):
    print(f"[model] 載入 {MODEL_ID} @ {REVISION}  device={device}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()
    print(f"[model] 完成 ({time.time() - t0:.1f}s)\n")
    return model


def classify_yes_no(answer: str) -> str:
    text = answer.strip().lower()
    if re.match(r"^(yes\b|yes[,.!\s])", text):
        return "yes"
    if re.match(r"^(no\b|no[,.!\s])", text):
        return "no"
    return "unknown"


def sample_frames(video_path: Path, every_sec: float):
    """從影片每 every_sec 秒抽一幀，yield (pil_image, time_sec)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] 無法開啟影片: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_total / fps) if fps > 0 else 0
    step_ms = max(every_sec, 0.5) * 1000.0
    pos_ms = 0.0

    while pos_ms <= duration * 1000.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield Image.fromarray(rgb), round(pos_ms / 1000.0, 1)
        pos_ms += step_ms

    cap.release()


def analyze(model, pil_img):
    enc = model.encode_image(pil_img)

    result = {
        "caption": model.caption(enc, length="normal")["caption"],
        "queries": {},
        "detections": {},
    }

    for qc in QUERY_CHECKS:
        ans = model.query(enc, qc["prompt"])["answer"]
        result["queries"][qc["name"]] = {
            "answer": ans,
            "label": classify_yes_no(ans),
        }

    for tgt in DETECT_TARGETS:
        try:
            objects = model.detect(enc, tgt)["objects"]
        except Exception as e:
            objects = []
            print(f"    [warn] detect '{tgt}' 失敗: {e}")
        result["detections"][tgt] = objects

    return result


def draw_annotations(pil_img, result):
    vis = pil_img.copy()
    draw = ImageDraw.Draw(vis)

    for tgt, objects in result["detections"].items():
        color = COLORS.get(tgt, DEFAULT_COLOR)
        for obj in objects:
            if "x_min" not in obj:
                continue
            x0 = int(obj["x_min"] * vis.width)
            y0 = int(obj["y_min"] * vis.height)
            x1 = int(obj["x_max"] * vis.width)
            y1 = int(obj["y_max"] * vis.height)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            draw.text((x0 + 4, y0 + 4), tgt, fill=color)

    # 右上角顯示 query 結果
    labels = []
    for qc in QUERY_CHECKS:
        lbl = result["queries"].get(qc["name"], {}).get("label", "?")
        icon = {"yes": "Y", "no": "N", "unknown": "?"}.get(lbl, "?")
        labels.append(f"{icon} {qc['name']}")

    y_pos = 6
    for line in labels:
        draw.text((6, y_pos), line, fill="white")
        y_pos += 18

    return vis


def flatten_row(video_name, time_sec, result):
    row = {
        "video": video_name,
        "time_sec": time_sec,
        "caption": result["caption"],
    }
    for qc in QUERY_CHECKS:
        q = result["queries"].get(qc["name"], {})
        row[f"{qc['name']}_label"] = q.get("label", "")
        row[f"{qc['name']}_answer"] = q.get("answer", "")
    for tgt in DETECT_TARGETS:
        objs = result["detections"].get(tgt, [])
        row[f"detect_{tgt.replace(' ', '_')}_count"] = sum(
            1 for o in objs if "x_min" in o
        )
    return row


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--every", type=float, default=SAMPLE_EVERY_SEC)
    parser.add_argument("--limit", type=int, default=0, help="每部影片最多幾幀，0=不限")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_root = Path(args.output)
    videos = sorted(data_path.glob("*.mkv")) + sorted(data_path.glob("*.mp4"))

    if not videos:
        print(f"找不到影片: {data_path}")
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    annotated_dir = run_dir / "annotated"
    run_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True)

    device, dtype = setup_device()
    model = load_model(device, dtype)

    all_rows = []
    jsonl_path = run_dir / "results.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for video in videos:
            print(f"=== {video.name} ===")
            frame_count = 0
            for pil_img, t_sec in sample_frames(video, args.every):
                if args.limit and frame_count >= args.limit:
                    break

                t0 = time.time()
                result = analyze(model, pil_img)
                elapsed = round(time.time() - t0, 2)

                # 印進度
                cap_short = result["caption"][:80]
                q_summary = "  ".join(
                    f"{q}={result['queries'].get(q, {}).get('label','?')}"
                    for q in ["worker_visible", "wheel_chock_present", "missing_helmet"]
                )
                print(f"  [{t_sec}s] {elapsed:.1f}s | {q_summary}")
                print(f"    caption: {cap_short}")

                # 存 JSONL
                entry = {
                    "video": video.name,
                    "time_sec": t_sec,
                    "runtime_sec": elapsed,
                    "result": result,
                }
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")

                # 存標註圖
                vis = draw_annotations(pil_img, result)
                fname = f"{video.stem}_{int(t_sec):05d}s.jpg"
                vis.save(annotated_dir / fname, quality=85)

                all_rows.append(flatten_row(video.name, t_sec, result))
                frame_count += 1

            print(f"  -> {frame_count} frames processed\n")

    # 存 CSV
    csv_path = run_dir / "summary.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    # 存 meta
    meta = {
        "run_id": run_id,
        "model": MODEL_ID,
        "revision": REVISION,
        "device": device,
        "sample_every_sec": args.every,
        "total_frames": len(all_rows),
        "videos": [v.name for v in videos],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    print(f"\n完成！輸出位置: {run_dir}")
    print(f"  results.jsonl  ({len(all_rows)} frames)")
    print(f"  summary.csv")
    print(f"  annotated/  ({len(all_rows)} 張標註圖)")

    # 印簡單統計
    print("\n=== 統計 ===")
    for qc in QUERY_CHECKS:
        yes = sum(1 for r in all_rows if r.get(f"{qc['name']}_label") == "yes")
        total = len(all_rows)
        print(f"  {qc['name']:30s}  yes={yes}/{total}")
    for tgt in DETECT_TARGETS:
        key = f"detect_{tgt.replace(' ', '_')}_count"
        detected = sum(1 for r in all_rows if r.get(key, 0) > 0)
        print(f"  detect {tgt:24s}  有框={detected}/{len(all_rows)}")


if __name__ == "__main__":
    main()
