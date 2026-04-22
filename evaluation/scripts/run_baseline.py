#!/usr/bin/env python3
"""
負樣本基線測試：對 data/ 影片密集抽幀，跑所有 YOLO Gap 場景的 query/detect/caption，
輸出 JSONL + CSV + 標註圖，用於統計 False Positive 率。

用法：
  cd evaluation
  python scripts/run_baseline.py
  python scripts/run_baseline.py --data ../data --every 5 --annotate
  python scripts/run_baseline.py --config configs/gap_baseline.json --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

EVAL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EVAL_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Moondream YOLO-Gap 負樣本基線測試",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data"),
        help="影片/圖片資料夾（預設 ../data/）",
    )
    parser.add_argument(
        "--config",
        default=str(EVAL_DIR / "configs" / "gap_baseline.json"),
        help="測試配置 JSON",
    )
    parser.add_argument(
        "--output",
        default=str(EVAL_DIR / "outputs"),
        help="輸出根目錄",
    )
    parser.add_argument(
        "--every",
        type=float,
        default=0,
        help="抽幀間隔秒數（0 = 使用 config 中的預設值）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多處理幾幀（0 = 不限）",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="儲存標註圖",
    )
    parser.add_argument(
        "--model-id",
        default="vikhyatk/moondream2",
    )
    parser.add_argument(
        "--revision",
        default="2025-01-09",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── 裝置與模型 ────────────────────────────────────────────────────

def choose_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_model(model_id: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM

    device, dtype = choose_device()
    print(f"[model] 載入 {model_id} @ {revision}  device={device}")
    t0 = time.time()
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()
    elapsed = round(time.time() - t0, 2)
    print(f"[model] 完成（{elapsed}s）\n")
    return model, device, elapsed


# ── 影片抽幀 ──────────────────────────────────────────────────────

def sample_frames(video_path: Path, every_sec: float):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] 無法開啟: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total / fps) if fps > 0 else 0
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


def collect_sources(data_path: Path):
    if data_path.is_file():
        return [data_path]
    return sorted(
        p for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS
    )


# ── 推理 ──────────────────────────────────────────────────────────

def classify_yes_no(answer: str) -> str:
    text = answer.strip().lower()
    if re.match(r"^(yes\b|yes[,.!\s])", text):
        return "yes"
    if re.match(r"^(no\b|no[,.!\s])", text):
        return "no"
    if re.match(r"^(unclear\b|unclear[,.!\s])", text):
        return "unclear"
    return "unknown"


def run_inference(model, pil_image, config: dict) -> dict:
    enc = model.encode_image(pil_image)
    caption_len = config.get("caption_length", "normal")

    result = {
        "caption": model.caption(enc, length=caption_len)["caption"],
        "queries": {},
        "detections": {},
    }

    for qc in config.get("query_checks", []):
        answer = model.query(enc, qc["prompt"])["answer"]
        result["queries"][qc["name"]] = {
            "prompt": qc["prompt"],
            "answer": answer,
            "label": classify_yes_no(answer),
            "expected": qc.get("expected_in_normal", ""),
            "category": qc.get("category", ""),
        }

    for dt in config.get("detect_targets", []):
        target = dt["name"]
        try:
            objects = model.detect(enc, target)["objects"]
        except Exception as e:
            objects = [{"error": str(e)}]
        result["detections"][target] = {
            "objects": objects,
            "count": sum(1 for o in objects if "x_min" in o),
            "category": dt.get("category", ""),
        }

    return result


# ── 標註 ──────────────────────────────────────────────────────────

DETECT_COLORS = {
    "person": "#FF4444",
    "helmet": "#44FF44",
    "wheel chock": "#FFD700",
    "cigarette": "#FF00FF",
    "fire": "#FF6600",
    "fallen person": "#00FFFF",
}


def draw_annotations(pil_img, result: dict):
    from PIL import ImageDraw

    vis = pil_img.copy()
    draw = ImageDraw.Draw(vis)

    for target, det in result.get("detections", {}).items():
        color = DETECT_COLORS.get(target, "#FF8800")
        for obj in det.get("objects", []):
            if "x_min" not in obj:
                continue
            x0 = int(obj["x_min"] * vis.width)
            y0 = int(obj["y_min"] * vis.height)
            x1 = int(obj["x_max"] * vis.width)
            y1 = int(obj["y_max"] * vis.height)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            draw.text((x0 + 4, y0 + 2), target, fill=color)

    y_pos = 6
    for name, q in result.get("queries", {}).items():
        lbl = q.get("label", "?")
        icon = {"yes": "Y", "no": "N"}.get(lbl, "?")
        cat = q.get("category", "")
        marker = " *FP*" if q.get("expected") == "no" and lbl == "yes" else ""
        draw.text((6, y_pos), f"{icon} {name}{marker}", fill="white")
        y_pos += 18

    return vis


# ── CSV 扁平化 ────────────────────────────────────────────────────

def flatten_row(video_name: str, time_sec: float, result: dict, config: dict) -> dict:
    row = {
        "video": video_name,
        "time_sec": time_sec,
        "caption": result["caption"],
    }

    for qc in config.get("query_checks", []):
        q = result["queries"].get(qc["name"], {})
        row[f"q_{qc['name']}_label"] = q.get("label", "")
        row[f"q_{qc['name']}_answer"] = q.get("answer", "")

    for dt in config.get("detect_targets", []):
        det = result["detections"].get(dt["name"], {})
        row[f"d_{dt['name'].replace(' ', '_')}_count"] = det.get("count", 0)

    alert_keywords = config.get("caption_alert_keywords", [])
    caption_lower = result["caption"].lower()
    hits = [kw for kw in alert_keywords if kw in caption_lower]
    row["caption_alert_keywords"] = "; ".join(hits) if hits else ""

    return row


# ── 主流程 ────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    data_path = Path(args.data).resolve()
    config_path = Path(args.config).resolve()
    output_root = Path(args.output).resolve()

    if not data_path.exists():
        print(f"[error] 找不到資料路徑: {data_path}", file=sys.stderr)
        return 1
    if not config_path.exists():
        print(f"[error] 找不到配置檔: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    sources = collect_sources(data_path)
    if not sources:
        print(f"[error] 找不到支援的檔案: {data_path}", file=sys.stderr)
        return 1

    every_sec = args.every if args.every > 0 else config.get("sampling", {}).get("default_every_sec", 5)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    annotated_dir = run_dir / "annotated"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.annotate:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 負樣本基線測試 ===")
    print(f"  資料: {data_path}")
    print(f"  配置: {config_path}")
    print(f"  抽幀間隔: {every_sec}s")
    print(f"  輸出: {run_dir}")
    print(f"  影片/圖片數: {len(sources)}")
    print()

    model, device, load_sec = load_model(args.model_id, args.revision)

    rows: list[dict] = []
    jsonl_path = run_dir / "results.jsonl"
    processed = 0
    t_start = time.time()

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for source in sources:
            suffix = source.suffix.lower()
            print(f"--- {source.name} ---")

            if suffix in VIDEO_EXTS:
                frames = sample_frames(source, every_sec)
            elif suffix in IMAGE_EXTS:
                from PIL import Image
                frames = [(Image.open(source).convert("RGB"), 0.0)]
            else:
                continue

            frame_count = 0
            for pil_img, t_sec in frames:
                if args.limit and processed >= args.limit:
                    break

                t0 = time.time()
                result = run_inference(model, pil_img, config)
                elapsed = round(time.time() - t0, 2)

                fp_flags = []
                for qc in config.get("query_checks", []):
                    q = result["queries"].get(qc["name"], {})
                    if qc.get("expected_in_normal") == "no" and q.get("label") == "yes":
                        fp_flags.append(qc["name"])

                worker = result["queries"].get("worker_visible", {}).get("label", "?")
                q_short = f"worker={worker}"
                if fp_flags:
                    q_short += f"  FP: {', '.join(fp_flags)}"

                print(f"  [{t_sec:6.1f}s] {elapsed:.1f}s | {q_short}")

                entry = {
                    "video": source.name,
                    "time_sec": t_sec,
                    "runtime_sec": elapsed,
                    "result": result,
                }
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")

                row = flatten_row(source.name, t_sec, result, config)
                rows.append(row)

                if args.annotate:
                    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", source.stem).strip("_")
                    fname = f"{slug}_{int(t_sec):05d}s.jpg"
                    draw_annotations(pil_img, result).save(annotated_dir / fname, quality=85)

                processed += 1
                frame_count += 1

            print(f"  → {frame_count} frames\n")

            if args.limit and processed >= args.limit:
                break

    total_time = round(time.time() - t_start, 1)

    csv_path = run_dir / "summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    meta = {
        "run_id": run_id,
        "test_type": "negative_sample_baseline",
        "description": "YOLO Gap 負樣本基線測試 — 正常作業影片的 FP 率統計",
        "model": args.model_id,
        "revision": args.revision,
        "device": device,
        "model_load_sec": load_sec,
        "sample_every_sec": every_sec,
        "total_frames": processed,
        "total_time_sec": total_time,
        "videos": [s.name for s in sources],
        "config_file": config_path.name,
        "annotate": args.annotate,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    # ── 快速統計 ──────────────────────────────────────────────────
    print(f"{'='*50}")
    print(f"完成！共 {processed} 幀 / {total_time}s")
    print(f"輸出: {run_dir}\n")

    print("--- Query FP 快速統計（expected=no 但 label=yes）---")
    for qc in config.get("query_checks", []):
        if qc.get("expected_in_normal") not in ("no", "yes"):
            continue
        name = qc["name"]
        expected = qc["expected_in_normal"]
        col = f"q_{name}_label"
        total = len(rows)
        if expected == "no":
            fp = sum(1 for r in rows if r.get(col) == "yes")
            pct = (fp / total * 100) if total else 0
            flag = " ← 注意" if fp > 0 else ""
            print(f"  {name:30s}  FP={fp}/{total} ({pct:.1f}%){flag}")
        elif expected == "yes":
            hit = sum(1 for r in rows if r.get(col) == "yes")
            pct = (hit / total * 100) if total else 0
            print(f"  {name:30s}  hit={hit}/{total} ({pct:.1f}%)")

    print("\n--- Detect 統計 ---")
    for dt in config.get("detect_targets", []):
        col = f"d_{dt['name'].replace(' ', '_')}_count"
        detected = sum(1 for r in rows if r.get(col, 0) > 0)
        cat = dt.get("category", "")
        tag = " (Gap: 應為0)" if cat == "yolo_gap" else ""
        print(f"  {dt['name']:24s}  有框={detected}/{len(rows)}{tag}")

    alert_rows = [r for r in rows if r.get("caption_alert_keywords")]
    if alert_rows:
        print(f"\n--- Caption 警示關鍵詞命中: {len(alert_rows)} 幀 ---")
        for r in alert_rows[:5]:
            print(f"  {r['video']} @ {r['time_sec']}s: {r['caption_alert_keywords']}")

    print(f"\n提示：執行 analyze_baseline.py 可產出完整報告")
    print(f"  python scripts/analyze_baseline.py --run {run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
