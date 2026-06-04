#!/usr/bin/env python3
"""Run one Moondream query that asks for both fall and smoking signals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from run_behavior_experiment import load_jsonl, load_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_RUN_DIRS = [
    OUTPUTS_DIR / "behavior_fall_smoking_20260520T041151Z",
    OUTPUTS_DIR / "behavior_fall_smoking_batch2_20260520T0520Z",
]
DEFAULT_REPORT_DIR = DEFAULT_RUN_DIRS[0]
DEFAULT_OUTPUT_DIR = DEFAULT_REPORT_DIR / "combined_query_experiment"

COMBINED_PROMPT = (
    "Answer two checks for this image. "
    "Fall: is a person fallen, collapsed, or lying on the ground? "
    "Smoking: is a person smoking or holding a cigarette? "
    "Use this format only: Fall=<yes/no>; Smoking=<yes/no>; Evidence=<short reason>."
)

REPORT_ID_BY_SOURCE_VIDEO = {
    "fall_01.mp4": "fall_01",
    "fall_02.mp4": "fall_02",
    "fall_03.mp4": "fall_03",
    "smoking_01.mp4": "smoking_01",
    "smoking_02.mp4": "smoking_02",
    "smoking_03.mp4": "smoking_03",
    "smoking_04.mp4": "smoking_04",
}

LABEL_RE = re.compile(r"\b(yes|no|unclear)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one combined fall/smoking Moondream query per frame.")
    p.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Input run directory with manifest.jsonl. Can be passed multiple times.",
    )
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--model-id", default="vikhyatk/moondream2")
    p.add_argument("--revision", default="2025-01-09")
    p.add_argument("--limit", type=int, default=0, help="Process at most N new frames.")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def parse_label(answer: str, name: str) -> str:
    text = " ".join(answer.strip().split())
    patterns = [
        rf"{name}\s*[:=-]\s*(yes|no|unclear)\b",
        rf"{name}\s*=\s*(yes|no|unclear)\b",
        rf"{name}\s+is\s+(yes|no|unclear)\b",
    ]
    if name == "smoking":
        patterns.insert(1, r"smoke\s*[:=-]\s*(yes|no|unclear)\b")
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return "unknown"


def fallback_parse(answer: str) -> tuple[str, str]:
    fall = parse_label(answer, "fall")
    smoking = parse_label(answer, "smoking")
    return fall, smoking


def make_global_item(run_dir: Path, item: dict, report_dir: Path) -> dict:
    source_video = item["source_video"]
    report_id = REPORT_ID_BY_SOURCE_VIDEO.get(source_video, item["video_id"])
    frame_id = f"{report_id}:{int(item['frame_index']):04d}"
    image_path = run_dir / item["frame_path"]
    return {
        **item,
        "frame_id": frame_id,
        "video_id": report_id,
        "source_run_dir": str(run_dir),
        "image_path": str(image_path),
        "html_frame_path": os.path.relpath(image_path, report_dir),
    }


def collect_manifest(run_dirs: list[Path], report_dir: Path) -> list[dict]:
    items = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        for row in load_jsonl(manifest_path):
            items.append(make_global_item(run_dir, row, report_dir))
    items.sort(key=lambda x: (x["video_id"], int(x["frame_index"])))
    return items


def run_frame(model, image_path: Path) -> dict:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    encoded = model.encode_image(image)
    answer = model.query(encoded, COMBINED_PROMPT)["answer"].strip()
    fall_label, smoking_label = fallback_parse(answer)
    return {
        "prompt": COMBINED_PROMPT,
        "answer": answer,
        "fall_label": fall_label,
        "smoking_label": smoking_label,
    }


def write_summary(output_dir: Path) -> None:
    rows = load_jsonl(output_dir / "combined_query_results.jsonl")
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)

    summary_rows = []
    for video_id in sorted(by_video):
        items = sorted(by_video[video_id], key=lambda x: float(x["time_sec"]))
        expected = items[0]["expected_event"]
        fall_yes = sum(1 for x in items if x["result"]["fall_label"] == "yes")
        fall_unclear = sum(1 for x in items if x["result"]["fall_label"] == "unclear")
        smoking_yes = sum(1 for x in items if x["result"]["smoking_label"] == "yes")
        smoking_unclear = sum(1 for x in items if x["result"]["smoking_label"] == "unclear")
        runtimes = [float(x["runtime_sec"]) for x in items]
        summary_rows.append(
            {
                "video_id": video_id,
                "expected_event": expected,
                "frames": len(items),
                "fall_yes": fall_yes,
                "fall_unclear": fall_unclear,
                "smoking_yes": smoking_yes,
                "smoking_unclear": smoking_unclear,
                "avg_runtime_sec": round(sum(runtimes) / len(runtimes), 2),
                "min_runtime_sec": round(min(runtimes), 2),
                "max_runtime_sec": round(max(runtimes), 2),
                "first_fall_yes_sec": first_yes_time(items, "fall_label"),
                "first_smoking_yes_sec": first_yes_time(items, "smoking_label"),
            }
        )

    with (output_dir / "combined_query_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def first_yes_time(items: list[dict], key: str) -> str:
    for item in items:
        if item["result"][key] == "yes":
            return f"{float(item['time_sec']):.1f}"
    return ""


def main() -> int:
    args = parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dir] or [p.resolve() for p in DEFAULT_RUN_DIRS]
    report_dir = Path(args.report_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "combined_query_results.jsonl"
    manifest = collect_manifest(run_dirs, report_dir)
    processed = set()
    if results_path.exists() and not args.no_resume:
        processed = {row["frame_id"] for row in load_jsonl(results_path)}
    todo = [row for row in manifest if row["frame_id"] not in processed]
    if args.limit:
        todo = todo[: args.limit]

    print("[run] combined fall/smoking query experiment")
    print(f"  frames: {len(manifest)} total, {len(todo)} to process")
    print(f"  output: {output_dir}")
    print(f"  prompt: {COMBINED_PROMPT}")

    if not todo:
        write_summary(output_dir)
        print("[run] nothing to process")
        return 0

    model, device, load_sec = load_model(args.model_id, args.revision)
    t_start = time.time()
    with results_path.open("a", encoding="utf-8") as f:
        for item in todo:
            t0 = time.time()
            result = run_frame(model, Path(item["image_path"]))
            elapsed = round(time.time() - t0, 2)
            entry = {
                **{k: v for k, v in item.items() if k != "image_path"},
                "runtime_sec": elapsed,
                "result": result,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"  {item['frame_id']} {item['time_sec']:>4.1f}s "
                f"fall={result['fall_label']} smoking={result['smoking_label']} {elapsed:.1f}s"
            )

    write_summary(output_dir)
    meta = {
        "experiment": "combined_fall_smoking_single_query",
        "prompt": COMBINED_PROMPT,
        "run_dirs": [str(p) for p in run_dirs],
        "model": args.model_id,
        "revision": args.revision,
        "device": device,
        "model_load_sec": load_sec,
        "total_frames": len(manifest),
        "processed_frames": len(load_jsonl(results_path)),
        "last_pass_sec": round(time.time() - t_start, 2),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "combined_query_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[run] results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
