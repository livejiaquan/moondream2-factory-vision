#!/usr/bin/env python3
"""Extract still frames for the fall/smoking behavior experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "behavior_videos"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract behavior-test frames from short videos.")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="Input folder containing fall/smoking videos.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT), help="Root output folder.")
    p.add_argument("--run-id", default="", help="Optional run folder name.")
    p.add_argument("--every", type=float, default=0.5, help="Frame sampling interval in seconds.")
    p.add_argument("--max-width", type=int, default=0, help="Resize frames wider than this. 0 keeps original size.")
    p.add_argument("--quality", type=int, default=92, help="JPEG quality.")
    p.add_argument(
        "--fall-patterns",
        default="",
        help="Comma-separated filename substrings that should be labeled as fall videos.",
    )
    p.add_argument(
        "--smoking-patterns",
        default="",
        help="Comma-separated filename substrings that should be labeled as smoking videos.",
    )
    return p.parse_args()


def safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "video"


def iter_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def split_patterns(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def assign_sources(videos: list[Path], fall_patterns: list[str], smoking_patterns: list[str]) -> list[dict]:
    smoking_idx = 1
    fall_idx = 1
    assigned = []
    for video in videos:
        lower = video.name.lower()
        is_fall = "fall" in lower or any(pattern in lower for pattern in fall_patterns)
        is_smoking = any(pattern in lower for pattern in smoking_patterns)
        if is_fall and is_smoking:
            raise ValueError(f"video matches both fall and smoking patterns: {video.name}")
        if is_fall:
            video_id = "fall" if fall_idx == 1 else f"fall_{fall_idx:02d}"
            fall_idx += 1
            expected_event = "fall"
        else:
            video_id = f"smoking_{smoking_idx:02d}"
            smoking_idx += 1
            expected_event = "smoking"
        assigned.append(
            {
                "video_id": video_id,
                "expected_event": expected_event,
                "source_name": video.name,
                "source_path": str(video.resolve()),
            }
        )
    return assigned


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_root).resolve()

    videos = iter_videos(input_path)
    if not videos:
        print(f"[error] no videos found: {input_path}")
        return 1

    run_id = args.run_id or "behavior_fall_smoking_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    frames_root = run_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    import cv2

    sources = assign_sources(videos, split_patterns(args.fall_patterns), split_patterns(args.smoking_patterns))
    manifest_rows: list[dict] = []
    total_frames = 0

    print("[extract] behavior frame extraction")
    print(f"  input:  {input_path}")
    print(f"  output: {run_dir}")
    print(f"  every:  {args.every}s")

    for source in sources:
        video_path = Path(source["source_path"])
        video_dir = frames_root / source["video_id"]
        video_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[warn] cannot open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_total / fps if fps else 0
        step_ms = max(args.every, 0.1) * 1000.0
        pos_ms = 0.0
        frame_index = 0

        print(f"--- {source['video_id']} ({video_path.name}) duration={duration:.2f}s fps={fps:.2f}")

        while pos_ms <= duration * 1000.0 + 1:
            cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if args.max_width and frame.shape[1] > args.max_width:
                scale = args.max_width / frame.shape[1]
                new_size = (args.max_width, int(frame.shape[0] * scale))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

            time_sec = round(pos_ms / 1000.0, 2)
            fname = f"{source['video_id']}_{frame_index:04d}_{time_sec:05.2f}s.jpg"
            out_path = video_dir / fname
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])

            rel_path = out_path.relative_to(run_dir)
            row = {
                "frame_id": f"{source['video_id']}:{frame_index:04d}",
                "video_id": source["video_id"],
                "expected_event": source["expected_event"],
                "source_video": source["source_name"],
                "source_path": source["source_path"],
                "frame_index": frame_index,
                "time_sec": time_sec,
                "frame_path": str(rel_path),
            }
            manifest_rows.append(row)
            frame_index += 1
            pos_ms += step_ms

        cap.release()
        total_frames += frame_index
        print(f"  frames: {frame_index}")

    manifest_jsonl = run_dir / "manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_csv = run_dir / "manifest.csv"
    if manifest_rows:
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    meta = {
        "run_id": run_id,
        "test_type": "behavior_fall_smoking",
        "stage": "frames_extracted",
        "input": str(input_path),
        "frame_every_sec": args.every,
        "total_frames": total_frames,
        "videos": sources,
        "frames_dir": str(frames_root.relative_to(run_dir)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[extract] manifest: {manifest_jsonl}")
    print(f"[extract] total frames: {total_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
