#!/usr/bin/env python3
"""Run Moondream detect() for the combined fall/smoking behavior dataset."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "outputs" / "behavior_fall_smoking_20260520T041151Z"
DEFAULT_BATCH2 = EXPERIMENT_DIR / "outputs" / "behavior_fall_smoking_batch2_20260520T0520Z"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Moondream bounding-box behavior localization test.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument(
        "--source-run",
        action="append",
        default=[],
        help="Run directory with manifest.jsonl. Can be passed multiple times.",
    )
    p.add_argument("--model-id", default="vikhyatk/moondream2")
    p.add_argument("--revision", default="2025-01-09")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--targeted-review",
        action="store_true",
        help=(
            "Run all fall/person frames, but only representative smoking/cigarette frames "
            "because cigarette detection is much slower."
        ),
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def choose_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_model(model_id: str, revision: str):
    from transformers import AutoModelForCausalLM

    device, dtype = choose_device()
    print(f"[model] loading {model_id} @ {revision} device={device}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()
    print(f"[model] ready in {time.time() - t0:.1f}s")
    return model, device


def report_source_folder(source_path: str) -> str:
    marker = "/data/"
    if marker in source_path:
        return "data/" + source_path.split(marker, 1)[1].rsplit("/", 1)[0]
    return str(Path(source_path).parent)


def load_clip_mapping(output_root: Path) -> dict[tuple[str, str], str]:
    mapping_path = output_root / "combined_behavior_summary.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing combined summary: {mapping_path}")
    mapping: dict[tuple[str, str], str] = {}
    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[(row["source_folder"], row["source_video"])] = row["clip_id"]
    return mapping


def target_for_event(event: str) -> str:
    if event == "fall":
        return "person"
    if event == "smoking":
        return "cigarette"
    return "person"


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str) -> None:
    font = ImageFont.load_default()
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], fill=fill)
    draw.text((x, y), text, fill="white", font=font)


def annotate(image: Image.Image, objects: list[dict], target: str) -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    color = "#ef4444" if target == "cigarette" else "#2563eb"
    for i, obj in enumerate(objects, 1):
        x0 = int(obj["x_min"] * image.width)
        y0 = int(obj["y_min"] * image.height)
        x1 = int(obj["x_max"] * image.width)
        y1 = int(obj["y_max"] * image.height)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
        draw_label(draw, (x0 + 5, max(5, y0 + 5)), f"{target} #{i}", color)
    if not objects:
        draw_label(draw, (12, 12), f"no {target} bbox", "#475569")
    return vis


def result_key(row: dict) -> str:
    return f"{row['clip_id']}:{row['frame_index']:04d}:{row['detect_target']}"


def build_manifest(source_runs: list[Path], output_root: Path) -> list[dict]:
    mapping = load_clip_mapping(output_root)
    items: list[dict] = []
    for source_run in source_runs:
        manifest = load_jsonl(source_run / "manifest.jsonl")
        for row in manifest:
            source_folder = report_source_folder(row.get("source_path", ""))
            clip_id = mapping.get((source_folder, row["source_video"]))
            if not clip_id:
                raise KeyError(f"Missing clip mapping for {source_folder} / {row['source_video']}")
            target = target_for_event(row["expected_event"])
            items.append(
                {
                    **row,
                    "source_run": source_run.name,
                    "source_run_path": str(source_run),
                    "source_folder": source_folder,
                    "clip_id": clip_id,
                    "detect_target": target,
                    "input_abs": str(source_run / row["frame_path"]),
                }
            )
    items.sort(key=lambda x: (x["clip_id"], x["frame_index"]))
    return items


def apply_targeted_review_filter(items: list[dict]) -> list[dict]:
    smoking_times = {0.0, 0.5, 1.0, 3.0, 3.5, 5.0, 6.0}
    filtered = []
    for row in items:
        if row["expected_event"] != "smoking":
            filtered.append(row)
            continue
        if round(float(row["time_sec"]), 1) in smoking_times:
            filtered.append(row)
    return filtered


def write_summary(out_dir: Path, results: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row["clip_id"]].append(row)

    rows = []
    for clip_id, items in sorted(grouped.items()):
        items.sort(key=lambda x: x["time_sec"])
        frames = len(items)
        with_box = sum(1 for x in items if x["object_count"] > 0)
        first = next((x["time_sec"] for x in items if x["object_count"] > 0), None)
        total_boxes = sum(x["object_count"] for x in items)
        rows.append(
            {
                "clip_id": clip_id,
                "expected_event": items[0]["expected_event"],
                "detect_target": items[0]["detect_target"],
                "frames": frames,
                "frames_with_box": with_box,
                "hit_rate": f"{with_box / frames * 100:.0f}%" if frames else "0%",
                "total_boxes": total_boxes,
                "first_box_time": f"{first:.1f} s" if first is not None else "未出現",
                "source_video": items[0]["source_video"],
            }
        )

    with (out_dir / "bbox_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    source_runs = [Path(p).resolve() for p in args.source_run]
    if not source_runs:
        source_runs = [output_root, DEFAULT_BATCH2.resolve()]

    out_dir = output_root / "bbox_experiment"
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "bbox_results.jsonl"

    manifest = build_manifest(source_runs, output_root)
    if args.targeted_review:
        manifest = apply_targeted_review_filter(manifest)
    processed = set()
    if results_path.exists() and not args.no_resume:
        processed = {result_key(row) for row in load_jsonl(results_path)}
    todo = [row for row in manifest if f"{row['clip_id']}:{row['frame_index']:04d}:{row['detect_target']}" not in processed]
    if args.limit:
        todo = todo[: args.limit]

    print("[bbox] behavior localization experiment")
    print(f"  output_root: {output_root}")
    print(f"  source_runs: {', '.join(p.name for p in source_runs)}")
    print(f"  frames: {len(manifest)} total, {len(todo)} to process")

    if not todo:
        write_summary(out_dir, load_jsonl(results_path))
        print("[bbox] nothing to process")
        return 0

    model, device = load_model(args.model_id, args.revision)
    with results_path.open("a", encoding="utf-8") as f:
        for row in todo:
            image_path = Path(row["input_abs"])
            image = Image.open(image_path).convert("RGB")
            t0 = time.time()
            encoded = model.encode_image(image)
            objects = model.detect(encoded, row["detect_target"])["objects"]
            elapsed = round(time.time() - t0, 2)

            clip_dir = annotated_dir / row["clip_id"]
            clip_dir.mkdir(parents=True, exist_ok=True)
            safe_target = row["detect_target"].replace(" ", "_")
            annotated_name = f"{row['clip_id']}_{row['frame_index']:04d}_{row['time_sec']:05.2f}s_{safe_target}.jpg"
            annotated_path = clip_dir / annotated_name
            annotate(image, objects, row["detect_target"]).save(annotated_path, quality=92)

            entry = {
                "clip_id": row["clip_id"],
                "frame_id": row["frame_id"],
                "video_id": row["video_id"],
                "expected_event": row["expected_event"],
                "source_video": row["source_video"],
                "source_folder": row["source_folder"],
                "source_run": row["source_run"],
                "frame_index": row["frame_index"],
                "time_sec": row["time_sec"],
                "frame_path": row["frame_path"],
                "detect_target": row["detect_target"],
                "object_count": len(objects),
                "objects": objects,
                "annotated_path": str(annotated_path.relative_to(output_root)),
                "runtime_sec": elapsed,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"  {row['clip_id']} {row['time_sec']:>4.1f}s "
                f"detect={row['detect_target']} boxes={len(objects)} {elapsed:.1f}s"
            )

    all_results = load_jsonl(results_path)
    write_summary(out_dir, all_results)
    meta = {
        "stage": "bbox_complete",
        "model": args.model_id,
        "revision": args.revision,
        "device": device,
        "frames": len(all_results),
        "source_runs": [str(p) for p in source_runs],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "bbox_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bbox] results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
