#!/usr/bin/env python3
"""Run Moondream prompt variants for fall/smoking behavior detection."""

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

PROMPT_VARIANTS = {
    "A": (
        "Answer two independent checks for this image.\n"
        "Fall check: Is a person fallen or lying on the ground? "
        "Fall=<yes/no>; Fall-evidence=<reason about fall only>\n"
        "Smoking check: Is a person visibly holding or smoking a cigarette? "
        "Smoking=<yes/no>; Smoking-evidence=<reason about smoking only>"
    ),
    "B": (
        "Answer these two independent safety checks. For each, look only at the evidence "
        "relevant to that specific check.\n"
        "1. FALL: Is a person fallen, collapsed, or lying on the ground? "
        "Answer: Fall=yes or Fall=no\n"
        "2. SMOKING: Can you see a cigarette, cigar, or vape in someone's hand or mouth? "
        "Answer: Smoking=yes or Smoking=no\n"
        "Provide one short reason for each answer."
    ),
    "C": (
        "Answer two checks for this image. Fall: is a person fallen, collapsed, or lying "
        "on the ground? Smoking: is a lit cigarette, cigar, or vape device visibly in a "
        "person's hand or mouth? Use this format only: Fall=<yes/no>; Smoking=<yes/no>; "
        "Evidence=<short reason>."
    ),
    "D": "Is anyone in this image fallen on the ground or visibly smoking a cigarette?",
}

REPORT_ID_BY_SOURCE_VIDEO = {
    "fall_01.mp4": "fall_01",
    "fall_02.mp4": "fall_02",
    "fall_03.mp4": "fall_03",
    "smoking_01.mp4": "smoking_01",
    "smoking_02.mp4": "smoking_02",
    "smoking_03.mp4": "smoking_03",
    "smoking_04.mp4": "smoking_04",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run four Moondream prompt variants for fall/smoking behavior detection."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(PROMPT_VARIANTS),
        help="Run only one prompt variant. Default: run all variants.",
    )
    parser.add_argument("--output-dir", default="", help="Optional existing/new output directory.")
    parser.add_argument("--model-id", default="vikhyatk/moondream2")
    parser.add_argument("--revision", default="2025-01-09")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--key-times",
        type=float,
        nargs="+",
        default=None,
        help="Only run frames at these time points (e.g. --key-times 0.0 3.0 6.0).",
    )
    return parser.parse_args()


def timestamped_output_dir() -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUTS_DIR / f"prompt_variants_{run_id}"


def verify_manifests(run_dirs: list[Path]) -> None:
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        if not os.access(manifest_path, os.R_OK):
            raise PermissionError(f"manifest is not readable: {manifest_path}")


def make_global_item(run_dir: Path, item: dict) -> dict:
    source_video = item.get("source_video", "")
    video_id = REPORT_ID_BY_SOURCE_VIDEO.get(source_video, item["video_id"])
    frame_index = int(item["frame_index"])
    return {
        "frame_id": f"{video_id}:{frame_index:04d}",
        "video_id": video_id,
        "expected_event": item["expected_event"],
        "time_sec": float(item["time_sec"]),
        "frame_index": frame_index,
        "frame_path": item["frame_path"],
        "image_path": run_dir / item["frame_path"],
    }


def collect_manifest(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        for item in load_jsonl(run_dir / "manifest.jsonl"):
            rows.append(make_global_item(run_dir, item))
    rows.sort(key=lambda x: (x["video_id"], x["frame_index"], x["time_sec"]))
    return rows


def result_path(output_dir: Path, variant: str) -> Path:
    return output_dir / f"variant_{variant}_results.jsonl"


def parse_structured_labels(answer: str) -> tuple[str, str]:
    return parse_field_label(answer, "fall"), parse_field_label(answer, "smoking")


def parse_field_label(answer: str, field: str) -> str:
    if re.search(rf"\b{field}\s*=\s*yes\b", answer, flags=re.IGNORECASE):
        return "yes"
    if re.search(rf"\b{field}\s*=\s*no\b", answer, flags=re.IGNORECASE):
        return "no"
    return "unknown"


def parse_direct_labels(answer: str) -> tuple[str, str]:
    text = answer.lower()
    fall_negative = any(marker in text for marker in ("no one", "not", "no fall"))
    smoking_negative = any(marker in text for marker in ("no one", "not", "no smoking"))
    fall_yes = ("fall" in text or "fallen" in text) and not fall_negative
    smoking_yes = ("smoking" in text or "cigarette" in text) and not smoking_negative
    return ("yes" if fall_yes else "unknown", "yes" if smoking_yes else "unknown")


def parse_labels(answer: str, variant: str) -> tuple[str, str]:
    if variant == "D":
        return parse_direct_labels(answer)
    return parse_structured_labels(answer)


def load_processed_keys(output_dir: Path, variants: list[str], no_resume: bool) -> set[tuple[str, str]]:
    if no_resume:
        return set()
    processed = set()
    for variant in variants:
        for row in load_jsonl(result_path(output_dir, variant)):
            processed.add((row.get("frame_id", ""), row.get("variant", variant)))
    return processed


def run_query(model, image_path: Path, prompt: str) -> str:
    from PIL import Image

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    encoded = model.encode_image(image)
    return model.query(encoded, prompt)["answer"].strip()


def write_summary(output_dir: Path, variants: list[str]) -> Path:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for variant in variants:
        for row in load_jsonl(result_path(output_dir, variant)):
            grouped[(variant, row["video_id"], row["expected_event"])].append(row)

    summary_path = output_dir / "variant_comparison_summary.csv"
    fieldnames = [
        "variant",
        "video_id",
        "expected_event",
        "frames",
        "fall_yes",
        "smoking_yes",
        "avg_runtime_sec",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped):
            rows = grouped[key]
            runtimes = [float(row["runtime_sec"]) for row in rows]
            writer.writerow(
                {
                    "variant": key[0],
                    "video_id": key[1],
                    "expected_event": key[2],
                    "frames": len(rows),
                    "fall_yes": sum(1 for row in rows if row["fall_label"] == "yes"),
                    "smoking_yes": sum(1 for row in rows if row["smoking_label"] == "yes"),
                    "avg_runtime_sec": round(sum(runtimes) / len(runtimes), 2),
                }
            )
    return summary_path


def print_summary(summary_path: Path) -> None:
    print("[summary]")
    print(summary_path.read_text(encoding="utf-8").rstrip())


def main() -> int:
    args = parse_args()
    variants = [args.variant] if args.variant else sorted(PROMPT_VARIANTS)
    run_dirs = [path.resolve() for path in DEFAULT_RUN_DIRS]
    verify_manifests(run_dirs)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = collect_manifest(run_dirs)
    if args.key_times is not None:
        key_set = set(args.key_times)
        manifest = [m for m in manifest if m["time_sec"] in key_set]
        print(f"  key-times filter: {sorted(key_set)} → {len(manifest)} frames kept")
    processed = load_processed_keys(output_dir, variants, args.no_resume)
    pending_count = sum(
        1
        for variant in variants
        for item in manifest
        if (item["frame_id"], variant) not in processed and item["image_path"].exists()
    )

    print("[run] prompt variant experiment")
    print(f"  manifests: {', '.join(str(path / 'manifest.jsonl') for path in run_dirs)}")
    print(f"  frames: {len(manifest)}")
    print(f"  variants: {', '.join(variants)}")
    print(f"  output: {output_dir}")
    print(f"  pending queries: {pending_count}")

    model = None
    device = ""
    load_sec = 0.0
    if pending_count:
        model, device, load_sec = load_model(args.model_id, args.revision)

    started_at = time.time()
    processed_this_run = 0
    skipped_missing = 0
    for variant in variants:
        prompt = PROMPT_VARIANTS[variant]
        path = result_path(output_dir, variant)
        with path.open("a", encoding="utf-8") as f:
            for item in manifest:
                key = (item["frame_id"], variant)
                if key in processed:
                    continue
                image_path = item["image_path"]
                if not image_path.exists():
                    print(f"[warn] missing frame, skipped: {image_path}")
                    skipped_missing += 1
                    continue
                if model is None:
                    raise RuntimeError("model was not loaded for pending work")

                t0 = time.time()
                raw_answer = run_query(model, image_path, prompt)
                elapsed = round(time.time() - t0, 2)
                fall_label, smoking_label = parse_labels(raw_answer, variant)
                entry = {
                    "frame_id": item["frame_id"],
                    "video_id": item["video_id"],
                    "expected_event": item["expected_event"],
                    "time_sec": item["time_sec"],
                    "variant": variant,
                    "raw_answer": raw_answer,
                    "fall_label": fall_label,
                    "smoking_label": smoking_label,
                    "runtime_sec": elapsed,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                processed.add(key)
                processed_this_run += 1
                print(
                    f"  variant={variant} {item['frame_id']} {item['time_sec']:>4.1f}s "
                    f"fall={fall_label} smoking={smoking_label} {elapsed:.1f}s"
                )

    summary_path = write_summary(output_dir, variants)
    meta = {
        "experiment": "prompt_variant_fall_smoking",
        "variants": variants,
        "prompts": PROMPT_VARIANTS,
        "run_dirs": [str(path) for path in run_dirs],
        "model": args.model_id,
        "revision": args.revision,
        "device": device,
        "model_load_sec": load_sec,
        "manifest_frames": len(manifest),
        "processed_this_run": processed_this_run,
        "skipped_missing": skipped_missing,
        "elapsed_sec": round(time.time() - started_at, 2),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[run] processed this run: {processed_this_run}")
    if skipped_missing:
        print(f"[run] missing frames skipped: {skipped_missing}")
    print(f"[run] summary: {summary_path}")
    print_summary(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
