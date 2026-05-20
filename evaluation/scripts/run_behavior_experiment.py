#!/usr/bin/env python3
"""Run Moondream behavior queries on extracted fall/smoking frames."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_CONFIG = EVAL_DIR / "configs" / "behavior_fall_smoking.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run frame-level behavior queries with Moondream.")
    p.add_argument("--run-dir", required=True, help="Run directory produced by extract_behavior_frames.py.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Behavior test config JSON.")
    p.add_argument("--model-id", default="vikhyatk/moondream2")
    p.add_argument("--revision", default="2025-01-09")
    p.add_argument("--limit", type=int, default=0, help="Process at most N new frames.")
    p.add_argument("--no-resume", action="store_true", help="Do not skip frames already present in results.jsonl.")
    p.add_argument(
        "--relevant-only",
        action="store_true",
        help="Run only queries for the frame's expected event plus abnormal/open evidence queries.",
    )
    p.add_argument(
        "--query-names",
        default="",
        help="Optional comma-separated allowlist of query names to run.",
    )
    p.add_argument("--skip-caption", action="store_true", help="Skip caption generation to reduce runtime.")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def classify_answer(answer: str) -> str:
    text = answer.strip().lower()
    if re.match(r"^(yes\b|yes[,.!\s])", text):
        return "yes"
    if re.match(r"^(no\b|no[,.!\s])", text):
        return "no"
    if re.match(r"^(unclear\b|unclear[,.!\s])", text):
        return "unclear"
    return "unknown"


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
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()
    elapsed = round(time.time() - t0, 2)
    print(f"[model] ready in {elapsed}s")
    return model, device, elapsed


def should_run_query(query: dict, expected_event: str, relevant_only: bool, allowed_names: set[str]) -> bool:
    if allowed_names and query.get("name") not in allowed_names:
        return False
    if not relevant_only:
        return True
    event = query.get("event", "")
    return event in {expected_event, "abnormal", "evidence"}


def run_frame(
    model,
    image_path: Path,
    config: dict,
    expected_event: str,
    relevant_only: bool,
    skip_caption: bool,
    allowed_names: set[str],
) -> dict:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    enc = model.encode_image(image)
    caption = ""
    if not skip_caption:
        caption = model.caption(enc, length=config.get("caption_length", "normal"))["caption"]
    result = {
        "caption": caption,
        "queries": {},
    }

    for query in config.get("queries", []):
        if not should_run_query(query, expected_event, relevant_only, allowed_names):
            continue
        answer = model.query(enc, query["prompt"])["answer"]
        label = classify_answer(answer) if query.get("kind") != "open" else "open"
        result["queries"][query["name"]] = {
            "answer": answer,
            "label": label,
            "prompt": query["prompt"],
            "method": query.get("method", ""),
            "event": query.get("event", ""),
            "kind": query.get("kind", ""),
        }

    return result


def flatten(entry: dict, config: dict) -> dict:
    row = {
        "frame_id": entry["frame_id"],
        "video_id": entry["video_id"],
        "expected_event": entry["expected_event"],
        "source_video": entry["source_video"],
        "time_sec": entry["time_sec"],
        "frame_path": entry["frame_path"],
        "runtime_sec": entry["runtime_sec"],
        "caption": entry["result"].get("caption", ""),
    }
    for query in config.get("queries", []):
        q = entry["result"]["queries"].get(query["name"], {})
        row[f"{query['name']}_label"] = q.get("label", "")
        row[f"{query['name']}_answer"] = q.get("answer", "")
    return row


def write_summary(run_dir: Path, config: dict) -> None:
    results = load_jsonl(run_dir / "results.jsonl")
    rows = [flatten(entry, config) for entry in results]
    if not rows:
        return
    out = run_dir / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def update_meta(run_dir: Path, update: dict) -> None:
    meta_path = run_dir / "run_meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text("utf-8"))
    meta.update(update)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve()
    manifest_path = run_dir / "manifest.jsonl"
    results_path = run_dir / "results.jsonl"

    if not manifest_path.exists():
        print(f"[error] missing manifest: {manifest_path}")
        return 1
    if not config_path.exists():
        print(f"[error] missing config: {config_path}")
        return 1

    config = json.loads(config_path.read_text("utf-8"))
    allowed_names = {name.strip() for name in args.query_names.split(",") if name.strip()}
    manifest = load_jsonl(manifest_path)
    processed_ids = set()
    if not args.no_resume:
        processed_ids = {row.get("frame_id") for row in load_jsonl(results_path)}

    todo = [row for row in manifest if row["frame_id"] not in processed_ids]
    if args.limit:
        todo = todo[: args.limit]

    print("[run] behavior experiment")
    print(f"  run_dir: {run_dir}")
    print(f"  config:  {config_path.name}")
    print(f"  frames:  {len(manifest)} total, {len(todo)} to process")

    if not todo:
        write_summary(run_dir, config)
        print("[run] nothing to process")
        return 0

    model, device, load_sec = load_model(args.model_id, args.revision)
    t_start = time.time()
    processed = 0

    with results_path.open("a", encoding="utf-8") as f:
        for item in todo:
            image_path = run_dir / item["frame_path"]
            t0 = time.time()
            result = run_frame(
                model,
                image_path,
                config,
                item["expected_event"],
                args.relevant_only,
                args.skip_caption,
                allowed_names,
            )
            elapsed = round(time.time() - t0, 2)

            entry = {
                **item,
                "runtime_sec": elapsed,
                "result": result,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

            fall = result["queries"].get("fall_direct", {}).get("label", "?")
            fall_s = result["queries"].get("fall_strict", {}).get("label", "?")
            smoke = result["queries"].get("smoking_direct", {}).get("label", "?")
            smoke_s = result["queries"].get("smoking_strict", {}).get("label", "?")
            print(
                f"  {item['frame_id']} {item['time_sec']:>4.1f}s "
                f"fall={fall}/{fall_s} smoke={smoke}/{smoke_s} {elapsed:.1f}s"
            )
            processed += 1

    total_sec = round(time.time() - t_start, 1)
    update_meta(
        run_dir,
        {
            "stage": "inference_complete",
            "config_file": str(config_path.relative_to(EVAL_DIR)),
            "model": args.model_id,
            "revision": args.revision,
            "device": device,
            "model_load_sec": load_sec,
            "inference_processed_frames": len(load_jsonl(results_path)),
            "last_inference_sec": total_sec,
            "relevant_only": args.relevant_only,
            "caption_skipped": args.skip_caption,
            "query_names": sorted(allowed_names),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_summary(run_dir, config)
    print(f"[run] processed this pass: {processed}")
    print(f"[run] results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
