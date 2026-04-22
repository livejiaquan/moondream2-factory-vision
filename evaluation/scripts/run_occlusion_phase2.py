#!/usr/bin/env python3
"""
Run Phase 2 of the occlusion experiment.

Phase 2 runs the selected `can_confirm_helmet` prompt on every baseline frame
where `detect(person) > 0`, using the original video frames from `data/`.

Usage:
  cd evaluation
  conda run -n moondream python scripts/run_occlusion_phase2.py
  conda run -n moondream python scripts/run_occlusion_phase2.py --baseline-run 20260421T100247Z
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = EVAL_DIR / "outputs"

PROMPT_NAME = "can_confirm_helmet"
PROMPT = (
    "Can you clearly and confidently confirm whether or not this worker is "
    "wearing a safety helmet — meaning the helmet area is clearly visible and "
    "unobstructed? Answer yes if you can clearly tell, no if you cannot "
    "clearly tell due to angle, distance, or obstruction."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 occlusion experiment on all frames with person detections."
    )
    parser.add_argument("--baseline-run", default="20260421T100247Z")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--model-id", default="vikhyatk/moondream2")
    parser.add_argument("--revision", default="2025-01-09")
    return parser.parse_args()


def classify_yes_no(text: str) -> str:
    t = re.sub(r"^[\"'`\s]+", "", text.strip().lower())
    if re.match(r"^yes\b", t):
        return "yes"
    if re.match(r"^no\b", t):
        return "no"
    if re.match(
        r"^((it is |it's )?(unclear|uncertain)|cannot tell|can't tell|"
        r"unable to determine|hard to tell|not clear)\b", t,
    ):
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
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if revision:
        kwargs["revision"] = revision
    print(f"[model] loading {model_id} @ {revision} on {device}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()
    print(f"[model] ready in {round(time.time() - t0, 2)}s\n")
    return model, device


def extract_frame(video_path: Path, time_sec: float):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame at {time_sec}s")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def print_crosstab(rows: list[dict]) -> None:
    print("\n=== Phase 2 Cross-tab ===")
    specs = [
        (1, "no",  "正常幀（detect 到帽且 query 正確）→ 預期 can_confirm=yes"),
        (1, "yes", "detect 有帽但 query 說沒有       → 預期 can_confirm=yes"),
        (0, "no",  "detect 沒帽但 query 說 OK        → 預期 can_confirm=no"),
        (0, "yes", "detect 沒帽且 query 也說沒有     → 預期 can_confirm=no"),
    ]
    for det_h, mh, note in specs:
        subset = [r for r in rows if r["detect_helmet_count"] == det_h and r["missing_helmet"] == mh]
        if not subset:
            continue
        c = Counter(r["can_confirm_label"] for r in subset)
        print(f"\n  detect_helmet={det_h}, missing_helmet={mh}  ({note})")
        print(f"  n={len(subset)}  → can_confirm: yes={c['yes']} no={c['no']} unclear={c['unclear']} unknown={c['unknown']}")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data).resolve()
    baseline_jsonl = OUTPUTS_DIR / args.baseline_run / "results.jsonl"
    if not baseline_jsonl.exists():
        sys.exit(f"[error] {baseline_jsonl}")

    entries = [json.loads(l) for l in baseline_jsonl.read_text("utf-8").splitlines() if l.strip()]
    selected = [e for e in entries if e["result"]["detections"]["person"]["count"] > 0]

    out_dir = OUTPUTS_DIR / f"{args.baseline_run}_occlusion_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_out = out_dir / "phase2_full.jsonl"

    print("=== Occlusion Phase 2 ===")
    print(f"  Baseline : {args.baseline_run}")
    print(f"  Frames   : {len(selected)}/{len(entries)} (detect person > 0)")
    print(f"  Prompt   : {PROMPT_NAME}")
    print(f"  Output   : {jsonl_out}\n")

    model, device = load_model(args.model_id, args.revision)

    summary_rows: list[dict] = []
    processed = 0

    with jsonl_out.open("w", encoding="utf-8") as jf:
        for idx, entry in enumerate(selected, 1):
            video = entry["video"]
            t_sec = float(entry["time_sec"])
            r = entry["result"]
            person_n = r["detections"]["person"]["count"]
            helmet_n = r["detections"]["helmet"]["count"]
            mh = r["queries"]["missing_helmet"]["label"]

            video_path = data_dir / video
            if not video_path.exists():
                print(f"[skip] {video}")
                continue

            try:
                frame = extract_frame(video_path, t_sec)
            except Exception as e:
                print(f"[skip] {video} @ {t_sec}s: {e}")
                continue

            enc = model.encode_image(frame)
            t0 = time.time()
            answer = model.query(enc, PROMPT)["answer"]
            label = classify_yes_no(answer)
            elapsed = round(time.time() - t0, 2)

            print(f"[{idx}/{len(selected)}] {video[:40]} @ {t_sec}s  helmet={helmet_n} missing={mh} → {label} ({elapsed}s)")

            jf.write(json.dumps({
                "video": video, "time_sec": t_sec,
                "baseline": {"detect_person": person_n, "detect_helmet": helmet_n, "missing_helmet": mh},
                "can_confirm_helmet": {"answer": answer, "label": label},
                "runtime_sec": elapsed,
            }, ensure_ascii=False) + "\n")

            summary_rows.append({"detect_helmet_count": helmet_n, "missing_helmet": mh, "can_confirm_label": label})
            processed += 1

    (out_dir / "phase2_run_meta.json").write_text(json.dumps({
        "experiment": "occlusion_phase2_full",
        "baseline_run": args.baseline_run,
        "frames_processed": processed,
        "prompt_name": PROMPT_NAME,
        "model": args.model_id,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print_crosstab(summary_rows)
    print(f"\n完成 {processed} 幀，輸出: {jsonl_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
