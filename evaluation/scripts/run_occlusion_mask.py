#!/usr/bin/env python3
"""
Run the mask/face-area equivalent of occlusion Phase 2.

This runs the selected `face_area_visible` prompt on every baseline frame
where `detect(person) > 0`, then cross-tabs the result against the baseline
`missing_face_mask` label.

Usage:
  cd evaluation
  conda run -n moondream python scripts/run_occlusion_mask.py
  conda run -n moondream python scripts/run_occlusion_mask.py --baseline-run 20260421T100247Z
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

PROMPT_NAME = "face_area_visible"
PROMPT = (
    "Can you clearly see the lower face area of this worker — meaning you can "
    "tell whether or not they are wearing a face mask or respirator? Answer "
    "yes if the face area is clearly visible, no if you cannot clearly see it "
    "due to angle, the worker facing away, distance, or obstruction."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mask/face-area Phase 2 occlusion experiment on frames with person detections."
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
    print("\n=== Mask Phase 2 Cross-tab ===")
    specs = [
        ("no", "正常幀，預期 yes"),
        ("yes", "FP幀，預期 no -> 被遮擋，應壓制"),
    ]
    for missing_face_mask, note in specs:
        subset = [r for r in rows if r["missing_face_mask"] == missing_face_mask]
        c = Counter(r["face_area_visible_label"] for r in subset)
        print(
            f"  missing_face_mask={missing_face_mask}  -> "
            f"face_area_visible=yes: {c['yes']}, no: {c['no']}  ({note})"
        )
        other = c["unclear"] + c["unknown"]
        if other:
            print(
                f"  missing_face_mask={missing_face_mask}  -> "
                f"other: {other} (unclear={c['unclear']}, unknown={c['unknown']})"
            )


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data).resolve()
    baseline_jsonl = OUTPUTS_DIR / args.baseline_run / "results.jsonl"
    if not baseline_jsonl.exists():
        sys.exit(f"[error] {baseline_jsonl}")

    entries = [json.loads(l) for l in baseline_jsonl.read_text("utf-8").splitlines() if l.strip()]
    # Keep the full person-detected set so the final cross-tab can compare
    # missing_face_mask=no vs yes. For FP-only mode, add that label check here.
    selected = [e for e in entries if e["result"]["detections"]["person"]["count"] > 0]

    out_dir = OUTPUTS_DIR / f"{args.baseline_run}_occlusion_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_out = out_dir / "mask_phase2_full.jsonl"

    print("=== Mask Occlusion Phase 2 ===")
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
            result = entry["result"]
            person_n = result["detections"]["person"]["count"]
            missing_face_mask = result["queries"]["missing_face_mask"]["label"]

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

            print(
                f"[{idx}/{len(selected)}] {video[:40]} @ {t_sec}s  "
                f"missing_face_mask={missing_face_mask} -> {label} ({elapsed}s)"
            )

            jf.write(
                json.dumps(
                    {
                        "video": video,
                        "time_sec": t_sec,
                        "baseline": {
                            "detect_person": person_n,
                            "missing_face_mask": missing_face_mask,
                        },
                        PROMPT_NAME: {"answer": answer, "label": label},
                        "runtime_sec": elapsed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            summary_rows.append(
                {
                    "missing_face_mask": missing_face_mask,
                    "face_area_visible_label": label,
                }
            )
            processed += 1

    (out_dir / "mask_phase2_run_meta.json").write_text(
        json.dumps(
            {
                "experiment": "occlusion_mask_phase2_full",
                "baseline_run": args.baseline_run,
                "frames_processed": processed,
                "prompt_name": PROMPT_NAME,
                "model": args.model_id,
                "revision": args.revision,
                "device": device,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_crosstab(summary_rows)
    print(f"\n完成 {processed} 幀，輸出: {jsonl_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
