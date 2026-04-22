#!/usr/bin/env python3
"""
Compare occlusion prompt variants on baseline helmet false-positive frames.

Phase 1: Test 5 prompt variants on FP frames to find the best occlusion prompt.
Phase 2: (uncomment below) Run best prompt on full dataset.

Usage:
  cd evaluation
  conda run -n moondream python scripts/run_occlusion_prompt_variants.py
  conda run -n moondream python scripts/run_occlusion_prompt_variants.py --baseline-run 20260421T100247Z
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = EVAL_DIR / "outputs"
CONFIG_PATH = EVAL_DIR / "configs" / "occlusion_variants.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run occlusion prompt variants on baseline helmet false-positive frames."
    )
    parser.add_argument("--baseline-run", default="20260421T100247Z")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--model-id", default="vikhyatk/moondream2")
    parser.add_argument("--revision", default="2025-01-09")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"[error] Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    variants = cfg.get("prompt_variants", [])
    if not variants:
        sys.exit(f"[error] No prompt_variants found in {path}")
    names = [v.get("name", "") for v in variants]
    if len(names) != len(set(names)):
        sys.exit(f"[error] Duplicate variant names in {path}")
    return cfg


def classify_yes_no(text: str) -> str:
    t = re.sub(r'^["\'\`\s]+', "", text.strip().lower())
    if re.match(r"^yes\b", t):
        return "yes"
    if re.match(r"^no\b", t):
        return "no"
    if re.match(
        r"^((it is |it's )?(unclear|uncertain)|cannot tell|can't tell|"
        r"unable to determine|hard to tell|not clear)\b", t
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
    print(f"[model] ready in {round(time.time()-t0, 2)}s\n")
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


def should_include(entry: dict) -> bool:
    result = entry["result"]
    if result["detections"]["person"]["count"] <= 0:
        return False
    helmet_count = result["detections"]["helmet"]["count"]
    missing_helmet = result["queries"]["missing_helmet"]["label"]
    return missing_helmet == "yes" or helmet_count == 0


def build_summary(flat_rows: list[dict], variants: list[dict], total: int) -> None:
    name_w = max(20, max(len(v["name"]) for v in variants))
    header = f"{'variant':<{name_w}}  {'flag_rule':<14}  {'flagged':>10}  {'yes':>5}  {'no':>5}  {'unclear':>8}  {'unknown':>8}"
    print(header)
    print("-" * len(header))
    for v in variants:
        flag_when = v.get("flag_when_label_is")
        rows = [r for r in flat_rows if r["variant_name"] == v["name"]]
        counts = {k: sum(1 for r in rows if r["label"] == k) for k in ("yes", "no", "unclear", "unknown")}
        flagged = sum(1 for r in rows if flag_when and r["label"] == flag_when)
        flagged_str = f"{flagged}/{total}" if flag_when else "n/a"
        rule = f"label={flag_when}" if flag_when else "n/a"
        print(f"{v['name']:<{name_w}}  {rule:<14}  {flagged_str:>10}  {counts['yes']:>5}  {counts['no']:>5}  {counts['unclear']:>8}  {counts['unknown']:>8}")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data)
    baseline_dir = OUTPUTS_DIR / args.baseline_run
    baseline_jsonl = baseline_dir / "results.jsonl"

    if not baseline_jsonl.exists():
        sys.exit(f"[error] Missing baseline results: {baseline_jsonl}")

    cfg = load_config(CONFIG_PATH)
    variants = cfg["prompt_variants"]

    entries = [json.loads(l) for l in baseline_jsonl.read_text("utf-8").splitlines() if l.strip()]
    selected = [e for e in entries if should_include(e)]

    out_dir = OUTPUTS_DIR / f"{args.baseline_run}_occlusion_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_out = out_dir / "variant_comparison.jsonl"
    csv_out = out_dir / "variant_summary.csv"

    print("=== Occlusion Prompt Variant Comparison ===")
    print(f"  Baseline : {args.baseline_run}")
    print(f"  Filter   : {len(selected)}/{len(entries)} 幀符合（person>0 且 missing_helmet=yes 或 helmet=0）")
    print(f"  Variants : {len(variants)}")
    print(f"  Output   : {out_dir}\n")

    model, device = load_model(args.model_id, args.revision)

    flat_rows: list[dict] = []
    processed = 0

    with jsonl_out.open("w", encoding="utf-8") as jf, csv_out.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["video", "time_sec", "variant_name", "label", "answer"])
        writer.writeheader()

        for idx, entry in enumerate(selected, 1):
            video = entry["video"]
            t_sec = entry["time_sec"]
            r = entry["result"]
            person_n = r["detections"]["person"]["count"]
            helmet_n = r["detections"]["helmet"]["count"]
            mh = r["queries"]["missing_helmet"]["label"]
            video_path = data_dir / video

            if not video_path.exists():
                print(f"[skip] {video} — 找不到影片")
                continue

            try:
                frame = extract_frame(video_path, float(t_sec))
            except Exception as e:
                print(f"[skip] {video} @ {t_sec}s: {e}")
                continue

            enc = model.encode_image(frame)
            t0 = time.time()

            print(f"[{idx}/{len(selected)}] {video[:45]} @ {t_sec}s  person={person_n} helmet={helmet_n} missing_helmet={mh}")

            variant_results = {}
            for v in variants:
                answer = model.query(enc, v["prompt"])["answer"]
                label = classify_yes_no(answer)
                flag_when = v.get("flag_when_label_is")
                variant_results[v["name"]] = {
                    "prompt": v["prompt"],
                    "answer": answer,
                    "label": label,
                    "flag_when_label_is": flag_when,
                    "flagged": label == flag_when if flag_when else None,
                }
                flat_rows.append({"video": video, "time_sec": t_sec, "variant_name": v["name"], "label": label, "answer": answer})
                writer.writerow({"video": video, "time_sec": t_sec, "variant_name": v["name"], "label": label, "answer": answer})
                print(f"  {v['name']}: {label}")

            elapsed = round(time.time() - t0, 2)
            print(f"  → {elapsed}s\n")

            jf.write(json.dumps({
                "video": video, "time_sec": t_sec,
                "baseline": {"detect_person": person_n, "detect_helmet": helmet_n, "missing_helmet": mh},
                "variants": variant_results,
                "runtime_sec": elapsed,
            }, ensure_ascii=False) + "\n")
            processed += 1

    (out_dir / "run_meta.json").write_text(json.dumps({
        "experiment": "occlusion_prompt_variants",
        "baseline_run": args.baseline_run,
        "frames_selected": len(selected),
        "frames_processed": processed,
        "variant_count": len(variants),
        "model": args.model_id,
        "revision": args.revision,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("=== Phase 1 Summary ===")
    print("Flagged = variant 認為頭部不可見/被遮擋的幀數\n")
    build_summary(flat_rows, variants, processed)
    print(f"\n輸出: {out_dir}")
    print("下一步: 選出最有效的 variant，在 Phase 2 跑全部 47 幀")
    print("  最有效 = flagged 率最高（代表最能識別遮擋）且答案穩定（unclear 少）")

    # ── Phase 2（選好 prompt 後取消註解）────────────────────────────
    # best_variant_name = "can_confirm_helmet"   # ← 根據 Phase 1 結果填入
    # best_prompt = next(v["prompt"] for v in variants if v["name"] == best_variant_name)
    # full_entries = [e for e in entries if e["result"]["detections"]["person"]["count"] > 0]
    # out2 = out_dir / "full_run_best_prompt.jsonl"
    # with out2.open("w", encoding="utf-8") as f2:
    #     for entry in full_entries:
    #         video_path = data_dir / entry["video"]
    #         frame = extract_frame(video_path, float(entry["time_sec"]))
    #         enc = model.encode_image(frame)
    #         answer = model.query(enc, best_prompt)["answer"]
    #         label = classify_yes_no(answer)
    #         f2.write(json.dumps({...}, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
