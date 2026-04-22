#!/usr/bin/env python3
"""
Small-sample mask prompt variant test.

Step 1: Test N prompt variants on a small sample (3 FP + 3 OK frames).
        Pick the best discriminating prompt.
Step 2: Run winner on all frames via --full-run.

Usage:
  cd evaluation
  # Step 1 – quick sample test (default 3+3 frames)
  conda run -n moondream python scripts/run_mask_prompt_variants.py

  # Step 2 – full run with chosen winner
  conda run -n moondream python scripts/run_mask_prompt_variants.py \
      --full-run --winner face_ppe_present
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR     = Path(__file__).resolve().parents[1]
OUTPUTS_DIR  = EVAL_DIR / "outputs"

BASELINE_RUN = "20260421T100247Z"

# ── Prompt variants ───────────────────────────────────────────────────────────
VARIANTS = [
    {
        "name": "face_ppe_present",
        "flag_when": "yes",    # yes = PPE found = NOT a violation
        "prompt": (
            "Can you see any face protection equipment on this worker — "
            "such as a mask, respirator, or any covering over the mouth and nose? "
            "This includes equipment visible from the front, side, or any angle. "
            "Answer yes if any face protection is visible, no if the face area "
            "appears unprotected, unclear if the view is completely blocked."
        ),
    },
    {
        "name": "anything_on_face",
        "flag_when": "yes",
        "prompt": (
            "Is there anything attached to or covering the lower half of this "
            "worker's face — including a mask, respirator, breathing apparatus, "
            "or similar equipment? Answer yes or no first, then briefly describe "
            "what you see around the face area."
        ),
    },
    {
        "name": "bare_face_visible",
        "flag_when": "yes",
        "prompt": (
            "Can you clearly see that this worker's mouth or nose area is "
            "completely bare and uncovered — with no mask, respirator, or face "
            "covering of any kind? Answer yes only if you can clearly see an "
            "unprotected face. Answer no if the face appears covered or you "
            "cannot tell. Answer unclear if the face area is not visible at all."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-run", default=BASELINE_RUN)
    p.add_argument("--data", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--sample-fp", type=int, default=3, help="FP frames for sample test")
    p.add_argument("--sample-ok", type=int, default=3, help="OK frames for sample test")
    p.add_argument("--full-run", action="store_true", help="Run winner on all frames")
    p.add_argument("--winner", default="", help="Variant name to use in --full-run")
    p.add_argument("--model-id", default="vikhyatk/moondream2")
    p.add_argument("--revision", default="2025-01-09")
    return p.parse_args()


def classify(text: str) -> str:
    t = re.sub(r"^[\"'`\s]+", "", text.strip().lower())
    if re.match(r"^yes\b", t): return "yes"
    if re.match(r"^no\b",  t): return "no"
    if re.match(r"^(unclear|cannot tell|can't tell|uncertain|hard to tell|not clear)\b", t):
        return "unclear"
    return "unknown"


def choose_device():
    import torch
    if torch.backends.mps.is_available(): return "mps", torch.float16
    if torch.cuda.is_available():         return "cuda", torch.float16
    return "cpu", torch.float32


def load_model(model_id, revision):
    from transformers import AutoModelForCausalLM
    device, dtype = choose_device()
    print(f"[model] loading {model_id} @ {revision} on {device}")
    t0 = time.time()
    model = (AutoModelForCausalLM
             .from_pretrained(model_id, trust_remote_code=True,
                              torch_dtype=dtype, revision=revision)
             .to(device).eval())
    print(f"[model] ready in {round(time.time()-t0,2)}s\n")
    return model, device


def extract_frame(video_path: Path, time_sec: float):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame at {time_sec}s")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def print_sample_table(results: list[dict]) -> None:
    col_w = 20
    vnames = [v["name"] for v in VARIANTS]
    header = f"{'frame':<35}  {'group':<5}  " + "  ".join(f"{n:<{col_w}}" for n in vnames)
    print(header)
    print("-" * len(header))
    for r in results:
        fname = f"{Path(r['video']).name[:25]} @{r['time_sec']}s"
        row = f"{fname:<35}  {r['group']:<5}  "
        row += "  ".join(f"{r['labels'].get(n,'?'):<{col_w}}" for n in vnames)
        print(row)

    print("\n=== 辨識率 (正確方向) ===")
    print(f"{'variant':<22}  {'FP→yes(suppressed)':<22}  {'OK→yes(kept)':<18}  {'score'}")
    print("-" * 80)
    fp_rows = [r for r in results if r["group"] == "FP"]
    ok_rows = [r for r in results if r["group"] == "OK"]
    for v in VARIANTS:
        n = v["name"]
        fp_yes = sum(1 for r in fp_rows if r["labels"].get(n) == "yes")
        ok_yes = sum(1 for r in ok_rows if r["labels"].get(n) == "yes")
        # score = FP correctly suppressed + OK correctly kept (both should be yes for face_ppe_present logic)
        score = fp_yes + ok_yes
        print(f"{n:<22}  {fp_yes}/{len(fp_rows)}{'':<18}  {ok_yes}/{len(ok_rows)}{'':<12}  {score}/{len(results)}")
    print()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data)
    mask_jsonl = OUTPUTS_DIR / f"{args.baseline_run}_occlusion_variants" / "mask_phase2_full.jsonl"
    rows = [json.loads(l) for l in mask_jsonl.read_text("utf-8").splitlines() if l.strip()]

    fp_pool = [r for r in rows if r.get("baseline", {}).get("missing_face_mask") == "yes"]
    ok_pool = [r for r in rows if r.get("baseline", {}).get("missing_face_mask") == "no"]

    out_dir = OUTPUTS_DIR / f"{args.baseline_run}_occlusion_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Full run ──────────────────────────────────────────────────────────────
    if args.full_run:
        if not args.winner:
            print("[error] --full-run requires --winner <variant-name>")
            return 1
        variant = next((v for v in VARIANTS if v["name"] == args.winner), None)
        if variant is None:
            print(f"[error] Unknown variant: {args.winner}")
            return 1

        all_person_rows = [r for r in rows]   # mask_phase2 already filtered person>0
        print(f"=== Mask Full Run: {args.winner} ({len(all_person_rows)} frames) ===\n")
        model, device = load_model(args.model_id, args.revision)
        out_jsonl = out_dir / f"mask_fullrun_{args.winner}.jsonl"

        summary = []
        with out_jsonl.open("w", encoding="utf-8") as jf:
            for idx, entry in enumerate(all_person_rows, 1):
                video = entry["video"]
                t_sec = float(entry["time_sec"])
                vp = data_dir / video
                if not vp.exists():
                    print(f"[skip] {video}")
                    continue
                try:
                    frame = extract_frame(vp, t_sec)
                except Exception as e:
                    print(f"[skip] {video} @ {t_sec}s: {e}")
                    continue
                enc = model.encode_image(frame)
                t0 = time.time()
                answer = model.query(enc, variant["prompt"])["answer"]
                label  = classify(answer)
                elapsed = round(time.time() - t0, 2)
                mfm = entry.get("baseline", {}).get("missing_face_mask", "?")
                print(f"[{idx}/{len(all_person_rows)}] {video[:40]} @ {t_sec}s  missing_mask={mfm} → {label} ({elapsed}s)")
                jf.write(json.dumps({
                    "video": video, "time_sec": t_sec,
                    "baseline": entry.get("baseline", {}),
                    "variant": args.winner,
                    args.winner: {"answer": answer, "label": label},
                    "runtime_sec": elapsed,
                }, ensure_ascii=False) + "\n")
                summary.append({"missing_face_mask": mfm, "label": label})

        # Summary stats
        from collections import Counter
        print("\n=== Full Run Summary ===")
        for grp, grp_label in [("yes", "FP frames (missing_mask=yes)"),
                                ("no",  "OK frames (missing_mask=no)")]:
            subset = [s for s in summary if s["missing_face_mask"] == grp]
            c = Counter(s["label"] for s in subset)
            print(f"  {grp_label}: n={len(subset)}  yes={c['yes']} no={c['no']} unclear={c['unclear']} unknown={c['unknown']}")

        (out_dir / f"mask_fullrun_{args.winner}_meta.json").write_text(json.dumps({
            "variant": args.winner,
            "prompt": variant["prompt"],
            "frames_processed": len(summary),
            "baseline_run": args.baseline_run,
            "model": args.model_id,
            "device": device,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n輸出: {out_jsonl}")
        return 0

    # ── Sample test ───────────────────────────────────────────────────────────
    sample_fp = fp_pool[:args.sample_fp]
    sample_ok = ok_pool[:args.sample_ok]
    sample = [("FP", r) for r in sample_fp] + [("OK", r) for r in sample_ok]

    print(f"=== Mask Prompt Variant Test (sample: {len(sample_fp)} FP + {len(sample_ok)} OK) ===")
    print(f"Variants: {[v['name'] for v in VARIANTS]}\n")

    model, _ = load_model(args.model_id, args.revision)
    results = []

    for idx, (grp, entry) in enumerate(sample, 1):
        video = entry["video"]
        t_sec = float(entry["time_sec"])
        vp = data_dir / video
        if not vp.exists():
            print(f"[skip] {video}")
            continue
        try:
            frame = extract_frame(vp, t_sec)
        except Exception as e:
            print(f"[skip] {video} @ {t_sec}s: {e}")
            continue

        enc = model.encode_image(frame)
        labels: dict[str, str] = {}
        print(f"[{idx}/{len(sample)}] [{grp}] {video[:40]} @ {t_sec}s")
        for v in VARIANTS:
            answer = model.query(enc, v["prompt"])["answer"]
            label  = classify(answer)
            labels[v["name"]] = label
            print(f"  {v['name']}: {label}  ({answer[:60]})")
        results.append({"video": video, "time_sec": t_sec, "group": grp, "labels": labels})
        print()

    print("=" * 80)
    print_sample_table(results)
    print("下一步：選最佳 variant，執行：")
    print("  conda run -n moondream python scripts/run_mask_prompt_variants.py \\")
    print("      --full-run --winner <variant_name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
