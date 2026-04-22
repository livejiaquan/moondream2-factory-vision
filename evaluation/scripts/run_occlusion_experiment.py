#!/usr/bin/env python3
"""
遮擋／角度實驗：在「有人但安全帽或面罩偵測／問答不確定」的幀上，
額外問 Moondream：是否主要因視角、遮擋、距離導致看不清楚，而非真的沒戴。

資料來源：既有 baseline 的 results.jsonl（同一批抽幀時間點），從原始影片擷取畫面（無標註框）。

用法：
  cd evaluation
  python scripts/run_occlusion_experiment.py --baseline-run 20260415T102259Z

輸出：outputs/<baseline_run>_occlusion/
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = EVAL_DIR / "outputs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-run", required=True, help="例如 20260415T102259Z")
    p.add_argument("--data", default=str(PROJECT_ROOT / "data"))
    p.add_argument(
        "--config",
        default=str(EVAL_DIR / "configs" / "occlusion_queries.json"),
    )
    p.add_argument("--model-id", default="vikhyatk/moondream2")
    p.add_argument("--revision", default="2025-01-09")
    return p.parse_args()


def classify_yes_no(text: str) -> str:
    t = text.strip().lower()
    if re.match(r"^(yes\b|yes[,.!\s])", t):
        return "yes"
    if re.match(r"^(no\b|no[,.!\s])", t):
        return "no"
    return "unknown"


def load_model(model_id: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()
    return model, device


def extract_frame(video_path: Path, time_sec: float):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame at {time_sec}s")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def should_include(entry: dict) -> bool:
    r = entry["result"]
    pc = r["detections"]["person"]["count"]
    if pc == 0:
        return False
    hc = r["detections"]["helmet"]["count"]
    mh = r["queries"]["missing_helmet"]["label"]
    mf = r["queries"]["missing_face_mask"]["label"]
    return hc == 0 or mh == "yes" or mf == "yes"


def main():
    args = parse_args()
    baseline_dir = OUTPUTS / args.baseline_run
    jsonl_path = baseline_dir / "results.jsonl"
    if not jsonl_path.exists():
        print(f"[error] 找不到 {jsonl_path}", file=sys.stderr)
        return 1

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    queries = cfg["queries"]

    entries = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    selected = [e for e in entries if should_include(e)]
    data_dir = Path(args.data)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUTPUTS / f"{args.baseline_run}_occlusion" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 遮擋／角度實驗 ===")
    print(f"  Baseline: {args.baseline_run}")
    print(f"  符合條件的幀: {len(selected)} / {len(entries)}")
    print(f"  輸出: {out_dir}\n")

    model, device = load_model(args.model_id, args.revision)
    print(f"  裝置: {device}\n")

    rows = []
    jsonl_out = out_dir / "occlusion_results.jsonl"

    with jsonl_out.open("w", encoding="utf-8") as jf:
        for i, entry in enumerate(selected, 1):
            vid_name = entry["video"]
            t_sec = entry["time_sec"]
            video_path = data_dir / vid_name
            if not video_path.exists():
                print(f"[skip] 找不到影片: {video_path}")
                continue

            try:
                pil = extract_frame(video_path, t_sec)
            except Exception as ex:
                print(f"[skip] {vid_name} @ {t_sec}s: {ex}")
                continue

            enc = model.encode_image(pil)
            r_prev = entry["result"]
            hc = r_prev["detections"]["helmet"]["count"]
            mh = r_prev["queries"]["missing_helmet"]["label"]
            mf = r_prev["queries"]["missing_face_mask"]["label"]

            q_out = {}
            t0 = time.time()
            for q in queries:
                ans = model.query(enc, q["prompt"])["answer"]
                q_out[q["name"]] = {
                    "prompt": q["prompt"],
                    "answer": ans,
                    "label": classify_yes_no(ans),
                }
            elapsed = round(time.time() - t0, 2)

            rec = {
                "video": vid_name,
                "time_sec": t_sec,
                "baseline": {
                    "detect_helmet_count": hc,
                    "missing_helmet_label": mh,
                    "missing_face_mask_label": mf,
                },
                "occlusion_queries": q_out,
                "runtime_sec": elapsed,
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            h_lim = q_out["helmet_visibility_limited"]["label"]
            m_lim = q_out["mask_visibility_limited"]["label"]
            print(f"[{i}/{len(selected)}] {vid_name[:20]}... @{t_sec}s  "
                  f"helmet_box={hc} miss_h={mh} miss_m={mf}  "
                  f"→ vis_helmet_limited={h_lim} vis_mask_limited={m_lim} ({elapsed}s)")

            rows.append({
                "video": vid_name,
                "time_sec": t_sec,
                "detect_helmet_count": hc,
                "missing_helmet": mh,
                "missing_face_mask": mf,
                "helmet_visibility_limited": h_lim,
                "mask_visibility_limited": m_lim,
                "helmet_vis_answer": q_out["helmet_visibility_limited"]["answer"][:200],
                "mask_vis_answer": q_out["mask_visibility_limited"]["answer"][:200],
            })

    if rows:
        csv_path = out_dir / "occlusion_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    meta = {
        "experiment": "occlusion_angle_visibility",
        "baseline_run": args.baseline_run,
        "source_jsonl": str(jsonl_path),
        "frames_processed": len(rows),
        "model": args.model_id,
        "revision": args.revision,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    report = _build_report_md(rows, entries)
    (out_dir / "occlusion_report.md").write_text(report, encoding="utf-8")

    print(f"\n完成。輸出: {out_dir}")
    print(report)
    return 0


def _build_report_md(rows: list[dict], all_entries: list[dict]) -> str:
    if not rows:
        return "# 遮擋實驗\n\n無有效幀。\n"

    mh_fp_baseline = sum(
        1
        for e in all_entries
        if e["result"]["queries"]["missing_helmet"]["label"] == "yes"
        and e["result"]["detections"]["person"]["count"] > 0
    )
    mf_fp_baseline = sum(
        1
        for e in all_entries
        if e["result"]["queries"]["missing_face_mask"]["label"] == "yes"
        and e["result"]["detections"]["person"]["count"] > 0
    )

    def sim_helmet_alarm(r):
        if r["missing_helmet"] != "yes":
            return False
        if r["helmet_visibility_limited"] == "yes":
            return False
        return True

    def sim_mask_alarm(r):
        if r["missing_face_mask"] != "yes":
            return False
        if r["mask_visibility_limited"] == "yes":
            return False
        return True

    mh_after = sum(1 for r in rows if sim_helmet_alarm(r))
    mf_after = sum(1 for r in rows if sim_mask_alarm(r))

    mh_in_rows_yes = sum(1 for r in rows if r["missing_helmet"] == "yes")
    mf_in_rows_yes = sum(1 for r in rows if r["missing_face_mask"] == "yes")

    h_yes = sum(1 for r in rows if r["helmet_visibility_limited"] == "yes")
    m_yes = sum(1 for r in rows if r["mask_visibility_limited"] == "yes")

    helmet_contra = sum(
        1 for r in rows if int(r["detect_helmet_count"]) > 0 and r["helmet_visibility_limited"] == "yes"
    )

    no_helmet_box = [r for r in rows if int(r["detect_helmet_count"]) == 0]

    def still_alarm_helmet_tight(r):
        """保守規則：僅在 detect helmet=0 時，用可見性降級；已框到帽則不套用可見性覆寫。"""
        if r["missing_helmet"] != "yes":
            return False
        if int(r["detect_helmet_count"]) > 0:
            # 已框到安全帽仍答 missing → 矛盾，不應只靠「可見性」結案，維持需關注／人工複核
            return True
        # 沒框到帽 + 說沒戴：若可見性受限 → 降級不告警
        return r["helmet_visibility_limited"] != "yes"

    mh_after_tight = sum(1 for r in rows if still_alarm_helmet_tight(r))
    no_box_mh_yes = sum(
        1 for r in rows if int(r["detect_helmet_count"]) == 0 and r["missing_helmet"] == "yes"
    )

    lines = [
        "# 遮擋／角度實驗報告",
        "",
        "## 目的",
        "",
        "當 YOLO 或 `detect helmet` 沒框到、或 `missing_*` query 為 Yes 時，",
        "可能是**真的沒戴**，也可能是**背對、遮擋、遠景、畫質**導致看不清楚。",
        "本實驗在相同畫面上額外詢問：是否**主要因可見性受限**而非清楚顯示未配戴。",
        "",
        "## 額外詢問的兩個問題（英文 prompt）",
        "",
        "1. **helmet_visibility_limited**：安全帽是否因角度／遮擋／距離等而難以看清，而非畫面清楚顯示沒戴？",
        "2. **mask_visibility_limited**：面罩／防毒面具是否因類似原因難以看清？",
        "",
        "若回答 **Yes** → 可與規則結合，將事件降級為「可見性不足／不確定」而非直接違規。",
        "",
        "## 樣本範圍",
        "",
        f"- 本實際跑在 baseline 中：**有人**且（`detect helmet=0` 或 `missing_helmet=yes` 或 `missing_face_mask=yes`）的幀，共 **{len(rows)}** 幀。",
        f"- 全資料有人幀中，baseline `missing_helmet=yes`：**{mh_fp_baseline}** 幀；`missing_face_mask=yes`：**{mf_fp_baseline}** 幀。",
        "",
        f"## 原始回答統計（本批 {len(rows)} 幀）",
        "",
        f"- `helmet_visibility_limited` = yes：**{h_yes}/{len(rows)}**",
        f"- `mask_visibility_limited` = yes：**{m_yes}/{len(rows)}**",
        f"- **矛盾檢查**：已框到安全帽（`detect helmet`≥1）但 `helmet_visibility_limited`=yes 的幀：**{helmet_contra}/{len(rows)}**",
        "",
        "若「可見性受限」在**已框到安全帽**時仍大量為 Yes，代表此問法**過寬**或模型慣性答 Yes，**不適合單獨當降級條件**，需改 prompt 或加上 `detect helmet=0` 等前置條件。",
        "",
        "## 規則模擬 A（寬鬆）：missing=yes 且 visibility≠yes 才告警",
        "",
        f"| 指標 | 實驗子集內 baseline | 規則後仍告警 |",
        f"|------|---------------------|--------------|",
        f"| missing_helmet=yes | {mh_in_rows_yes} | {mh_after} |",
        f"| missing_face_mask=yes | {mf_in_rows_yes} | {mf_after} |",
        "",
        f"本批結果：若全為 0，表示幾乎所有幀都被判成「可見性受限」— **須對照上方矛盾檢查**，避免誤以為已解決 YOLO gap。",
        "",
        "## 規則模擬 B（保守）：僅在 `detect helmet=0` 時用可見性降級；已框到帽不套用",
        "",
        f"- `detect helmet=0` 的幀：**{len(no_helmet_box)}**；其中 `missing_helmet=yes`：**{no_box_mh_yes}**",
        f"- 套用規則後仍視為「需關注／告警」的幀（含：已框到帽但 missing=yes 的矛盾列）：**{mh_after_tight}**",
        "",
        "> **建議**：可見性問題僅在 **YOLO／detect 也框不到** 時與「遮擋／角度」敘述一併使用；若已框到帽，就不應再以「看不清」覆蓋。",
        "",
        "## 解讀",
        "",
        "- Moondream **可以**用自然語言描述「角度、遮擋、距離」— 這是 YOLO 沒有的敘述能力。",
        "- 但 **單一 Yes/No 問題容易過寬或與 detect 矛盾**，實務上應：**Worker Gate → detect → 再問可見性（僅在框不到時）→ 連續幀**。",
        "- 本實驗**不取代**真實標註；若要宣稱「提升」需標註 ground truth（是否真的未戴／是否遮擋）。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
