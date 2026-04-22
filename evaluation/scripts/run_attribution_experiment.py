#!/usr/bin/env python3
"""
誤判歸因實驗（在「現場工人應已配戴」的假設下）

問題：在有人、且 baseline 出現「沒框到帽／missing 說沒戴」等誤判時，
有多少可歸因於「影像上看不清楚」（背面、遮擋、遠景），
又有多少是模型仍「堅稱能清楚看到未配戴」？

使用較嚴格的問法（attribution_queries.json），輸出可量化表格。

用法：
  python scripts/run_attribution_experiment.py --baseline-run 20260415T102259Z

輸出：outputs/<baseline_run>_attribution/<timestamp>/
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
    p.add_argument("--baseline-run", required=True)
    p.add_argument("--data", default=str(PROJECT_ROOT / "data"))
    p.add_argument(
        "--config",
        default=str(EVAL_DIR / "configs" / "attribution_queries.json"),
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
        raise RuntimeError(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("read")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def should_include(entry: dict) -> bool:
    r = entry["result"]
    if r["detections"]["person"]["count"] == 0:
        return False
    hc = r["detections"]["helmet"]["count"]
    mh = r["queries"]["missing_helmet"]["label"]
    mf = r["queries"]["missing_face_mask"]["label"]
    return hc == 0 or mh == "yes" or mf == "yes"


def main():
    args = parse_args()
    baseline_dir = OUTPUTS / args.baseline_run
    jsonl_in = baseline_dir / "results.jsonl"
    if not jsonl_in.exists():
        sys.exit(f"[error] {jsonl_in}")

    cfg = json.loads(Path(args.config).read_text("utf-8"))
    queries = cfg["queries"]

    entries = [json.loads(l) for l in jsonl_in.read_text().splitlines() if l.strip()]
    selected = [e for e in entries if should_include(e)]
    data_dir = Path(args.data)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUTPUTS / f"{args.baseline_run}_attribution" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 誤判歸因實驗 ===")
    print(f"  假設：現場工人應已配戴（合規）。")
    print(f"  分析幀：有人 且（detect helmet=0 或 missing_helmet 或 missing_face_mask）→ {len(selected)} 幀\n")

    model, device = load_model(args.model_id, args.revision)
    print(f"  device={device}\n")

    rows_out = []
    jsonl_path = out_dir / "attribution_results.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for i, entry in enumerate(selected, 1):
            vid = entry["video"]
            t_sec = entry["time_sec"]
            vp = data_dir / vid
            if not vp.exists():
                print(f"[skip] {vid}")
                continue
            try:
                pil = extract_frame(vp, t_sec)
            except Exception as ex:
                print(f"[skip] {vid} @ {t_sec}: {ex}")
                continue

            enc = model.encode_image(pil)
            r0 = entry["result"]
            hc = r0["detections"]["helmet"]["count"]
            mh = r0["queries"]["missing_helmet"]["label"]
            mf = r0["queries"]["missing_face_mask"]["label"]

            qres = {}
            t0 = time.time()
            for q in queries:
                ans = model.query(enc, q["prompt"])["answer"]
                qres[q["name"]] = {"answer": ans, "label": classify_yes_no(ans)}
            elapsed = round(time.time() - t0, 2)

            rec = {
                "video": vid,
                "time_sec": t_sec,
                "baseline": {
                    "detect_helmet_count": hc,
                    "missing_helmet": mh,
                    "missing_face_mask": mf,
                },
                "attribution": qres,
                "runtime_sec": elapsed,
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            ch = qres["clear_absence_helmet"]["label"]
            hv = qres["head_region_visibility_limited"]["label"]
            cm = qres["clear_absence_mask"]["label"]
            print(f"[{i}/{len(selected)}] @{t_sec}s  det_h={hc} m_h={mh} m_m={mf}  |  clearNoHelmet={ch} visLim={hv} clearNoMask={cm} ({elapsed}s)")

            rows_out.append({
                "video": vid,
                "time_sec": t_sec,
                "detect_helmet_count": hc,
                "missing_helmet": mh,
                "missing_face_mask": mf,
                "clear_absence_helmet": ch,
                "head_region_visibility_limited": hv,
                "clear_absence_mask": cm,
            })

    if rows_out:
        with (out_dir / "attribution_summary.csv").open("w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    report = _build_report_md(rows_out, cfg)
    (out_dir / "attribution_report.md").write_text(report, encoding="utf-8")

    meta = {
        "experiment": "misdetection_attribution",
        "baseline_run": args.baseline_run,
        "frames": len(rows_out),
        "config": str(args.config),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成: {out_dir}\n")
    print(report)
    return 0


def _build_report_md(rows: list[dict], cfg: dict) -> str:
    n = len(rows)
    if n == 0:
        return "# 歸因實驗\n無資料。\n"

    def filt(pred):
        return [r for r in rows if pred(r)]

    only_no_det = filt(lambda r: int(r["detect_helmet_count"]) == 0)
    only_miss_q = filt(lambda r: r["missing_helmet"] == "yes")
    both = filt(lambda r: int(r["detect_helmet_count"]) == 0 and r["missing_helmet"] == "yes")

    lines = [
        "# 誤判歸因實驗報告",
        "",
        "## 前提（與本專案測試一致）",
        "",
        "- 本批影片視為 **正常合規作業**：現場工人 **應已配戴** 安全帽與必要面罩。",
        "- 因此 baseline 出現的 `missing_*=yes` 或 `detect helmet=0`，先視為 **系統誤判**，再問：誤判比較像 **影像條件** 還是 **模型仍堅稱能清楚看到違規**？",
        "",
        "## 使用的問法（較嚴格）",
        "",
    ]
    for q in cfg.get("queries", []):
        lines.append(f"### `{q['name']}`")
        lines.append("")
        lines.append(f"> {q['prompt']}")
        lines.append("")

    lines.extend([
        "## 樣本",
        "",
        f"- 分析幀數 **{n}**（有人，且至少一項：未框到帽／missing 安全帽／missing 面罩）。",
        f"- 其中僅 **detect helmet=0**：{len(only_no_det)} 幀。",
        f"- 其中僅 **missing_helmet=yes**：{len(only_miss_q)} 幀。",
        f"- 兩者同時：**{len(both)}** 幀。",
        "",
        "---",
        "",
        "## 核心指標（安全帽相關）",
        "",
        "定義（可重現）：",
        "",
        "- **可归因於影像／可見性**（較適合降級、不當成「清楚違規」）：",
        "  - `head_region_visibility_limited` = **yes**（頭部／帽區主要看不清楚），**或**",
        "  - `clear_absence_helmet` = **no**（模型承認無法「清楚、無歧義」地看到未戴帽）。",
        "",
        "- **模型仍像「堅稱清楚看到未戴帽」**（較像模型判讀問題，與合規假設衝突）：",
        "  - `clear_absence_helmet` = **yes** 且 `head_region_visibility_limited` = **no**。",
        "",
    ])

    image_expl = sum(
        1
        for r in rows
        if r["head_region_visibility_limited"] == "yes" or r["clear_absence_helmet"] == "no"
    )
    model_insist = sum(
        1
        for r in rows
        if r["clear_absence_helmet"] == "yes" and r["head_region_visibility_limited"] == "no"
    )
    unknown = max(0, n - image_expl - model_insist)

    admit_unclear = sum(1 for r in rows if r["clear_absence_helmet"] == "no")
    claims_clear_no_helmet = sum(1 for r in rows if r["clear_absence_helmet"] == "yes")
    vis_all_yes = sum(1 for r in rows if r["head_region_visibility_limited"] == "yes")

    lines.extend([
        "### 指標 A：模型是否承認「無法清楚斷定未戴帽」",
        "",
        "只看 **`clear_absence_helmet`**（與 `head_region` 獨立，較不受「可見性題全選 yes」影響）：",
        "",
        f"- 回答 **No**（依題意：無法清楚、無歧義地看到「沒戴帽」）→ **{admit_unclear} / {n}**（**{100*admit_unclear/n:.1f}%**）→ 可解讀為：**誤判／不確定較可能與影像條件有關**，適合降級為「無法判定」而非「違規」。",
        f"- 回答 **Yes**（模型聲稱能清楚看到未戴帽）→ **{claims_clear_no_helmet} / {n}**（**{100*claims_clear_no_helmet/n:.1f}%**）→ 在「實際有戴」的假設下，較像 **模型與現場認知衝突**，需規則或複核。",
        "",
        "### 指標 B：頭部可見性題 + 與 A 交叉（若可見性題有效時）",
        "",
        f"- `head_region_visibility_limited`=yes：**{vis_all_yes} / {n}**（若接近 100%，表示此題可能過寬，不宜單獨當主指標。）",
        "",
        f"| 分類（OR 定義） | 幀數 | 佔比 |",
        f"|------|------|------|",
        f"| 可归因影像／可見性（vis=yes 或 clear_absence=no） | {image_expl} | {100*image_expl/n:.1f}% |",
        f"| 模型：clear_absence=yes 且 vis=no（堅稱清楚違規且頭可辨） | {model_insist} | {100*model_insist/n:.1f}% |",
        f"| 其它（含 unknown） | {unknown} | {100*unknown/n:.1f}% |",
        "",
    ])
    if vis_all_yes >= n * 0.9:
        lines.extend([
            "> **注意**：本批 `head_region_visibility_limited` 幾乎全為 Yes，與「已框到安全帽」的幀並存時會顯得矛盾。**建議以指標 A（`clear_absence_helmet`）為主** 估計「有多少誤判可改標為不確定」。",
            "",
        ])
    lines.extend([
        f"**粗估可改善比例**：若告警僅在 `clear_absence_helmet=yes` 時升級，其餘改標「不確定／不告警」，則在這 **{n}** 幀子集內，約 **{100*admit_unclear/n:.1f}%** 幀可被歸為「模型承認無法清楚指認未戴帽」（實際須與業務規則對齊）。",
        "",
    ])

    if only_no_det:
        ne = len(only_no_det)
        ie = sum(
            1
            for r in only_no_det
            if r["head_region_visibility_limited"] == "yes" or r["clear_absence_helmet"] == "no"
        )
        lines.extend([
            "### 子集合：僅 detect 未框到安全帽（可能與 YOLO 漏檢重疊）",
            "",
            f"- 幀數 {ne}。其中「可归因影像」：**{ie}**（{100*ie/ne:.1f}%）。",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 面罩相關（補充）",
        "",
        "- `clear_absence_mask` = **no**：模型承認無法清楚斷定未戴面罩（臉不清楚／背對等）→ 與 `missing_face_mask=yes` 同時出現時，誤判較可能與 **臉部可見性** 有關。",
        "",
    ])
    mask_no_clear = sum(1 for r in rows if r["clear_absence_mask"] == "no")
    mask_yes_clear = sum(1 for r in rows if r["clear_absence_mask"] == "yes")
    lines.extend([
        f"- 本批 {n} 幀中：`clear_absence_mask`=no → **{mask_no_clear}**（{100*mask_no_clear/n:.1f}%）；=yes → **{mask_yes_clear}**（{100*mask_yes_clear/n:.1f}%）。",
        "",
        "---",
        "",
        "## 限制",
        "",
        "- 合規假設來自業務情境，非像素級標註。",
        "- Moondream 回答仍有隨機性；可改 prompt 或做多幀投票。",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
