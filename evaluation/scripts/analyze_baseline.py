#!/usr/bin/env python3
"""
分析負樣本基線測試結果，產出 baseline_report.md 統計報告。

用法：
  python scripts/analyze_baseline.py --run 20260415T120000Z
  python scripts/analyze_baseline.py --run latest
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = EVAL_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析基線測試結果，產出報告")
    parser.add_argument(
        "--run",
        default="latest",
        help="run_id（時間戳目錄名）或 'latest' 自動選最新",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="覆蓋輸出目錄（預設寫入該 run 目錄）",
    )
    return parser.parse_args()


def find_run_dir(run_id: str) -> Path:
    if run_id == "latest":
        dirs = sorted(
            (d for d in OUTPUTS_DIR.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
        if not dirs:
            print("[error] outputs/ 內沒有任何測試結果", file=sys.stderr)
            sys.exit(1)
        return dirs[0]
    target = OUTPUTS_DIR / run_id
    if not target.exists():
        print(f"[error] 找不到 run 目錄: {target}", file=sys.stderr)
        sys.exit(1)
    return target


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(run_dir: Path) -> dict:
    meta = load_meta(run_dir / "run_meta.json")
    config_name = meta.get("config_file", "gap_baseline.json")
    config_path = EVAL_DIR / "configs" / config_name
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── 統計函式 ──────────────────────────────────────────────────────

def compute_query_stats(rows: list[dict], config: dict) -> list[dict]:
    stats = []
    for qc in config.get("query_checks", []):
        name = qc["name"]
        col = f"q_{name}_label"
        expected = qc.get("expected_in_normal", "")
        category = qc.get("category", "")

        total = len(rows)
        yes_count = sum(1 for r in rows if r.get(col) == "yes")
        no_count = sum(1 for r in rows if r.get(col) == "no")
        unknown_count = total - yes_count - no_count

        stats.append({
            "name": name,
            "category": category,
            "expected": expected,
            "total": total,
            "yes": yes_count,
            "no": no_count,
            "unknown": unknown_count,
            "fp_rate": (yes_count / total * 100) if total and expected == "no" else None,
        })

    return stats


def compute_worker_gate_stats(rows: list[dict], config: dict) -> list[dict]:
    """只看 worker_visible=yes 的帧，重新算各 query 的 FP 率"""
    gated = [r for r in rows if r.get("q_worker_visible_label") == "yes"]
    if not gated:
        return []

    stats = []
    for qc in config.get("query_checks", []):
        name = qc["name"]
        if name in ("worker_visible", "scene_summary"):
            continue
        expected = qc.get("expected_in_normal", "")
        if expected != "no":
            continue

        col = f"q_{name}_label"
        total = len(gated)
        fp = sum(1 for r in gated if r.get(col) == "yes")

        stats.append({
            "name": name,
            "total_gated": total,
            "fp_gated": fp,
            "fp_rate_gated": (fp / total * 100) if total else 0,
        })

    return stats


def compute_detect_stats(rows: list[dict], config: dict) -> list[dict]:
    stats = []
    for dt in config.get("detect_targets", []):
        name = dt["name"]
        col = f"d_{name.replace(' ', '_')}_count"
        category = dt.get("category", "")

        total = len(rows)
        detected = sum(1 for r in rows if int(r.get(col, 0)) > 0)
        total_boxes = sum(int(r.get(col, 0)) for r in rows)

        stats.append({
            "name": name,
            "category": category,
            "total_frames": total,
            "frames_with_detection": detected,
            "total_boxes": total_boxes,
            "detection_rate": (detected / total * 100) if total else 0,
        })

    return stats


def compute_caption_alerts(rows: list[dict]) -> list[dict]:
    alerts = []
    for r in rows:
        kws = r.get("caption_alert_keywords", "")
        if kws:
            alerts.append({
                "video": r["video"],
                "time_sec": r["time_sec"],
                "keywords": kws,
                "caption": r.get("caption", ""),
            })
    return alerts


# ── 報告產生 ──────────────────────────────────────────────────────

def generate_report(
    meta: dict,
    query_stats: list[dict],
    gate_stats: list[dict],
    detect_stats: list[dict],
    caption_alerts: list[dict],
    total_rows: int,
) -> str:
    lines = []
    lines.append("# 負樣本基線測試報告\n")
    lines.append(f"**Run ID**: `{meta.get('run_id', 'N/A')}`  ")
    lines.append(f"**日期**: {meta.get('created_at', 'N/A')[:10]}  ")
    lines.append(f"**模型**: `{meta.get('model', 'N/A')}` @ `{meta.get('revision', 'N/A')}`  ")
    lines.append(f"**裝置**: {meta.get('device', 'N/A')}  ")
    lines.append(f"**抽幀間隔**: {meta.get('sample_every_sec', 'N/A')}s  ")
    lines.append(f"**總幀數**: {total_rows}  ")
    lines.append(f"**總耗時**: {meta.get('total_time_sec', 'N/A')}s  ")
    lines.append(f"**影片**: {', '.join(meta.get('videos', []))}\n")

    lines.append("---\n")
    lines.append("## 測試說明\n")
    lines.append("本測試使用**全部為正常作業**的影片。任何被模型判定為異常（label=yes）的結果，")
    lines.append("在已知全為正常的前提下，即為 **False Positive（誤報）**。\n")
    lines.append("目的：建立 Moondream 在 YOLO Gap 場景的誤報基線，供後續正樣本測試對照。\n")

    # ── Query FP 統計 ────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## YOLO Gap 場景 — Query 誤報率\n")
    lines.append("| 場景 | 類別 | 預期 | Yes | No | Unknown | FP率 | 判定 |")
    lines.append("|------|------|------|-----|-----|---------|------|------|")

    for s in query_stats:
        if s["expected"] not in ("yes", "no"):
            continue
        fp_str = f"{s['fp_rate']:.1f}%" if s["fp_rate"] is not None else "N/A"
        if s["expected"] == "no":
            if s["fp_rate"] is not None and s["fp_rate"] == 0:
                verdict = "OK"
            elif s["fp_rate"] is not None and s["fp_rate"] <= 5:
                verdict = "低風險"
            elif s["fp_rate"] is not None and s["fp_rate"] <= 15:
                verdict = "需規則保護"
            else:
                verdict = "高風險"
        else:
            hit = s["yes"]
            pct = (hit / s["total"] * 100) if s["total"] else 0
            fp_str = f"{pct:.1f}% hit"
            verdict = "基準OK" if pct >= 80 else "偏低"

        lines.append(
            f"| {s['name']} | {s['category']} | {s['expected']} "
            f"| {s['yes']} | {s['no']} | {s['unknown']} "
            f"| {fp_str} | {verdict} |"
        )

    lines.append("")

    # ── Worker Gate 效果 ─────────────────────────────────────────
    if gate_stats:
        lines.append("---\n")
        lines.append("## Worker Gate 效果（只看有人的幀）\n")
        lines.append("幻覺最常發生在無人畫面。以下對比加入 `worker_visible=yes` gate 前後的 FP 率：\n")
        lines.append("| 場景 | 無 Gate FP | 有 Gate FP | 變化 |")
        lines.append("|------|-----------|-----------|------|")

        q_map = {s["name"]: s for s in query_stats}
        for g in gate_stats:
            name = g["name"]
            qs = q_map.get(name, {})
            raw_fp = qs.get("fp_rate")
            gated_fp = g["fp_rate_gated"]
            if raw_fp is not None and raw_fp > 0:
                change = f"{((gated_fp - raw_fp) / raw_fp * 100):.0f}%"
            else:
                change = "—"
            raw_str = f"{qs.get('yes', '?')}/{qs.get('total', '?')} ({raw_fp:.1f}%)" if raw_fp is not None else "N/A"
            gated_str = f"{g['fp_gated']}/{g['total_gated']} ({gated_fp:.1f}%)"
            lines.append(f"| {name} | {raw_str} | {gated_str} | {change} |")

        lines.append("")

    # ── Detect 統計 ──────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Detect 物件統計\n")
    lines.append("| 目標 | 類別 | 有框幀數 | 總框數 | 檢出率 | 備註 |")
    lines.append("|------|------|---------|--------|--------|------|")

    for s in detect_stats:
        note = ""
        if s["category"] == "yolo_gap":
            note = "Gap: 正常畫面應為 0" if s["frames_with_detection"] == 0 else "FP 注意"
        elif s["category"] == "positive_baseline":
            note = "正面基準"
        lines.append(
            f"| {s['name']} | {s['category']} "
            f"| {s['frames_with_detection']}/{s['total_frames']} "
            f"| {s['total_boxes']} "
            f"| {s['detection_rate']:.1f}% "
            f"| {note} |"
        )

    lines.append("")

    # ── Caption 警示 ─────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Caption 警示關鍵詞\n")
    if caption_alerts:
        lines.append(f"共 {len(caption_alerts)} 幀的 caption 中出現警示關鍵詞：\n")
        for a in caption_alerts[:10]:
            lines.append(f"- **{a['video']}** @ {a['time_sec']}s: `{a['keywords']}`")
            lines.append(f"  > {a['caption'][:120]}...")
        if len(caption_alerts) > 10:
            lines.append(f"\n（另有 {len(caption_alerts) - 10} 筆，詳見 summary.csv）")
    else:
        lines.append("正常畫面中未出現任何警示關鍵詞。\n")

    # ── 結論 ─────────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 結論與建議\n")

    safe_gaps = [s for s in query_stats if s["expected"] == "no" and s.get("fp_rate") is not None and s["fp_rate"] == 0]
    risky_gaps = [s for s in query_stats if s["expected"] == "no" and s.get("fp_rate") is not None and s["fp_rate"] > 5]
    low_risk = [s for s in query_stats if s["expected"] == "no" and s.get("fp_rate") is not None and 0 < s["fp_rate"] <= 5]

    if safe_gaps:
        names = ", ".join(s["name"] for s in safe_gaps)
        lines.append(f"**FP=0（可推進）**: {names}\n")
    if low_risk:
        names = ", ".join(s["name"] for s in low_risk)
        lines.append(f"**低 FP（加規則可用）**: {names}\n")
    if risky_gaps:
        names = ", ".join(s["name"] for s in risky_gaps)
        lines.append(f"**高 FP（需正樣本驗證 + 規則保護）**: {names}\n")

    lines.append("\n### 下一步\n")
    lines.append("1. 收集異常正樣本，做正負對比測試（策略 B/C）")
    lines.append("2. 對高 FP 場景加入 Worker Gate + 規則引擎")
    lines.append("3. 與 YOLO 做同圖對照測試")

    return "\n".join(lines)


# ── 主流程 ────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    run_dir = find_run_dir(args.run)

    csv_path = run_dir / "summary.csv"
    meta_path = run_dir / "run_meta.json"

    if not csv_path.exists():
        print(f"[error] 找不到 summary.csv: {csv_path}", file=sys.stderr)
        return 1
    if not meta_path.exists():
        print(f"[error] 找不到 run_meta.json: {meta_path}", file=sys.stderr)
        return 1

    rows = load_csv(csv_path)
    meta = load_meta(meta_path)
    config = load_config(run_dir)

    if not rows:
        print("[error] summary.csv 是空的", file=sys.stderr)
        return 1

    print(f"[analyze] Run: {run_dir.name}  共 {len(rows)} 幀\n")

    query_stats = compute_query_stats(rows, config)
    gate_stats = compute_worker_gate_stats(rows, config)
    detect_stats = compute_detect_stats(rows, config)
    caption_alerts = compute_caption_alerts(rows)

    report = generate_report(meta, query_stats, gate_stats, detect_stats, caption_alerts, len(rows))

    out_dir = Path(args.output_dir) if args.output_dir else run_dir
    report_path = out_dir / "baseline_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\n{'='*50}")
    print(f"報告已儲存: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
