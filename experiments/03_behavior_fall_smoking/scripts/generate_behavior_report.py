#!/usr/bin/env python3
"""Generate a Chinese HTML report for the fall/smoking behavior experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "behavior_fall_smoking.json"


KEY_QUERY_BY_EVENT = {
    "fall": ["fall_direct", "fall_strict"],
    "smoking": ["smoking_direct", "smoking_strict"],
}

LABEL_TEXT = {
    "yes": "是",
    "no": "否",
    "unclear": "不確定",
    "unknown": "未知",
    "open": "開放回答",
    "not_run": "未執行",
    "missing": "無資料",
}

EVENT_TEXT = {
    "fall": "跌倒",
    "smoking": "抽菸",
    "abnormal": "異常行為",
    "evidence": "開放式畫面證據",
}

QUERY_TEXT = {
    "fall_direct": "跌倒 direct",
    "fall_strict": "跌倒 strict",
    "smoking_direct": "抽菸 direct",
    "smoking_strict": "抽菸 strict",
    "abnormal_direct": "泛用異常",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate behavior experiment HTML report.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def count_labels(rows: list[dict], query_name: str) -> Counter:
    counts = Counter()
    for row in rows:
        label = row["result"]["queries"].get(query_name, {}).get("label", "not_run")
        counts[label] += 1
    return counts


def query_available(rows: list[dict], query_name: str) -> int:
    return sum(1 for row in rows if query_name in row["result"].get("queries", {}))


def pct(num: int, den: int) -> str:
    return f"{num / den * 100:.0f}%" if den else "0%"


def method_hit_count(rows: list[dict], query_name: str) -> int:
    return count_labels(rows, query_name)["yes"]


def evidence_signal(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(k.lower())}\b", lower) for k in keywords)


def classify_open_evidence(row: dict, event: str, config: dict) -> bool:
    kws = config.get("open_evidence_keywords", {}).get(event, [])
    q = row["result"]["queries"].get("scene_evidence", {})
    text = " ".join([row["result"].get("caption", ""), q.get("answer", "")])
    return evidence_signal(text, kws)


def open_evidence_status(row: dict, event: str, config: dict) -> str:
    q = row["result"].get("queries", {}).get("scene_evidence")
    caption = row["result"].get("caption", "")
    if not q and not caption:
        return "not_run"
    return "yes" if classify_open_evidence(row, event, config) else "no"


def by_video(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    for items in grouped.values():
        items.sort(key=lambda x: float(x["time_sec"]))
    return dict(sorted(grouped.items()))


def badge(label: str) -> str:
    label = label or "not_run"
    cls = {
        "yes": "yes",
        "no": "no",
        "unclear": "unclear",
        "unknown": "unknown",
        "open": "open",
        "not_run": "not-run",
    }.get(label, "unknown")
    text = LABEL_TEXT.get(label, label)
    return f'<span class="badge {cls}">{escape(text)}</span>'


def event_text(event: str) -> str:
    return EVENT_TEXT.get(event, event)


def query_text(query_name: str) -> str:
    return QUERY_TEXT.get(query_name, query_name)


def query_prompt(config: dict, query_name: str) -> str:
    for query in config.get("queries", []):
        if query.get("name") == query_name:
            return query.get("prompt", "")
    return ""


def query_method(config: dict, query_name: str) -> str:
    for query in config.get("queries", []):
        if query.get("name") == query_name:
            return query.get("method", "")
    return ""


def short(text: str, n: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "..."


def img_tag(row: dict) -> str:
    return f'<img src="{escape(row["frame_path"])}" alt="{escape(row["frame_id"])}">'


def answer_text(row: dict, query_name: str) -> str:
    q = row["result"].get("queries", {}).get(query_name)
    if not q:
        return "本張未執行此 query。"
    return q.get("answer", "").strip() or "模型沒有輸出文字回答。"


def query_result_line(row: dict, query_name: str) -> str:
    q = row["result"].get("queries", {}).get(query_name)
    if not q:
        return f"""
<div class="result-line">
  <div class="result-label">{escape(query_text(query_name))}</div>
  <div>{badge('not_run')}</div>
  <div class="answer">本張未執行此 query。</div>
</div>
"""
    return f"""
<div class="result-line">
  <div class="result-label">{escape(query_text(query_name))}</div>
  <div>{badge(q.get('label', 'unknown'))}</div>
  <div class="answer">{escape(short(q.get('answer', ''), 260))}</div>
</div>
"""


def frame_card(row: dict, focus_event: str | None = None) -> str:
    q = row["result"].get("queries", {})
    qnames = KEY_QUERY_BY_EVENT.get(focus_event or row["expected_event"], [])
    if not qnames:
        qnames = [name for name in ["fall_direct", "fall_strict", "smoking_direct", "smoking_strict"] if name in row["result"].get("queries", {})]
    result_lines = "".join(query_result_line(row, name) for name in qnames)
    if "abnormal_direct" in row["result"].get("queries", {}):
        result_lines += query_result_line(row, "abnormal_direct")

    evidence = q.get("scene_evidence", {}).get("answer", "")
    caption = row["result"].get("caption", "")
    extra = ""
    if caption:
        extra += f"<p><strong>caption</strong>: {escape(short(caption, 220))}</p>"
    if evidence:
        extra += f"<p><strong>開放式證據</strong>: {escape(short(evidence, 220))}</p>"
    return f"""
<article class="frame-card">
  <div class="thumb">{img_tag(row)}</div>
  <div class="frame-meta">
    <strong>{escape(row['video_id'])}</strong><br>
    來源：{escape(row['source_video'])}<br>
    時間：{float(row['time_sec']):.2f}s · 預期事件：{escape(event_text(row['expected_event']))}
  </div>
  <div class="result-stack">{result_lines}</div>
  {extra}
</article>
"""


frame_card.config = {}


def summary_table(grouped: dict[str, list[dict]], config: dict) -> str:
    rows_html = []
    for video_id, items in grouped.items():
        expected = items[0]["expected_event"]
        n = len(items)
        fall_direct = count_labels(items, "fall_direct")
        fall_strict = count_labels(items, "fall_strict")
        smoke_direct = count_labels(items, "smoking_direct")
        smoke_strict = count_labels(items, "smoking_strict")
        abnormal = count_labels(items, "abnormal_direct")
        abnormal_n = query_available(items, "abnormal_direct")
        open_fall = sum(1 for row in items if open_evidence_status(row, "fall", config) == "yes")
        open_smoke = sum(1 for row in items if open_evidence_status(row, "smoking", config) == "yes")
        open_n = query_available(items, "scene_evidence")
        rows_html.append(
            f"""
<tr>
  <td>{escape(video_id)}</td>
  <td>{escape(event_text(expected))}</td>
  <td>{n}</td>
  <td>{fall_direct['yes']} direct 是 / {fall_strict['yes']} strict 是</td>
  <td>{smoke_direct['yes']} direct 是 / {smoke_strict['yes']} strict 是</td>
  <td>{abnormal['yes']} 是 / 實測 {abnormal_n} 張</td>
  <td>跌倒 {open_fall} / 抽菸 {open_smoke} / 實測 {open_n} 張</td>
  <td>{pct(method_hit_count(items, expected + '_direct'), n) if expected in {'fall', 'smoking'} else 'n/a'}</td>
</tr>
"""
        )
    return f"""
<table>
  <thead>
    <tr>
      <th>影片</th><th>預期事件</th><th>影格數</th>
      <th>跌倒 query</th><th>抽菸 query</th>
      <th>泛用異常 query</th><th>開放式證據</th><th>direct 命中率</th>
    </tr>
  </thead>
  <tbody>{''.join(rows_html)}</tbody>
</table>
	"""


def prompt_table(config: dict) -> str:
    rows = []
    include = [
        ("fall_direct", "direct 問法", "直接問畫面是否已經出現跌倒/倒地。"),
        ("fall_strict", "strict 問法", "要求模型只有清楚看到倒地或倒下姿勢才回答是，模糊時可回答不確定。"),
        ("smoking_direct", "direct 問法", "直接問畫面中是否有人抽菸或拿香菸。"),
        ("smoking_strict", "strict 問法", "要求模型看到嘴邊香菸、可見香菸或煙霧等較明確證據才回答是。"),
        ("abnormal_direct", "泛用異常", "用較寬的問題詢問是否有跌倒或抽菸等不安全事件。"),
        ("scene_evidence", "開放式證據", "不先要求 yes/no，而是請模型描述姿勢與手部/嘴部活動。"),
    ]
    for name, label, note in include:
        prompt = query_prompt(config, name)
        if not prompt:
            continue
        rows.append(
            f"""
<tr>
  <td><code>{escape(name)}</code></td>
  <td>{escape(label)}</td>
  <td>{escape(note)}</td>
  <td class="prompt-cell">{escape(prompt)}</td>
</tr>
"""
        )
    return f"""
<table>
  <thead><tr><th>Query 名稱</th><th>用途</th><th>設計理由</th><th>實際 prompt</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""


def input_table(grouped: dict[str, list[dict]]) -> str:
    rows = []
    for video_id, items in grouped.items():
        rows.append(
            f"""
<tr>
  <td>{escape(video_id)}</td>
  <td>{escape(items[0]['source_video'])}</td>
  <td>{escape(event_text(items[0]['expected_event']))}</td>
  <td>{len(items)}</td>
  <td>{float(items[0]['time_sec']):.1f}s - {float(items[-1]['time_sec']):.1f}s</td>
</tr>
"""
        )
    return f"""
<table>
  <thead><tr><th>報告代號</th><th>來源影片</th><th>預期事件</th><th>抽出影格</th><th>時間範圍</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""


def first_yes_context(items: list[dict], query_name: str, radius: int = 2) -> list[dict]:
    idx = next((i for i, row in enumerate(items) if row["result"]["queries"].get(query_name, {}).get("label") == "yes"), None)
    if idx is None:
        return []
    lo = max(0, idx - radius)
    hi = min(len(items), idx + radius + 1)
    return items[lo:hi]


def timeline_table(items: list[dict], event: str, config: dict) -> str:
    direct, strict = KEY_QUERY_BY_EVENT[event]
    trs = []
    for row in items:
        q = row["result"]["queries"]
        open_label = open_evidence_status(row, event, config)
        trs.append(
            f"""
<tr>
  <td>{float(row['time_sec']):.2f}s</td>
  <td>{badge(q.get(direct, {}).get('label', 'not_run'))}</td>
  <td>{badge(q.get(strict, {}).get('label', 'not_run'))}</td>
  <td>{badge(q.get('abnormal_direct', {}).get('label', 'not_run'))}</td>
  <td>{badge(open_label)}</td>
  <td>{escape(short(answer_text(row, direct), 170))}</td>
  <td>{escape(short(answer_text(row, strict), 190))}</td>
</tr>
"""
        )
    return f"""
<table class="timeline">
  <thead><tr><th>時間</th><th>direct</th><th>strict</th><th>泛用異常</th><th>開放證據</th><th>direct 原始回答</th><th>strict 原始回答</th></tr></thead>
  <tbody>{''.join(trs)}</tbody>
</table>
"""


def video_section(video_id: str, items: list[dict], config: dict) -> str:
    expected = items[0]["expected_event"]
    focus_event = expected if expected in {"fall", "smoking"} else None
    context = []
    if focus_event:
        for qname in KEY_QUERY_BY_EVENT[focus_event]:
            frames = first_yes_context(items, qname)
            if frames:
                cards = "".join(frame_card(row, focus_event) for row in frames)
                context.append(f"<h4>邊界觀察：{escape(query_text(qname))} 第一次判定「是」附近</h4><div class=\"grid\">{cards}</div>")
            else:
                context.append(f"<p class=\"note\">{escape(query_text(qname))} 沒有任何 frame 被判定為「是」。</p>")

    positive = []
    if focus_event:
        direct, strict = KEY_QUERY_BY_EVENT[focus_event]
        positive = [
            row
            for row in items
            if row["result"]["queries"].get(direct, {}).get("label") == "yes"
            or row["result"]["queries"].get(strict, {}).get("label") == "yes"
            or open_evidence_status(row, focus_event, config) == "yes"
        ]
    cards = "".join(frame_card(row, focus_event) for row in (positive or items[:6]))

    return f"""
<section>
  <h2>{escape(video_id)} · 預期事件：{escape(event_text(expected))}</h2>
  <p class="source">來源影片：{escape(items[0]['source_video'])} · 影格數：{len(items)}</p>
  {video_takeaway_html(video_id, items)}
  <h3>逐張影格判斷表</h3>
  <p>下表每一列對應一張抽出的影像，列出該時間點實際得到的 direct / strict 結果，以及模型的原始文字回答。</p>
  {timeline_table(items, focus_event, config) if focus_event else ''}
  {''.join(context)}
  <h3>代表畫面</h3>
  <div class="grid">{cards}</div>
</section>
"""


def write_summary_csv(run_dir: Path, grouped: dict[str, list[dict]], config: dict) -> None:
    rows = []
    for video_id, items in grouped.items():
        n = len(items)
        expected = items[0]["expected_event"]
        rows.append(
            {
                "video_id": video_id,
                "expected_event": expected,
                "frames": n,
                "fall_direct_yes": count_labels(items, "fall_direct")["yes"],
                "fall_direct_queried": query_available(items, "fall_direct"),
                "fall_strict_yes": count_labels(items, "fall_strict")["yes"],
                "fall_strict_queried": query_available(items, "fall_strict"),
                "smoking_direct_yes": count_labels(items, "smoking_direct")["yes"],
                "smoking_direct_queried": query_available(items, "smoking_direct"),
                "smoking_strict_yes": count_labels(items, "smoking_strict")["yes"],
                "smoking_strict_queried": query_available(items, "smoking_strict"),
                "abnormal_direct_yes": count_labels(items, "abnormal_direct")["yes"],
                "abnormal_direct_queried": query_available(items, "abnormal_direct"),
                "open_fall_signal": sum(1 for row in items if open_evidence_status(row, "fall", config) == "yes"),
                "open_smoking_signal": sum(1 for row in items if open_evidence_status(row, "smoking", config) == "yes"),
                "open_evidence_queried": query_available(items, "scene_evidence"),
            }
        )
    with (run_dir / "behavior_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def first_label_time(items: list[dict], query_name: str, label: str = "yes") -> float | None:
    for row in items:
        if row["result"]["queries"].get(query_name, {}).get("label") == label:
            return float(row["time_sec"])
    return None


def fmt_time(value: float | None) -> str:
    return f"{value:.1f}s" if value is not None else "未出現"


def yes_count_text(items: list[dict], query_name: str) -> str:
    counts = count_labels(items, query_name)
    queried = query_available(items, query_name)
    return f"{counts['yes']}/{queried}" if queried else "未執行"


def event_query_names(expected_event: str) -> tuple[str, str]:
    if expected_event == "fall":
        return "fall_direct", "fall_strict"
    return "smoking_direct", "smoking_strict"


def event_name(expected_event: str) -> str:
    return "跌倒" if expected_event == "fall" else "抽菸"


def label_runs(items: list[dict], query_name: str) -> str:
    parts = []
    last_label = None
    start = None
    end = None
    for row in items:
        label = row["result"]["queries"].get(query_name, {}).get("label", "not_run")
        t = float(row["time_sec"])
        if label != last_label:
            if last_label is not None:
                parts.append(f"{fmt_time(start)}-{fmt_time(end)}：{LABEL_TEXT.get(last_label, last_label)}")
            last_label = label
            start = t
        end = t
    if last_label is not None:
        parts.append(f"{fmt_time(start)}-{fmt_time(end)}：{LABEL_TEXT.get(last_label, last_label)}")
    return "；".join(parts)


def video_takeaway_html(video_id: str, items: list[dict]) -> str:
    expected = items[0]["expected_event"]
    if expected == "fall":
        strict_unclear = first_label_time(items, "fall_strict", "unclear")
        strict_yes = first_label_time(items, "fall_strict", "yes")
        direct_yes = first_label_time(items, "fall_direct", "yes")
        strict_counts = count_labels(items, "fall_strict")
        direct_counts = count_labels(items, "fall_direct")
        return f"""
<div class="video-summary">
  <h3>本段小結</h3>
  <p>這段影片的預期事件是跌倒。direct 與 strict 逐幀判斷可用來觀察模型從「未跌倒」轉為「跌倒」的時間點；若出現「不確定」，代表該幀已接近判斷邊界但證據尚未完全穩定。</p>
  <ul>
    <li>direct 統計：是 {direct_counts['yes']} 張、否 {direct_counts['no']} 張、不確定 {direct_counts['unclear']} 張。</li>
    <li>strict 統計：是 {strict_counts['yes']} 張、否 {strict_counts['no']} 張、不確定 {strict_counts['unclear']} 張。</li>
    <li>strict 首次不確定：{fmt_time(strict_unclear)}</li>
    <li>strict 首次判定跌倒：{fmt_time(strict_yes)}</li>
    <li>direct 首次判定跌倒：{fmt_time(direct_yes)}</li>
    <li>direct 時序：{escape(label_runs(items, 'fall_direct'))}</li>
    <li>strict 時序：{escape(label_runs(items, 'fall_strict'))}</li>
  </ul>
</div>
"""

    direct_yes = first_label_time(items, "smoking_direct", "yes")
    strict_yes = first_label_time(items, "smoking_strict", "yes")
    strict_counts = count_labels(items, "smoking_strict")
    extra = ""
    if video_id == "smoking_03":
        extra = "這支影片的 strict 結果較不穩，中間多次回到「否」，代表香菸或嘴部動作在某些影格不夠清楚。"
    else:
        extra = "這支影片在 0.5s 後 direct 與 strict 大多一致判定為抽菸，訊號相對穩定。"
    return f"""
<div class="video-summary">
  <h3>本段小結</h3>
  <p>這段影片的預期事件是抽菸。0.0s 通常還沒有被判定為抽菸，後續才開始出現「是」的結果。{escape(extra)}</p>
  <ul>
    <li>direct 首次判定抽菸：{fmt_time(direct_yes)}</li>
    <li>strict 首次判定抽菸：{fmt_time(strict_yes)}</li>
    <li>strict 統計：是 {strict_counts['yes']} 張、否 {strict_counts['no']} 張、不確定 {strict_counts['unclear']} 張。</li>
  </ul>
</div>
"""


def key_findings_html(grouped: dict[str, list[dict]]) -> str:
    lines = []
    for vid, items in grouped.items():
        if items[0].get("expected_event") != "fall":
            continue
        strict_yes = first_label_time(items, "fall_strict", "yes")
        direct_yes = first_label_time(items, "fall_direct", "yes")
        strict_unclear = first_label_time(items, "fall_strict", "unclear")
        boundary = (
            f"，<strong>{strict_unclear:.1f}s</strong> 是 strict 的邊界不確定點"
            if strict_unclear is not None
            else "，本支影片沒有出現 strict 不確定點"
        )
        lines.append(
            f"{escape(vid)}：跌倒 strict 首次「是」為 <strong>{fmt_time(strict_yes)}</strong>，"
            f"direct 首次「是」為 <strong>{fmt_time(direct_yes)}</strong>{boundary}。"
        )
    for vid, items in grouped.items():
        if items[0].get("expected_event") != "smoking":
            continue
        d = yes_count_text(items, "smoking_direct")
        s = yes_count_text(items, "smoking_strict")
        first_d = first_label_time(items, "smoking_direct", "yes")
        first_s = first_label_time(items, "smoking_strict", "yes")
        if first_d is None:
            lines.append(f"{escape(vid)} 沒有任何抽菸 direct「是」結果。")
        else:
            lines.append(
                f"{escape(vid)}：抽菸 direct 命中 <strong>{d}</strong>，"
                f"strict 命中 <strong>{s}</strong>；direct 首次「是」為 "
                f"<strong>{fmt_time(first_d)}</strong>，strict 首次「是」為 "
                f"<strong>{fmt_time(first_s)}</strong>。"
            )
    items = "".join(f"<li>{line}</li>" for line in lines)
    return f"<ul class=\"finding-list\">{items}</ul>"


def timeline_text(items: list[dict], query_name: str) -> str:
    parts = []
    for row in items:
        label = row["result"]["queries"].get(query_name, {}).get("label", "not_run")
        parts.append(f"{float(row['time_sec']):.1f}s={LABEL_TEXT.get(label, label)}")
    return "、".join(parts)


def write_experiment_record(run_dir: Path, grouped: dict[str, list[dict]], meta: dict) -> None:
    lines = [
        "# 0520 行為辨識實驗紀錄",
        "",
        "## 基本資訊",
        "",
        f"- Run：`{run_dir.name}`",
        f"- 抽幀間隔：{meta.get('frame_every_sec', '')} 秒",
        f"- Frame 數：{sum(len(v) for v in grouped.values())}",
        f"- 模型：`{meta.get('model', '')}` / `{meta.get('revision', '')}`",
        f"- 裝置：`{meta.get('device', '')}`",
        "- 說明：本次是獨立的跌倒與抽菸正樣本測試，沒有混入先前 PPE / 輪檔 baseline。",
        "- direct / strict 是本次實驗設計的 prompt 策略，不是 Moondream 內建模式。",
        "",
        "## Prompt 設計",
        "",
        f"- `fall_direct`：{query_prompt(frame_card.config, 'fall_direct')}",
        f"- `fall_strict`：{query_prompt(frame_card.config, 'fall_strict')}",
        f"- `smoking_direct`：{query_prompt(frame_card.config, 'smoking_direct')}",
        f"- `smoking_strict`：{query_prompt(frame_card.config, 'smoking_strict')}",
        f"- `abnormal_direct`：{query_prompt(frame_card.config, 'abnormal_direct')}",
        f"- `scene_evidence`：{query_prompt(frame_card.config, 'scene_evidence')}",
        "",
        "## 結果摘要",
        "",
    ]

    for vid, items in grouped.items():
        expected = items[0].get("expected_event")
        if expected == "fall":
            direct_name, strict_name = event_query_names(expected)
            strict_unclear = first_label_time(items, strict_name, "unclear")
            strict_yes = first_label_time(items, strict_name, "yes")
            direct_yes = first_label_time(items, direct_name, "yes")
            lines.extend(
                [
                    f"### {vid}（{event_name(expected)}影片）",
                    "",
                    f"- `{direct_name}`：{yes_count_text(items, direct_name)} 張判定為「是」。",
                    f"- `{strict_name}`：{yes_count_text(items, strict_name)} 張判定為「是」。",
                    f"- 邊界：`{strict_name}` 首次不確定為 {fmt_time(strict_unclear)}，首次「是」為 {fmt_time(strict_yes)}；`{direct_name}` 首次「是」為 {fmt_time(direct_yes)}。",
                    f"- `{direct_name}` timeline：{timeline_text(items, direct_name)}",
                    f"- `{strict_name}` timeline：{timeline_text(items, strict_name)}",
                    "",
                ]
            )
        elif expected == "smoking":
            direct_name, strict_name = event_query_names(expected)
            lines.extend(
                [
                    f"### {vid}（{event_name(expected)}影片）",
                    "",
                    f"- `{direct_name}`：{yes_count_text(items, direct_name)} 張判定為「是」。",
                    f"- `{strict_name}`：{yes_count_text(items, strict_name)} 張判定為「是」。",
                    f"- `{direct_name}` timeline：{timeline_text(items, direct_name)}",
                    f"- `{strict_name}` timeline：{timeline_text(items, strict_name)}",
                    "",
                ]
            )
            continue

    ran_extra_queries = any(
        {"abnormal_direct", "scene_evidence"} & set(row["result"].get("queries", {}))
        for items in grouped.values()
        for row in items
    )
    lines.extend(["## 執行備註", ""])
    if ran_extra_queries:
        lines.append("- 本 run 有部分影格額外執行泛用異常與開放式證據 query；主要結論仍以 direct / strict 事件 query 為準。")
    else:
        lines.append("- 本 run 全量執行 direct / strict 事件 query，未執行泛用異常與開放式證據 query。")
    lines.extend(
        [
            "- 主要結論以 `fall_direct`、`fall_strict`、`smoking_direct`、`smoking_strict` 為準。",
            "- HTML 報告中的「未執行」代表該影格沒有跑該方法，不代表模型回答否。",
            "",
        ]
    )

    (run_dir / "experiment_record.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = json.loads(Path(args.config).read_text("utf-8"))
    frame_card.config = config

    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        print(f"[error] missing results: {results_path}")
        return 1

    rows = load_jsonl(results_path)
    grouped = by_video(rows)
    write_summary_csv(run_dir, grouped, config)

    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
    write_experiment_record(run_dir, grouped, meta)
    css = """
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f6f8fb; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 22px 60px; }
h1 { font-size: 32px; margin: 0 0 8px; }
h2 { margin-top: 38px; border-top: 1px solid #d9e1ec; padding-top: 28px; }
h3 { margin-top: 24px; }
h4 { margin: 22px 0 10px; }
.hero { background: #fff; border: 1px solid #d9e1ec; border-radius: 8px; padding: 24px; box-shadow: 0 6px 20px rgba(22, 31, 52, 0.06); }
.meta { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.pill { background: #edf2f7; border: 1px solid #d9e1ec; border-radius: 999px; padding: 6px 10px; font-size: 13px; }
.note { padding: 12px 14px; background: #fff8e6; border: 1px solid #edd39a; border-radius: 8px; }
.source { color: #5b6472; }
.finding-list { background: #fff; border: 1px solid #d9e1ec; border-radius: 8px; padding: 18px 24px 18px 34px; line-height: 1.7; }
.explain { background: #fff; border: 1px solid #d9e1ec; border-radius: 8px; padding: 16px 18px; line-height: 1.7; }
.video-summary { background: #ffffff; border-left: 4px solid #2563eb; border-radius: 8px; padding: 14px 18px; margin: 14px 0 18px; box-shadow: 0 3px 12px rgba(22, 31, 52, 0.04); }
.video-summary h3 { margin-top: 0; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; background: #fff; border: 1px solid #d9e1ec; }
th, td { padding: 9px 10px; border-bottom: 1px solid #e6ebf2; text-align: left; vertical-align: top; font-size: 13px; }
th { background: #eef3f8; color: #243047; }
.prompt-cell { line-height: 1.55; color: #26354c; }
.timeline td:first-child { white-space: nowrap; }
.badge { display: inline-block; min-width: 52px; text-align: center; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 700; }
.yes { background: #d7f5e6; color: #166534; }
.no { background: #eef2f7; color: #475569; }
.unclear { background: #fff1c2; color: #92400e; }
.unknown { background: #f4d8dc; color: #991b1b; }
.open { background: #dbeafe; color: #1d4ed8; }
.not-run { background: #f1f5f9; color: #64748b; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.frame-card { background: #fff; border: 1px solid #d9e1ec; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 14px rgba(22, 31, 52, 0.05); }
.thumb { background: #111827; aspect-ratio: 16 / 9; display: flex; align-items: center; justify-content: center; }
.thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
.frame-meta, .labels, .frame-card p { padding: 0 12px; }
.frame-meta { margin-top: 10px; font-size: 13px; color: #334155; }
.labels { margin-top: 8px; line-height: 1.9; }
.frame-card p { font-size: 13px; color: #334155; }
.result-stack { padding: 8px 12px 0; }
.result-line { display: grid; grid-template-columns: 88px 62px 1fr; gap: 8px; align-items: start; padding: 7px 0; border-top: 1px solid #edf2f7; font-size: 13px; }
.result-label { font-weight: 700; color: #243047; }
.answer { color: #334155; line-height: 1.45; }
code { background: #eef2f7; padding: 2px 5px; border-radius: 4px; }
"""

    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Moondream 跌倒與抽菸辨識測試報告</title>
  <style>{css}</style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Moondream 跌倒與抽菸辨識測試</h1>
    <p>本報告是獨立的行為辨識測試，沒有混入先前 PPE / 輪檔 baseline。這次先把短影片轉成照片，再逐張用 Moondream 做靜態影像判斷。</p>
    <div class="meta">
      <span class="pill">實驗編號：{escape(run_dir.name)}</span>
      <span class="pill">影格數：{len(rows)}</span>
      <span class="pill">抽幀：每 {escape(str(meta.get('frame_every_sec', '')))} 秒一張</span>
      <span class="pill">模型：{escape(str(meta.get('model', '')))}</span>
      <span class="pill">裝置：{escape(str(meta.get('device', '')))}</span>
    </div>
  </section>

  <section>
    <h2>專案目的</h2>
    <div class="explain">
      <p>這次測試的目的，是確認 Moondream 在「單張靜態影像」上能不能辨識兩種行為事件：人員跌倒，以及人員抽菸。原始輸入雖然是影片，但 Moondream 本身吃的是圖片，所以流程會先用 OpenCV 把影片切成連續影格，再逐張送進模型。</p>
      <p>本次先完成第一階段：離線辨識結果測試。第二階段的即時錄影/串流處理暫時不納入，本報告只討論靜態影格的模型判斷效果。</p>
    </div>
  </section>

  <section>
    <h2>輸入資料與輸出對應</h2>
    <p>四支短影片都約 6 秒，以每 0.5 秒一張的方式抽出影格，因此每支影片 13 張，共 52 張。下表列出本報告使用的影片代號與來源檔案。</p>
    {input_table(grouped)}
  </section>

  <section>
    <h2>結論摘要</h2>
    {key_findings_html(grouped)}
    <p class="note">注意：為了讓整批測試在本機 MPS 上完成，後半段改為全量執行 direct / strict 事件 query。報告中的「未執行」代表該張影格沒有跑該方法，不代表模型回答否。</p>
  </section>

  <section>
    <h2>方法總覽</h2>
    <div class="explain">
      <p><strong>direct 與 strict 不是 Moondream 的內建功能。</strong>它們是這次實驗設計的兩種 prompt 策略。</p>
      <p><strong>direct</strong> 是直接問「畫面中有沒有發生某事件」，例如有沒有人抽菸、有沒有人倒地。這種問法比較接近正式告警時最直覺的問題。</p>
      <p><strong>strict</strong> 是更嚴格的問法，要求模型只有在清楚看到證據時才回答「是」，如果看不清楚就回答「不確定」或「否」。這是為了觀察模型在邊界影格上的保守程度。</p>
      <p>方法 C 是泛用異常行為問題。方法 D 則是開放式畫面描述加關鍵字訊號。本次因單張耗時過長，後半段改為全量執行 direct / strict；因此主要結論以 direct / strict 為準。</p>
    </div>
    {summary_table(grouped, config)}
  </section>

  <section>
    <h2>實際輸入給模型的 prompt</h2>
    <p>下表列出每種 query 的實際文字。模型對每張影格的操作，就是把該影格轉成 PIL Image，先做 <code>encode_image</code>，再對應執行這些 <code>query</code>。</p>
    {prompt_table(config)}
  </section>

  {''.join(video_section(video_id, items, config) for video_id, items in grouped.items())}

  <section>
    <h2>輸出檔案</h2>
    <p>本次結果都保存在同一個 run folder 內：<code>results.jsonl</code> 是逐張 frame 的模型原始回答，<code>summary.csv</code> 是扁平化表格，<code>behavior_summary.csv</code> 是影片層級摘要，<code>experiment_record.md</code> 是中文實驗紀錄，<code>frames/</code> 是抽出的照片。</p>
    <p>報告產生時間：{escape(datetime.now(timezone.utc).isoformat())}</p>
  </section>
</main>
</body>
</html>
"""
    out = run_dir / "behavior_report.html"
    out.write_text(html, encoding="utf-8")

    meta["stage"] = "report_complete"
    meta["report_file"] = out.name
    meta["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
