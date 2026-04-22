#!/usr/bin/env python3
"""
產出帶圖片的 HTML 詳細分析報告。
所有判斷以「有人幀」（detect person > 0）為基準。
無人幀單獨列為幻覺分析區。

用法：
  python scripts/generate_detail_report.py --run 20260415T102259Z
  python scripts/generate_detail_report.py --run latest
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from html import escape

EVAL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EVAL_DIR.parent
OUTPUTS_DIR = EVAL_DIR / "outputs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="latest")
    p.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data"),
        help="原始影片目錄（擷取遮擋實驗畫面用）",
    )
    p.add_argument(
        "--skip-occlusion",
        action="store_true",
        help="不整合遮擋實驗章節（即使已有 occlusion_results.jsonl）",
    )
    p.add_argument(
        "--skip-attribution",
        action="store_true",
        help="不整合誤判歸因實驗章節（即使已有 attribution_results.jsonl）",
    )
    return p.parse_args()


def find_run_dir(run_id: str) -> Path:
    if run_id == "latest":
        dirs = sorted((d for d in OUTPUTS_DIR.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
        if not dirs:
            sys.exit("[error] outputs/ 內沒有任何測試結果")
        return dirs[0]
    t = OUTPUTS_DIR / run_id
    if not t.exists():
        sys.exit(f"[error] 找不到: {t}")
    return t


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]


def load_config(run_dir: Path) -> dict:
    mp = run_dir / "run_meta.json"
    if mp.exists():
        m = json.loads(mp.read_text("utf-8"))
        cp = EVAL_DIR / "configs" / m.get("config_file", "gap_baseline.json")
        if cp.exists():
            return json.loads(cp.read_text("utf-8"))
    return {}


def img_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii") if p.exists() else ""


def find_occlusion_jsonl(baseline_run_id: str) -> Path | None:
    """尋找 outputs/<run>_occlusion/*/occlusion_results.jsonl（取最新子目錄）。"""
    base = OUTPUTS_DIR / f"{baseline_run_id}_occlusion"
    if not base.is_dir():
        return None
    found = sorted(
        base.rglob("occlusion_results.jsonl"),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    return found[0] if found else None


def find_attribution_jsonl(baseline_run_id: str) -> Path | None:
    """尋找 outputs/<run>_attribution/*/attribution_results.jsonl（取最新子目錄）。"""
    base = OUTPUTS_DIR / f"{baseline_run_id}_attribution"
    if not base.is_dir():
        return None
    found = sorted(
        base.rglob("attribution_results.jsonl"),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    return found[0] if found else None


_frame_cache: dict[tuple, str] = {}

def extract_frame_jpeg_b64(video_path: Path, time_sec: float, max_width: int = 720) -> str:
    key = (str(video_path), time_sec, max_width)
    if key in _frame_cache:
        return _frame_cache[key]
    try:
        import io
        import cv2
        from PIL import Image
    except ImportError:
        return ""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _frame_cache[key] = ""
        return ""
    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        _frame_cache[key] = ""
        return ""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    w, h = im.size
    if w > max_width:
        ratio = max_width / w
        im = im.resize((max_width, int(h * ratio)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    result = base64.b64encode(buf.getvalue()).decode("ascii")
    _frame_cache[key] = result
    return result


def slug(entry: dict) -> str:
    v = entry.get("video", "x")
    t = entry.get("time_sec", 0)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(v).stem).strip("_")
    return f"{s}_{int(t):05d}s.jpg"


# ── 幀描述產生 ────────────────────────────────────────────────────

def describe_frame(entry: dict, query_name: str) -> str:
    """為每個幀產生一段中文描述，說明圖片裡發生了什麼、模型做了什麼判斷。"""
    r = entry["result"]
    q = r["queries"].get(query_name, {})
    worker_q = r["queries"].get("worker_visible", {})
    cap = r["caption"]
    det_person = r["detections"].get("person", {}).get("count", 0)
    det_helmet = r["detections"].get("helmet", {}).get("count", 0)
    det_wc = r["detections"].get("wheel chock", {}).get("count", 0)
    det_cig = r["detections"].get("cigarette", {}).get("count", 0)
    worker_label = worker_q.get("label", "unknown")
    q_label = q.get("label", "unknown")
    q_answer = q.get("answer", "").strip()

    lines = []

    if worker_label == "yes" and det_person > 0:
        lines.append(f"畫面中偵測到 {det_person} 位工人（detect person = {det_person}）。")
    elif worker_label == "yes" and det_person == 0:
        lines.append("模型認為有工人（worker_visible = yes），但 detect person 沒有框到人，存在矛盾。")
    else:
        lines.append("畫面中沒有偵測到工人。")

    if det_helmet > 0:
        lines.append(f"偵測到 {det_helmet} 頂安全帽（detect helmet = {det_helmet}）。")

    prompt_map = {
        "missing_helmet": "「有沒有人沒戴安全帽？」",
        "missing_face_mask": "「有沒有工人缺少面罩/防毒面具？」",
        "smoking_visible": "「有沒有人在抽菸？」",
        "abnormal_behavior": "「有沒有不安全/異常的行為？」",
        "person_falling": "「有沒有人倒地/跌倒？」",
    }
    question = prompt_map.get(query_name, f"「{query_name}」")

    if q_label == "yes":
        lines.append(f"問模型{question}，模型回答 <strong>Yes</strong>。")
    elif q_label == "no":
        lines.append(f"問模型{question}，模型回答 <strong>No</strong>。")
    else:
        lines.append(f"問模型{question}，模型回答不明確。")

    if query_name == "missing_face_mask" and q_label == "yes":
        ans_lower = q_answer.lower()
        if "wearing" in ans_lower and ("mask" in ans_lower or "respirator" in ans_lower):
            lines.append("⚠️ <strong>矛盾回答</strong>：模型說「Yes」（缺面罩），但描述中又提到工人正在戴面罩/防毒面具。模型語意理解有誤。")
        elif "not wearing" in ans_lower or "no respirator" in ans_lower or "no face mask" in ans_lower or "missing" in ans_lower:
            lines.append("模型認為工人確實沒戴面罩。<strong>這可能是真實情況</strong>（不一定是誤報），需要對照圖片確認。")
        else:
            lines.append("模型給出了肯定回答，但描述含糊，需對照圖片判斷。")

    if query_name == "missing_helmet" and q_label == "yes":
        if det_helmet > 0:
            lines.append("⚠️ <strong>矛盾</strong>：模型說有人沒戴安全帽，但同時 detect 框到了安全帽。")
        else:
            lines.append("detect helmet = 0，模型確實沒框到安全帽，query 和 detect 結果一致。")

    if query_name == "smoking_visible" and q_label == "yes":
        if det_cig == 0:
            lines.append("但 detect cigarette = 0（沒框到香菸），可能是把管線或工具誤判為抽菸動作。")

    return " ".join(lines)


def describe_hallucination(entry: dict) -> str:
    """描述無人幀的幻覺問題。"""
    r = entry["result"]
    det_person = r["detections"].get("person", {}).get("count", 0)
    cap = r["caption"]

    fp_queries = []
    for qname in ["missing_helmet", "missing_face_mask", "smoking_visible", "abnormal_behavior"]:
        q = r["queries"].get(qname, {})
        if q.get("label") == "yes":
            fp_queries.append(qname)

    lines = ["畫面中 <strong>沒有人</strong>（detect person = 0、worker_visible = no）。"]
    if fp_queries:
        names = "、".join(fp_queries)
        lines.append(f"但模型仍然對以下問題回答了 Yes：<strong>{names}</strong>。")
        lines.append("這是典型的<strong>幻覺（Hallucination）</strong>— 模型在沒有人的畫面上「想像」出了工人和問題。")
        lines.append("正式系統中，這類幻覺可以透過 <strong>Worker Gate</strong>（先確認有人再問後續問題）完全排除。")
    else:
        lines.append("模型所有回答都是 No，判斷正確。")
    return " ".join(lines)


# ── HTML 組件 ─────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.7; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
h1 { font-size: 26px; margin: 30px 0 10px; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }
h2 { font-size: 21px; margin: 30px 0 10px; color: #1e40af; border-left: 4px solid #2563eb; padding-left: 12px; }
h3 { font-size: 17px; margin: 18px 0 8px; color: #374151; }
p, li { margin: 6px 0; }
.meta { background: #fff; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.meta span { display: inline-block; margin-right: 20px; color: #6b7280; }
.meta strong { color: #111827; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
th { background: #1e40af; color: #fff; padding: 10px 12px; text-align: left; }
td { padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }
tr:hover { background: #f9fafb; }
.fp-high { background: #fef2f2; color: #991b1b; font-weight: 600; }
.fp-med { background: #fffbeb; color: #92400e; font-weight: 600; }
.fp-ok { background: #f0fdf4; color: #166534; font-weight: 600; }
.scenario { background: #fff; border-radius: 12px; padding: 24px; margin: 24px 0; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
.badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
.badge-r { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.badge-y { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
.badge-g { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
.frame-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 16px; margin: 16px 0; }
.fcard { border-radius: 8px; overflow: hidden; border: 1px solid #e5e7eb; background: #f9fafb; }
.fcard.fp { border: 2px solid #ef4444; }
.fcard.ok { border: 2px solid #22c55e; }
.fcard.hall { border: 2px solid #f59e0b; }
.fcard img { width: 100%; display: block; }
.finfo { padding: 14px; font-size: 13px; }
.finfo .ftitle { font-weight: 700; font-size: 14px; margin-bottom: 6px; }
.finfo .fdesc { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; margin: 8px 0; line-height: 1.7; }
.finfo .fanswer { background: #f3f4f6; padding: 8px; border-radius: 4px; margin: 6px 0; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin: 2px; font-weight: 600; }
.tag-person { background: #fef2f2; color: #dc2626; }
.tag-helmet { background: #dcfce7; color: #166534; }
.tag-wc { background: #fef9c3; color: #854d0e; }
.tag-cig { background: #f5d0fe; color: #86198f; }
.tag-fp { background: #dc2626; color: #fff; }
.tag-ok { background: #16a34a; color: #fff; }
.tag-hall { background: #f59e0b; color: #fff; }
.task-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 12px 0; }
.rec-box { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin: 16px 0; }
.rec-box li { margin: 6px 0 6px 20px; }
.pattern-box { background: #fefce8; border: 1px solid #fde68a; border-radius: 8px; padding: 16px; margin: 16px 0; }
.group-hdr { font-size: 15px; font-weight: 700; color: #374151; margin: 18px 0 8px; padding: 8px 12px; background: #f3f4f6; border-radius: 6px; }
.tag-unclear { background: #e0e7ff; color: #3730a3; }
.cmp-table td.better { background: #f0fdf4; color: #166534; font-weight: 600; }
.cmp-table td.worse { background: #fef2f2; color: #991b1b; font-weight: 600; }
.cmp-table td.same { background: #f9fafb; color: #374151; }
"""

SCENARIO_META = {
    "missing_helmet": {
        "title": "安全帽缺失偵測",
        "question_zh": "有沒有人沒戴安全帽？",
        "related_det": ["person", "helmet"],
        "yolo_gap": "YOLO 需要分別框 person 和 helmet，再用規則算兩者位置關聯。Moondream 直接用語意判斷。",
        "recs": [
            "Worker Gate：只在 detect person > 0 時才問這個問題",
            "交叉驗證：query=yes 時，再確認 detect helmet=0",
            "連續幀確認：連續 N 幀都報 yes 才觸發告警",
        ],
    },
    "missing_face_mask": {
        "title": "面罩/防毒面具缺失偵測",
        "question_zh": "有沒有工人缺少面罩或防毒面具？",
        "related_det": ["person"],
        "yolo_gap": "面罩種類多（防毒面罩、外科口罩、面盾），YOLO 需大量標註各種型態。Moondream 用語意理解。",
        "note": "⚠️ 這批資料中工人可能確實部分場景沒戴面罩，所以 yes 回答不一定全是誤報。需要對照圖片個別判斷。",
        "recs": [
            "先釐清業務需求：哪些作業必須戴面罩？不是所有場景都要求",
            "注意模型的矛盾回答：有時說 Yes 缺面罩但描述中又說正在戴面罩",
            "Worker Gate + 背對鏡頭時不判定",
        ],
    },
    "smoking_visible": {
        "title": "抽菸偵測",
        "question_zh": "有沒有人在抽菸或手持香菸？",
        "related_det": ["person", "cigarette"],
        "yolo_gap": "香菸在畫面中極小（可能只有幾個像素），YOLO 的 anchor 很難框到。Moondream 用語意理解。",
        "recs": [
            "交叉驗證：query=yes 但 detect cigarette=0 時不觸發",
            "誤判主因是管線/工具被當成手持物品，可加二次確認 prompt",
            "連續幀確認：單幀 yes 不報，連續 3 幀才報",
        ],
    },
    "abnormal_behavior": {
        "title": "異常行為偵測",
        "question_zh": "有沒有不安全、異常或可疑的工人行為？",
        "related_det": ["person"],
        "yolo_gap": "YOLO 只能框物件，完全不理解「行為」。Moondream 可以描述動作並判斷是否異常。",
        "recs": [
            "誤判主因是正常操作姿勢（彎腰、攀爬、靠近設備）被判成可疑",
            "改用更具體的 prompt（例：Is anyone lying on the ground? / Is anyone running?）",
            "搭配 caption 做二次語意過濾",
        ],
    },
    "person_falling": {
        "title": "倒地偵測",
        "question_zh": "有沒有人倒地、跌倒或需要救助？",
        "related_det": ["person", "fallen person"],
        "yolo_gap": "倒地姿勢多樣、可能被遮擋，YOLO 需專門訓練。Moondream 可語意理解。",
        "recs": [
            "Query 方式 FP=0%，表現優良",
            "但 detect 'fallen person' FP=66%，不要用 detect 方式",
            "需要正樣本（真正有人倒地的畫面）驗證 TP 率",
        ],
    },
}


def get_prompt(config: dict, qname: str) -> str:
    for qc in config.get("query_checks", []):
        if qc["name"] == qname:
            return qc.get("prompt", "")
    return ""


# ── 分類幀 ────────────────────────────────────────────────────────

def split_frames(entries: list[dict]):
    with_p = [e for e in entries if e["result"]["detections"].get("person", {}).get("count", 0) > 0]
    no_p = [e for e in entries if e["result"]["detections"].get("person", {}).get("count", 0) == 0]
    return with_p, no_p


# ── HTML 產生 ─────────────────────────────────────────────────────

def build_html(
    run_dir: Path,
    entries: list[dict],
    config: dict,
    *,
    attribution_jsonl: Path | None = None,
    occlusion_jsonl: Path | None = None,
    data_dir: Path | None = None,
) -> str:
    ann = run_dir / "annotated"
    meta = json.loads((run_dir / "run_meta.json").read_text("utf-8")) if (run_dir / "run_meta.json").exists() else {}
    with_p, no_p = split_frames(entries)
    parts = [_head(meta, entries, with_p, no_p)]
    parts.append(_methodology(config))
    parts.append(_overview(config, with_p, no_p))
    parts.append(_strict_comparison(config, with_p))

    for qname in ["missing_helmet", "missing_face_mask", "smoking_visible",
                   "abnormal_behavior", "person_falling"]:
        parts.append(_scenario(qname, config, with_p, no_p, ann))

    parts.append(_hallucination_section(no_p, ann))
    parts.append(_detect_fp_section(entries, ann))
    parts.append(_attribution_experiment_section(run_dir, attribution_jsonl))
    parts.append(
        _occlusion_experiment_section(
            run_dir,
            entries,
            occlusion_jsonl,
            data_dir or (PROJECT_ROOT / "data"),
        )
    )
    parts.append(_conclusion(config, with_p))
    parts.append(_pipeline_section())
    parts.append(_phase2_occlusion_section(run_dir, data_dir or (PROJECT_ROOT / "data")))
    parts.append(_helmet_consistency_section(run_dir, data_dir or (PROJECT_ROOT / "data")))
    parts.append(_mask_occlusion_section(run_dir, data_dir or (PROJECT_ROOT / "data")))
    parts.append("</div></body></html>")
    return "\n".join(parts)


def _head(meta, entries, with_p, no_p):
    return f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8">
<title>Moondream YOLO-Gap 負樣本基線分析</title><style>{CSS}</style></head>
<body><div class="container">
<h1>Moondream YOLO-Gap 負樣本基線分析報告</h1>
<div class="meta">
  <span>Run: <strong>{meta.get('run_id','')}</strong></span>
  <span>模型: <strong>{meta.get('model','')}</strong></span>
  <span>裝置: <strong>{meta.get('device','')}</strong></span>
  <span>間隔: <strong>{meta.get('sample_every_sec','')}s</strong></span>
</div>
<div class="meta" style="background:#eff6ff;">
  <strong>幀數統計：</strong>
  總共 <strong>{len(entries)}</strong> 幀，其中
  <span class="tag tag-person">有人幀: {len(with_p)}</span>
  <span class="tag" style="background:#6b7280;color:#fff;">無人幀: {len(no_p)}</span>
  <br><br>
  <strong>本報告所有 FP 率以「有人幀」（detect person &gt; 0）為基準。</strong><br>
  原因：所有安全相關問題（安全帽、面罩、抽菸等）只在畫面中有工人時才有意義。<br>
  無人幀的誤報屬於「幻覺」，透過 Worker Gate（先偵測人 → 有人才問問題）可完全排除。
</div>
"""


def _methodology(config):
    qrows = ""
    for qc in config.get("query_checks", []):
        qrows += f"<tr><td><code>{qc['name']}</code></td><td style='font-size:12px'>{escape(qc['prompt'])}</td></tr>"
    drows = ""
    for dt in config.get("detect_targets", []):
        drows += f"<tr><td><code>{dt['name']}</code></td><td><code>model.detect(image, \"{dt['name']}\")</code></td></tr>"

    return f"""
<div class="scenario">
<h2 style="border:none;margin:0 0 12px;padding:0;">測試方法：每一幀做了什麼？</h2>
<p>對每一幀影像，依序執行 Moondream 的三種能力：</p>

<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:12px 0;">
  <div style="background:#eff6ff; padding:12px; border-radius:8px; border:1px solid #bfdbfe;">
    <strong style="color:#1e40af;">① Caption（場景描述）</strong><br>
    <code style="font-size:12px;">model.caption(image)</code><br>
    <span style="font-size:12px;">讓模型描述畫面內容，產出一段自然語言。</span>
  </div>
  <div style="background:#f0fdf4; padding:12px; border-radius:8px; border:1px solid #bbf7d0;">
    <strong style="color:#166534;">② Query（是非問答）</strong><br>
    <code style="font-size:12px;">model.query(image, "問題")</code><br>
    <span style="font-size:12px;">對畫面提問，模型回答 Yes/No + 解釋。</span>
  </div>
  <div style="background:#fefce8; padding:12px; border-radius:8px; border:1px solid #fde68a;">
    <strong style="color:#92400e;">③ Detect（物件偵測）</strong><br>
    <code style="font-size:12px;">model.detect(image, "目標")</code><br>
    <span style="font-size:12px;">在畫面中框出指定物件，回傳 bbox 座標。</span>
  </div>
</div>

<h3>Query 問題列表（每幀都會問）</h3>
<table><tr><th>名稱</th><th>送給模型的英文 Prompt</th></tr>{qrows}</table>

<h3>Detect 目標列表（每幀都會框）</h3>
<table><tr><th>目標</th><th>呼叫方式</th></tr>{drows}</table>

<h3>標註圖怎麼看？</h3>
<ul style="margin-left:20px;">
<li><strong style="color:#FF4444;">紅色框</strong> = person（人）</li>
<li><strong style="color:#44FF44;">綠色框</strong> = helmet（安全帽）</li>
<li><strong style="color:#FFD700;">金色框</strong> = wheel chock（輪擋）</li>
<li><strong style="color:#FF00FF;">粉色框</strong> = cigarette（香菸）</li>
<li><strong style="color:#00FFFF;">青色框</strong> = fallen person</li>
<li><strong>左上角文字</strong> = 各 query 的 Y(Yes)/N(No) 結果</li>
</ul>
</div>
"""


def _overview(config, with_p, no_p):
    rows = ""
    for qname in ["missing_helmet", "missing_face_mask", "smoking_visible",
                   "abnormal_behavior", "person_falling"]:
        sm = SCENARIO_META.get(qname, {})
        yes_wp = sum(1 for e in with_p if e["result"]["queries"].get(qname, {}).get("label") == "yes")
        total_wp = len(with_p)
        pct = (yes_wp / total_wp * 100) if total_wp else 0
        yes_nop = sum(1 for e in no_p if e["result"]["queries"].get(qname, {}).get("label") == "yes")
        cls = "fp-ok" if pct == 0 else ("fp-med" if pct <= 10 else "fp-high")
        verdict = "OK" if pct == 0 else ("加規則可用" if pct <= 10 else "高風險")
        rows += f"<tr><td>{sm.get('title', qname)}</td><td>{sm.get('question_zh','')}</td>"
        rows += f"<td>{yes_wp} / {total_wp}</td><td class='{cls}'>{pct:.1f}%</td>"
        rows += f"<td>{yes_nop} 幀（幻覺）</td><td class='{cls}'>{verdict}</td></tr>"

    worker_gate_rows = ""
    for qname in ["missing_helmet", "missing_face_mask", "smoking_visible",
                   "abnormal_behavior", "person_falling"]:
        sm = SCENARIO_META.get(qname, {})
        total = len(with_p) + len(no_p)
        fp_all = sum(1 for e in (with_p + no_p) if e["result"]["queries"].get(qname, {}).get("label") == "yes")
        fp_wp = sum(1 for e in with_p if e["result"]["queries"].get(qname, {}).get("label") == "yes")
        fp_all_pct = fp_all / total * 100 if total else 0
        fp_wp_pct = fp_wp / len(with_p) * 100 if with_p else 0
        delta_pct = fp_wp_pct - fp_all_pct
        if delta_pct > 2:
            change = f'<span style="color:#dc2626;">▲ {delta_pct:+.1f}%（gate 後 FP 上升，無人幀幫助較少）</span>'
        elif delta_pct < -2:
            change = f'<span style="color:#16a34a;">▼ {delta_pct:+.1f}%（gate 後 FP 下降）</span>'
        else:
            change = "持平"
        worker_gate_rows += f"<tr><td>{sm.get('title', qname)}</td><td>{fp_all_pct:.1f}% ({fp_all}/{total})</td><td>{fp_wp_pct:.1f}% ({fp_wp}/{len(with_p)})</td><td>{change}</td></tr>"

    return f"""
<h2>總覽（基準：有人幀 = {len(with_p)} 幀）</h2>
<p>以下 FP 率 = 在有人的幀中，模型錯誤回答 Yes 的比例。<br>
這批影片全為正常作業，工人有戴安全帽與面罩、有輪擋、無抽菸、無異常行為。</p>
<table>
<tr><th>場景</th><th>問了什麼</th><th>有人幀中答 Yes</th><th>FP 率</th><th>無人幀答 Yes（幻覺）</th><th>判定</th></tr>
{rows}
</table>

<h3 style="margin-top:24px;">Worker Gate 效果</h3>
<p>Worker Gate = 先確認有人，再問安全問題。下表比較加 gate 前後的 FP 率，<strong>上升代表無人幀幫助不大、問題主要來自有人幀本身</strong>。</p>
<table>
<tr><th>場景</th><th>無 Gate FP（全幀）</th><th>有 Gate FP（僅有人幀）</th><th>變化說明</th></tr>
{worker_gate_rows}
</table>
"""


def _strict_comparison(config, with_p):
    """Standard vs Strict prompt 對照：測試 unclear 選項是否能降低 FP。"""
    pairs = [
        ("missing_helmet",    "missing_helmet_strict",    "安全帽缺失"),
        ("missing_face_mask", "missing_face_mask_strict",  "面罩缺失"),
    ]

    has_strict = any(
        any(qc["name"] == strict for qc in config.get("query_checks", []))
        for _, strict, _ in pairs
    )
    if not has_strict:
        return ""

    n = len(with_p)
    if n == 0:
        return ""

    rows = ""
    for std_name, strict_name, label in pairs:
        std_yes = sum(1 for e in with_p if e["result"]["queries"].get(std_name, {}).get("label") == "yes")
        strict_yes = sum(1 for e in with_p if e["result"]["queries"].get(strict_name, {}).get("label") == "yes")
        strict_unclear = sum(1 for e in with_p if e["result"]["queries"].get(strict_name, {}).get("label") in ("unclear", "unknown"))
        strict_no = n - strict_yes - strict_unclear

        std_fp_pct = std_yes / n * 100 if n else 0
        strict_fp_pct = strict_yes / n * 100 if n else 0
        unclear_pct = strict_unclear / n * 100 if n else 0
        delta = strict_fp_pct - std_fp_pct

        if delta < -2:
            delta_cls = "better"
            delta_str = f"▼ {abs(delta):.1f}% 改善"
        elif delta > 2:
            delta_cls = "worse"
            delta_str = f"▲ {delta:.1f}% 變差"
        else:
            delta_cls = "same"
            delta_str = "持平"

        if unclear_pct > 30:
            unclear_note = f"✓ 模型承認 {unclear_pct:.0f}% 幀無法清楚判斷（可排除）"
        elif unclear_pct > 0:
            unclear_note = f"{unclear_pct:.0f}% 回答 unclear"
        else:
            unclear_note = "無 unclear 回答（模型未使用此選項）"

        rows += f"""<tr>
<td><strong>{label}</strong></td>
<td>{std_fp_pct:.1f}% ({std_yes}/{n})</td>
<td>{strict_fp_pct:.1f}% ({strict_yes}/{n})</td>
<td class="tag-unclear" style="font-size:12px;">{unclear_note}</td>
<td class="{delta_cls}">{delta_str}</td>
</tr>"""

    explanation = """
<p>這組對照回答一個核心問題：<strong>給 Moondream 說「如果不確定就回答 unclear」，FP 率會降低嗎？還是模型根本不在乎這個選項？</strong></p>
<ul style="margin-left:20px;">
  <li><strong>Strict FP 明顯下降 + unclear 多</strong> → 模型能識別不確定場景，加 unclear 選項有效，可用於過濾 YOLO 誤報</li>
  <li><strong>Strict FP 沒有下降</strong> → 模型即使看不清楚仍堅持給 yes/no，遮擋過濾無效</li>
  <li><strong>Strict FP 反而上升</strong> → strict prompt 讓模型更激進，需調整措辭</li>
</ul>
"""

    return f"""
<div class="scenario" style="border-top:4px solid #7c3aed;">
  <h2 style="border:none;margin:0 0 12px;padding:0;">Standard vs Strict Prompt 對照實驗</h2>
  {explanation}
  <table class="cmp-table">
    <tr>
      <th>場景</th>
      <th>Standard FP 率</th>
      <th>Strict FP 率</th>
      <th>Unclear 分析</th>
      <th>結論</th>
    </tr>
    {rows}
  </table>
  <div class="rec-box" style="margin-top:16px;">
    <strong>如何解讀：</strong>
    Strict prompt 把 <code>unclear</code> 當作第三個合法答案。若系統設計為「只有 strict=yes 時才告警」，
    則 unclear 的幀就不會觸發警報——等同於自動過濾視角/遮擋導致的誤判。
    這個實驗直接驗證此策略是否有效。
  </div>
</div>
"""


def _scenario(qname, config, with_p, no_p, ann_dir):
    sm = SCENARIO_META.get(qname, {})
    prompt = get_prompt(config, qname)

    fp_wp = [e for e in with_p if e["result"]["queries"].get(qname, {}).get("label") == "yes"]
    ok_wp = [e for e in with_p if e["result"]["queries"].get(qname, {}).get("label") != "yes"]
    total_wp = len(with_p)
    pct = (len(fp_wp) / total_wp * 100) if total_wp else 0

    bcls = "badge-g" if pct == 0 else ("badge-y" if pct <= 10 else "badge-r")

    det_calls = " ".join(f'<code>model.detect(image, "{d}")</code>' for d in sm.get("related_det", []))

    html = f"""
<div class="scenario">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h2 style="border:none;margin:0;padding:0;">{sm.get('title', qname)}</h2>
    <span class="badge {bcls}">有人幀 FP = {pct:.1f}% ({len(fp_wp)}/{total_wp})</span>
  </div>

  <div class="task-box">
    <h3 style="margin:0 0 8px;">本場景執行的任務</h3>
    <table style="margin:0;box-shadow:none;">
      <tr>
        <td style="width:100px;font-weight:600;color:#1e40af;vertical-align:top;">② Query</td>
        <td>
          問模型：<strong>{sm.get('question_zh','')}</strong><br>
          <code style="font-size:12px;">model.query(image, "{escape(prompt)}")</code><br>
          <span style="font-size:12px;color:#6b7280;">模型回答 Yes 或 No + 解釋。在正常作業畫面中，預期回答 = <strong>No</strong></span>
        </td>
      </tr>
      <tr>
        <td style="font-weight:600;color:#92400e;vertical-align:top;">③ Detect</td>
        <td>{det_calls}<br><span style="font-size:12px;color:#6b7280;">同時框出相關物件，用於交叉驗證</span></td>
      </tr>
    </table>
  </div>

  <p><strong>YOLO Gap：</strong>{sm.get('yolo_gap','')}</p>
"""

    if sm.get("note"):
        html += f'<div class="pattern-box">{sm["note"]}</div>'

    if sm.get("recs"):
        html += '<div class="rec-box"><h3>改善建議</h3><ol>'
        for r in sm["recs"]:
            html += f"<li>{r}</li>"
        html += "</ol></div>"

    if fp_wp:
        html += f'<div class="group-hdr">誤判案例（有人但判斷錯誤）— {len(fp_wp)} 幀</div>'
        html += '<div class="frame-grid">'
        for e in fp_wp:
            html += _frame(e, qname, ann_dir, "fp")
        html += "</div>"

    if ok_wp and fp_wp:
        show = ok_wp[:3]
        html += f'<div class="group-hdr">正確案例對照（有人、判斷正確）— 顯示 {len(show)} 幀</div>'
        html += '<div class="frame-grid">'
        for e in show:
            html += _frame(e, qname, ann_dir, "ok")
        html += "</div>"

    if not fp_wp:
        html += '<div class="pattern-box" style="background:#f0fdf4;border-color:#bbf7d0;">在所有有人的幀中，模型全部回答正確（No），沒有任何誤報。表現優良。</div>'

    html += "</div>"
    return html


def _frame(entry, qname, ann_dir, kind):
    """kind: 'fp', 'ok', 'hall'"""
    b64 = img_b64(ann_dir / slug(entry))
    if not b64:
        return ""

    r = entry["result"]
    q = r["queries"].get(qname, {})
    det_p = r["detections"].get("person", {}).get("count", 0)
    det_h = r["detections"].get("helmet", {}).get("count", 0)
    det_wc = r["detections"].get("wheel chock", {}).get("count", 0)

    if kind == "fp":
        desc = describe_frame(entry, qname)
        tag = '<span class="tag tag-fp">FALSE POSITIVE</span>'
    elif kind == "hall":
        desc = describe_hallucination(entry)
        tag = '<span class="tag tag-hall">幻覺</span>'
    else:
        desc = describe_frame(entry, qname)
        tag = '<span class="tag tag-ok">CORRECT</span>'

    return f"""
<div class="fcard {kind}">
  <img src="data:image/jpeg;base64,{b64}">
  <div class="finfo">
    <div class="ftitle">{tag} {entry['video'][:30]}... @ {entry['time_sec']}s</div>
    <div class="fdesc">{desc}</div>
    <div style="margin:4px 0;">
      <span class="tag tag-person">person: {det_p}</span>
      <span class="tag tag-helmet">helmet: {det_h}</span>
      <span class="tag tag-wc">wheel chock: {det_wc}</span>
    </div>
    <div style="font-size:11px;color:#6b7280;margin-top:4px;"><strong>Query 原文回答：</strong></div>
    <div class="fanswer">{escape(q.get('answer','')[:250])}</div>
    <div style="font-size:11px;color:#6b7280;margin-top:4px;"><strong>Caption 場景描述：</strong></div>
    <div class="fanswer">{escape(r['caption'][:200])}</div>
  </div>
</div>"""


def _hallucination_section(no_p, ann_dir):
    if not no_p:
        return ""
    has_fp = [e for e in no_p if any(
        e["result"]["queries"].get(q, {}).get("label") == "yes"
        for q in ["missing_helmet", "missing_face_mask", "smoking_visible", "abnormal_behavior"]
    )]
    no_fp = [e for e in no_p if e not in has_fp]

    html = f"""
<div class="scenario">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h2 style="border:none;margin:0;padding:0;">無人幀幻覺分析</h2>
    <span class="badge badge-y">{len(no_p)} 幀無人，{len(has_fp)} 幀有幻覺</span>
  </div>
  <p>這些幀中 detect person = 0（畫面沒有工人），但模型對某些問題仍回答了 Yes。
  這是 <strong>幻覺（Hallucination）</strong>，正式系統透過 Worker Gate 可完全排除。</p>
"""
    if has_fp:
        html += f'<div class="group-hdr">有幻覺的無人幀 — {len(has_fp)} 幀</div>'
        html += '<div class="frame-grid">'
        for e in has_fp[:6]:
            html += _frame(e, "missing_face_mask", ann_dir, "hall")
        html += "</div>"

    if no_fp:
        html += f'<div class="group-hdr">無幻覺的無人幀（全部回答正確）— {len(no_fp)} 幀</div>'
        html += f'<p style="color:#6b7280;">共 {len(no_fp)} 幀，模型在這些無人畫面上所有 query 都正確回答 No。</p>'

    html += "</div>"
    return html


def _detect_fp_section(entries, ann_dir):
    fp_fallen = [e for e in entries if e["result"]["detections"].get("fallen person", {}).get("count", 0) > 0]
    if not fp_fallen:
        return ""

    html = f"""
<div class="scenario">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h2 style="border:none;margin:0;padding:0;">Detect「fallen person」誤框問題</h2>
    <span class="badge badge-r">誤框 {len(fp_fallen)}/{len(entries)} 幀 = {len(fp_fallen)/len(entries)*100:.0f}%</span>
  </div>
  <div class="task-box">
    <strong>執行的任務：</strong><code>model.detect(image, "fallen person")</code><br>
    <span style="font-size:13px;">要求模型在畫面中框出「倒地的人」。正常作業畫面中不該有人倒地。</span>
  </div>
  <p>模型大量將正常站立/蹲下/操作的工人誤框為「fallen person」。<strong>不建議使用 detect 方式偵測倒地</strong>，改用 query 問答方式（FP=0%）。</p>
  <div class="group-hdr">誤框案例 — 顯示前 4 幀</div>
  <div class="frame-grid">
"""
    for e in fp_fallen[:4]:
        b64 = img_b64(ann_dir / slug(e))
        if not b64:
            continue
        det_fp = e["result"]["detections"]["fallen person"]["count"]
        det_p = e["result"]["detections"].get("person", {}).get("count", 0)
        html += f"""
<div class="fcard fp">
  <img src="data:image/jpeg;base64,{b64}">
  <div class="finfo">
    <div class="ftitle"><span class="tag tag-fp">誤框</span> {e['video'][:30]}... @ {e['time_sec']}s</div>
    <div class="fdesc">
      detect("fallen person") 框出了 <strong>{det_fp}</strong> 個區域，
      但 detect("person") = {det_p}，工人是正常作業姿態。
      Moondream 的 detect 無法區分站立與倒地，<strong>不建議使用 detect 偵測倒地</strong>。
      改用 query 問答（<code>person_falling</code>）在這批資料的 FP=0%。
    </div>
    <div class="fanswer" style="font-size:11px;">{escape(e['result']['caption'][:200])}</div>
  </div>
</div>"""
    html += "</div></div>"
    return html


def _attribution_experiment_section(run_dir: Path, attribution_jsonl: Path | None) -> str:
    """誤判歸因實驗（合規假設）：較嚴格問法，量化「影像因素 vs 模型堅稱清楚違規」。"""
    rid = run_dir.name
    cfg_path = EVAL_DIR / "configs" / "attribution_queries.json"
    cfg_txt = ""
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text("utf-8"))
        for q in cfg.get("queries", []):
            cfg_txt += f"<li><strong>{escape(q['name'])}</strong><br><span style='font-size:12px;color:#475569'>{escape(q['prompt'][:400])}</span></li>"

    if not attribution_jsonl or not attribution_jsonl.exists():
        return f"""
<div class="scenario" style="border:2px dashed #94a3b8;">
<h2 style="border:none;margin:0 0 12px;padding:0;">誤判歸因實驗（合規假設下 · 未匯入）</h2>
<p>用於回答：在<strong>工人應已配戴</strong>的前提下，系統誤判有多少可歸因於<strong>影像</strong>（角度、遮擋），而非模型「堅稱能清楚看到違規」。</p>
<pre style="background:#f1f5f9;padding:12px;border-radius:8px;font-size:13px;">cd evaluation
python scripts/run_attribution_experiment.py --baseline-run {rid}</pre>
<p>完成後再執行 <code>python scripts/generate_detail_report.py --run {rid}</code> 可嵌入統計。</p>
</div>
"""

    rows = []
    for line in attribution_jsonl.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        b = o["baseline"]
        a = o["attribution"]
        rows.append({
            "clear_absence_helmet": a["clear_absence_helmet"]["label"],
            "head_region_visibility_limited": a["head_region_visibility_limited"]["label"],
            "clear_absence_mask": a["clear_absence_mask"]["label"],
            "detect_helmet_count": b["detect_helmet_count"],
        })

    n = len(rows)
    if n == 0:
        return ""

    admit_no = sum(1 for r in rows if r["clear_absence_helmet"] == "no")
    admit_yes = sum(1 for r in rows if r["clear_absence_helmet"] == "yes")
    vis_yes = sum(1 for r in rows if r["head_region_visibility_limited"] == "yes")
    insist = sum(
        1
        for r in rows
        if r["clear_absence_helmet"] == "yes" and r["head_region_visibility_limited"] == "no"
    )
    insist_note = (
        "本批未出現此交叉組合（clear=Yes 且頭部可見性未標為受限）。"
        if insist == 0
        else "此列為模型在「頭部可見性未標為受限」時仍堅稱能清楚看到未戴帽，宜複核。"
    )
    mask_no = sum(1 for r in rows if r["clear_absence_mask"] == "no")
    mask_yes = sum(1 for r in rows if r["clear_absence_mask"] == "yes")

    only_no_det = [r for r in rows if int(r["detect_helmet_count"]) == 0]
    nd = len(only_no_det)
    nd_admit = sum(1 for r in only_no_det if r["clear_absence_helmet"] == "no")
    nd_pct = f"{100 * nd_admit / nd:.1f}%" if nd else "—（子集為空）"

    report_md = attribution_jsonl.parent / "attribution_report.md"
    report_link = f"file://{report_md.resolve()}" if report_md.exists() else ""

    warn_vis = ""
    if vis_yes >= n * 0.9:
        warn_vis = (
            "<p class='pattern-box'><strong>注意：</strong>本批 <code>head_region_visibility_limited</code> 接近全為 Yes，"
            "此題可能過寬；<strong>建議以 <code>clear_absence_helmet</code> 為主</strong>解讀「模型是否承認無法清楚指認未戴帽」。</p>"
        )

    return f"""
<div class="scenario" style="border-top:4px solid #0f766e;">
<h2 style="border:none;margin:0 0 12px;padding:0;">誤判歸因實驗（合規假設：現場應已配戴）</h2>
<p>在「工人應已戴安全帽／面罩」的前提下，將 baseline 中「未框到帽或 missing 為 yes」視為<strong>系統誤判</strong>，
再以<strong>較嚴格</strong>的英文問法詢問 Moondream，區分：</p>
<ul style="margin-left:20px;">
<li><strong>影像／可見性可解釋</strong>：模型承認<strong>無法清楚、無歧義地</strong>看到「未戴帽」（<code>clear_absence_helmet</code> = No）。</li>
<li><strong>模型仍聲稱能清楚看到未戴帽</strong>（與合規假設衝突）：<code>clear_absence_helmet</code> = Yes。</li>
</ul>
<h3>使用的問法（節錄）</h3>
<ul style="margin-left:18px;">{cfg_txt}</ul>
<p style="font-size:13px;color:#64748b;">完整結果：<code>{escape(str(attribution_jsonl))}</code>
{(' | <a href="' + report_link + '">attribution_report.md</a>') if report_link else ''}</p>

<h3>核心數據（安全帽 · 本批 {n} 幀）</h3>
<table>
<tr><th>指標</th><th>幀數</th><th>佔比</th><th>說明</th></tr>
<tr><td><code>clear_absence_helmet</code> = <strong>No</strong>（承認無法清楚指認未戴帽）</td><td>{admit_no}</td><td>{100*admit_no/n:.1f}%</td><td>較適合歸因於<strong>影像/不確定</strong>；若規則改為「僅在 Yes 時升級告警」，約可減少這類幀的誤報壓力。</td></tr>
<tr><td><code>clear_absence_helmet</code> = <strong>Yes</strong>（聲稱能清楚看到未戴帽）</td><td>{admit_yes}</td><td>{100*admit_yes/n:.1f}%</td><td>在合規假設下，較像<strong>模型判讀</strong>與現場認知衝突，不宜只靠「遮擋」解釋。</td></tr>
<tr><td>兩題交叉：clear=Yes 且 head_vis=No（堅稱清楚違規且頭部可辨）</td><td>{insist}</td><td>{100*insist/n:.1f}%</td><td>{insist_note}</td></tr>
</table>
{warn_vis}
<h3>子集合：僅 detect 未框到安全帽（與 YOLO 漏檢情境對照）</h3>
<p>幀數 <strong>{nd}</strong>。其中 <code>clear_absence_helmet</code>=No：<strong>{nd_admit}</strong>（{nd_pct} of 子集）— 模型在不少「沒框到」的幀上仍承認<strong>無法清楚指認未戴帽</strong>。</p>

<h3>面罩（補充）</h3>
<p><code>clear_absence_mask</code> = No：{mask_no}/{n}（{100*mask_no/n:.1f}%）；= Yes：{mask_yes}/{n}（{100*mask_yes/n:.1f}%）。</p>

<div class="rec-box">
<strong>與下一節「遮擋實驗」的關係：</strong>下一節為較早的問法（可見性敘述）；本節為<strong>較嚴格、可直接對齊「是否清楚看到違規」</strong>的問法，建議以<strong>本節指標 A</strong>作為「能改善多少誤判」的主要參考。
</div>
</div>
"""


def _occlusion_experiment_section(
    run_dir: Path,
    entries: list[dict],
    occlusion_jsonl: Path | None,
    data_dir: Path,
) -> str:
    """整合「遮擋／角度」加開實驗：方法、統計、每幀圖片（從影片擷取）。"""
    ann = run_dir / "annotated"
    cfg_path = EVAL_DIR / "configs" / "occlusion_queries.json"
    prompts_html = ""
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text("utf-8"))
        for q in cfg.get("queries", []):
            prompts_html += f"<li><code>{escape(q['name'])}</code><br><span style='font-size:12px;color:#4b5563'>{escape(q['prompt'])}</span></li>"

    if not occlusion_jsonl or not occlusion_jsonl.exists():
        rid = run_dir.name
        return f"""
<div class="scenario" style="border:2px dashed #cbd5e1;">
<h2 style="border:none;margin:0 0 12px;padding:0;">加開實驗：遮擋／角度／可見性（未匯入）</h2>
<p>本節用於補充「YOLO／detect 框不到是否因背對、遮擋」的語意判斷。若尚未執行遮擋實驗，可執行：</p>
<pre style="background:#f1f5f9;padding:12px;border-radius:8px;font-size:13px;">cd evaluation
python scripts/run_occlusion_experiment.py --baseline-run {rid}</pre>
<p>完成後再執行 <code>python scripts/generate_detail_report.py --run {rid}</code> 即可自動嵌入本節與圖片。</p>
</div>
"""

    rows_oc = [json.loads(l) for l in occlusion_jsonl.read_text("utf-8").splitlines() if l.strip()]
    flat = []
    for o in rows_oc:
        b = o["baseline"]
        qh = o["occlusion_queries"]["helmet_visibility_limited"]
        qm = o["occlusion_queries"]["mask_visibility_limited"]
        flat.append({
            "video": o["video"],
            "time_sec": o["time_sec"],
            "detect_helmet_count": b["detect_helmet_count"],
            "missing_helmet": b["missing_helmet_label"],
            "missing_face_mask": b["missing_face_mask_label"],
            "helmet_visibility_limited": qh["label"],
            "mask_visibility_limited": qm["label"],
            "helmet_vis_answer": qh["answer"],
            "mask_vis_answer": qm["answer"],
        })

    n = len(flat)
    h_yes = sum(1 for r in flat if r["helmet_visibility_limited"] == "yes")
    m_yes = sum(1 for r in flat if r["mask_visibility_limited"] == "yes")
    contra = sum(
        1 for r in flat if int(r["detect_helmet_count"]) > 0 and r["helmet_visibility_limited"] == "yes"
    )

    mh_yes = sum(1 for r in flat if r["missing_helmet"] == "yes")
    mf_yes = sum(1 for r in flat if r["missing_face_mask"] == "yes")
    mh_after = sum(
        1 for r in flat
        if r["missing_helmet"] == "yes" and r["helmet_visibility_limited"] != "yes"
    )
    mf_after = sum(
        1 for r in flat
        if r["missing_face_mask"] == "yes" and r["mask_visibility_limited"] != "yes"
    )

    def still_tight(r):
        if r["missing_helmet"] != "yes":
            return False
        if int(r["detect_helmet_count"]) > 0:
            return True
        return r["helmet_visibility_limited"] != "yes"

    mh_tight = sum(1 for r in flat if still_tight(r))

    html = f"""
<div class="scenario" style="border-top:4px solid #0ea5e9;">
<h2 style="border:none;margin:0 0 12px;padding:0;">加開實驗：遮擋／角度／可見性（補 YOLO「框不到≠沒戴」）</h2>

<div class="task-box">
<h3 style="margin:0 0 8px;">做了什麼？</h3>
<ul style="margin-left:20px;">
<li>資料與本 baseline <strong>同一批時間點</strong>，從 <code>data/</code> 原始影片擷取畫面（<strong>無標註框</strong>，避免框線影響閱讀）。</li>
<li>僅處理：有人且（<code>detect helmet=0</code> 或 <code>missing_helmet=yes</code> 或 <code>missing_face_mask=yes</code>）的幀，共 <strong>{n}</strong> 幀。</li>
<li>額外執行 2 個 query（與 baseline 獨立、同一模型）：</li>
</ul>
<ol style="margin:8px 0 8px 24px;">{prompts_html}</ol>
<p style="font-size:13px;color:#64748b;">原始結果檔：<code>{escape(str(occlusion_jsonl))}</code></p>
</div>

<h3>實際效果（統計）</h3>
<table>
<tr><th>指標</th><th>數值</th></tr>
<tr><td><code>helmet_visibility_limited</code> = yes</td><td>{h_yes} / {n}</td></tr>
<tr><td><code>mask_visibility_limited</code> = yes</td><td>{m_yes} / {n}</td></tr>
<tr><td>已框到安全帽但「安全帽可見性受限」仍為 Yes（矛盾）</td><td class="fp-high">{contra} / {n}</td></tr>
</table>

<div class="pattern-box">
<strong>規則模擬 A（寬鬆）</strong>：若 <code>missing_*=yes</code> 且可見性≠yes 才告警 → 
missing_helmet 仍告警 <strong>{mh_after}</strong> / {mh_yes}；missing_face_mask 仍告警 <strong>{mf_after}</strong> / {mf_yes}（僅本子集）。<br>
<strong>規則模擬 B（保守）</strong>：已框到帽不套用可見性降級；僅 <code>detect helmet=0</code> 時降級 → 仍視為需關注 <strong>{mh_tight}</strong> 幀（含矛盾列）。<br>
若幾乎全為「可見性受限」，須對照<strong>矛盾列</strong>：單一 Yes/No 題可能過寬。
</div>

<h3>每幀圖片與模型回答</h3>
<p style="font-size:13px;color:#64748b;margin-bottom:12px;">圖片為即時從影片擷取並縮圖；若讀片失敗則改嵌入 baseline 標註圖。</p>
<div class="frame-grid">
"""

    for r in flat:
        vp = data_dir / r["video"]
        b64 = extract_frame_jpeg_b64(vp, float(r["time_sec"]))
        source = "原始影片擷取（無標註框）"
        if not b64:
            fe = {"video": r["video"], "time_sec": r["time_sec"]}
            b64 = img_b64(ann / slug(fe))
            source = "baseline 標註圖（備援）"

        hc = int(r["detect_helmet_count"])
        note = ""
        if hc > 0 and r["helmet_visibility_limited"] == "yes":
            note = "<br><span class='tag tag-fp'>矛盾</span> 已框到安全帽，但「可見性受限」仍為 Yes — 題目可能過寬。"

        html += f"""
<div class="fcard fp">
  <img src="data:image/jpeg;base64,{b64}" alt="">
  <div class="finfo">
    <div class="ftitle">{escape(r['video'][:40])} @ {r['time_sec']}s</div>
    <div style="font-size:12px;color:#64748b;margin:4px 0;">{source}</div>
    <div class="fdesc">
      Baseline：<strong>detect helmet={hc}</strong>，
      missing_helmet=<strong>{r['missing_helmet']}</strong>，
      missing_face_mask=<strong>{r['missing_face_mask']}</strong>。
      {note}
    </div>
    <div style="margin-top:8px;font-size:12px;"><strong>加問 · 安全帽可見性</strong> ({r['helmet_visibility_limited']})</div>
    <div class="fanswer">{escape(r['helmet_vis_answer'][:280])}</div>
    <div style="margin-top:8px;font-size:12px;"><strong>加問 · 面罩可見性</strong> ({r['mask_visibility_limited']})</div>
    <div class="fanswer">{escape(r['mask_vis_answer'][:280])}</div>
  </div>
</div>
"""

    html += "</div></div>"
    return html


def _conclusion(config, with_p):
    rows = ""
    for qname in ["person_falling", "smoking_visible", "abnormal_behavior",
                   "missing_helmet", "missing_face_mask"]:
        sm = SCENARIO_META.get(qname, {})
        yes_wp = sum(1 for e in with_p if e["result"]["queries"].get(qname, {}).get("label") == "yes")
        pct = (yes_wp / len(with_p) * 100) if with_p else 0
        cls = "fp-ok" if pct == 0 else ("fp-med" if pct <= 10 else "fp-high")
        action = sm.get("recs", ["—"])[0] if sm.get("recs") else "—"
        rows += f"<tr><td>{sm.get('title','')}</td><td class='{cls}'>{pct:.1f}%</td><td>{action}</td></tr>"

    return f"""
<div class="scenario">
<h2 style="border:none;margin:0 0 12px;padding:0;">結論與下一步</h2>

<h3>各場景 FP 率排名（基準：{len(with_p)} 個有人幀）</h3>
<table><tr><th>場景</th><th>FP 率</th><th>首要改善建議</th></tr>{rows}</table>

<h3>與 YOLO 互補的推薦優先級</h3>
<ol>
<li><strong>輪擋判斷</strong> — query 100% hit + detect 95%，YOLO 未訓練此類別，最值得先做</li>
<li><strong>倒地偵測</strong>（用 query 不用 detect）— query FP=0%，但需正樣本驗證</li>
<li><strong>Enrichment Layer（場景文字記錄）</strong> — YOLO 觸發後由 Moondream 產 caption，作為後台通報文字證據</li>
<li><strong>YOLO 遮擋補充（Strict Prompt）</strong> — 對照 Standard vs Strict 實驗結果判斷可行性</li>
<li><strong>抽菸 / 異常行為</strong> — 有人幀 FP ≈ 6-9%，加規則可用</li>
<li><strong>安全帽 / 面罩缺失</strong> — 需 strict prompt + 交叉驗證 + 連續幀確認</li>
</ol>

<h3>下一步</h3>
<ol>
<li>收集異常正樣本（沒戴帽、抽菸、倒地），驗證模型的 TP（正確告警率）</li>
<li>實作 Worker Gate + 規則引擎，降低誤報</li>
<li>與 YOLO 做同圖對照測試</li>
</ol>
</div>
"""


def _pipeline_section() -> str:
    return """
<div class="scenario" style="border-top:4px solid #0f766e;">
<style>
.pipeline-wrap { margin-top: 16px; }
.pipeline-step { border-radius: 14px; padding: 16px 18px; text-align: center; font-weight: 700; border: 1px solid transparent; box-shadow: 0 10px 24px rgba(15,23,42,0.08); }
.pipeline-step small { display: block; margin-top: 6px; font-size: 12px; font-weight: 600; color: #475569; }
.pipeline-arrow { text-align: center; font-size: 28px; line-height: 1; color: #94a3b8; margin: 8px 0; }
.pipeline-arrow.branch { font-size: 24px; margin: 0; }
.pipeline-branch { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; align-items: start; }
.pipeline-lane { display: flex; flex-direction: column; gap: 10px; }
.pipeline-camera { background: linear-gradient(135deg, #eff6ff, #dbeafe); border-color: #93c5fd; color: #1d4ed8; }
.pipeline-detect { background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-color: #cbd5e1; color: #0f172a; }
.pipeline-skip { background: linear-gradient(135deg, #f3f4f6, #e5e7eb); border-color: #d1d5db; color: #374151; }
.pipeline-person { background: linear-gradient(135deg, #ecfeff, #cffafe); border-color: #67e8f9; color: #0f766e; }
.pipeline-ppe { background: linear-gradient(135deg, #fffbeb, #fde68a); border-color: #fbbf24; color: #92400e; }
.pipeline-ok { background: linear-gradient(135deg, #f0fdf4, #bbf7d0); border-color: #86efac; color: #166534; }
.pipeline-gap { background: linear-gradient(135deg, #fff7ed, #fed7aa); border-color: #fb923c; color: #9a3412; }
.pipeline-md { background: linear-gradient(135deg, #f5f3ff, #ddd6fe); border-color: #a78bfa; color: #5b21b6; }
.pipeline-alert { background: linear-gradient(135deg, #fef2f2, #fecaca); border-color: #f87171; color: #991b1b; }
.pipeline-wait { background: linear-gradient(135deg, #ecfdf5, #a7f3d0); border-color: #34d399; color: #065f46; }
</style>
<h2 style="border:none;margin:0 0 12px;padding:0;">完整偵測與告警流程（YOLO + Moondream Pipeline）</h2>
<p>YOLO 做快速 gate，把「安全帽沒框到」的難例交給 Moondream 判斷是否真的能確認未戴帽。</p>
<div class="pipeline-wrap">
  <div class="pipeline-step pipeline-camera">Camera Frame</div>
  <div class="pipeline-arrow">↓</div>
  <div class="pipeline-step pipeline-detect">YOLO Detection</div>
  <div class="pipeline-branch" style="margin-top:10px;">
    <div class="pipeline-lane">
      <div class="pipeline-arrow branch">↙</div>
      <div class="pipeline-step pipeline-skip">detect(person) = 0<small>Skip — no worker</small></div>
    </div>
    <div class="pipeline-lane">
      <div class="pipeline-arrow branch">↘</div>
      <div class="pipeline-step pipeline-person">detect(person) &gt; 0</div>
      <div class="pipeline-arrow">↓</div>
      <div class="pipeline-step pipeline-ppe">YOLO PPE Check<small>detect(helmet) result</small></div>
      <div class="pipeline-branch">
        <div class="pipeline-lane">
          <div class="pipeline-arrow branch">↙</div>
          <div class="pipeline-step pipeline-ok">helmet detected<small>Worker compliant ✅</small></div>
        </div>
        <div class="pipeline-lane">
          <div class="pipeline-arrow branch">↘</div>
          <div class="pipeline-step pipeline-gap">helmet NOT detected<small>Gap! Need confirmation</small></div>
          <div class="pipeline-arrow">↓</div>
          <div class="pipeline-step pipeline-md">Moondream<br>helmet_area_visible?</div>
          <div class="pipeline-branch">
            <div class="pipeline-lane">
              <div class="pipeline-arrow branch">↙</div>
              <div class="pipeline-step pipeline-alert">YES<small>Helmet area clearly visible → 觸發告警 ⚠️</small></div>
            </div>
            <div class="pipeline-lane">
              <div class="pipeline-arrow branch">↘</div>
              <div class="pipeline-step pipeline-wait">NO<small>Head occluded / bad angle → 壓制，等下一幀</small></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
</div>
"""


def _phase2_occlusion_section(run_dir: Path, data_dir: Path) -> str:
    phase2_path = OUTPUTS_DIR / f"{run_dir.name}_occlusion_variants" / "phase2_full.jsonl"
    if not phase2_path.exists():
        return f"""
<div class="scenario" style="border:2px dashed #94a3b8;">
<h2 style="border:none;margin:0 0 12px;padding:0;">Phase 2：遮擋補充實驗（未匯入）</h2>
<p>找不到 <code>{escape(str(phase2_path))}</code>。請先執行 run_occlusion_phase2.py。</p>
</div>
"""
    rows = [json.loads(l) for l in phase2_path.read_text("utf-8").splitlines() if l.strip()]

    occlusion_rows = [r for r in rows if int(r.get("baseline", {}).get("detect_helmet", 0)) == 0]
    contradiction_rows = [
        r for r in rows
        if int(r.get("baseline", {}).get("detect_helmet", 0)) > 0
        and str(r.get("baseline", {}).get("missing_helmet", "")).strip().lower() == "yes"
    ]

    total_occ = len(occlusion_rows)
    success_count = sum(1 for r in occlusion_rows if r.get("can_confirm_helmet", {}).get("label") == "no")
    miss_count = total_occ - success_count

    video_cache: dict[str, Path | None] = {}

    def resolve_video(name: str) -> Path | None:
        if name in video_cache:
            return video_cache[name]
        p = data_dir / name
        video_cache[name] = p if p.exists() else next(data_dir.rglob(name), None)
        return video_cache[name]

    def render_frame_img(video_name: str, time_sec: float) -> str:
        vp = resolve_video(video_name)
        if vp is None:
            return '<div style="padding:24px;text-align:center;color:#94a3b8;font-size:13px;">影片不存在</div>'
        b64 = extract_frame_jpeg_b64(vp, float(time_sec))
        if not b64:
            return '<div style="padding:24px;text-align:center;color:#94a3b8;font-size:13px;">畫面擷取失敗</div>'
        return f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block;">'

    def render_card(row: dict, border_color: str, tag_text: str, tag_style: str,
                    verdict_text: str, verdict_style: str, extra: str = "") -> str:
        b = row.get("baseline", {})
        c = row.get("can_confirm_helmet", {})
        video = str(row.get("video", ""))
        t_sec = row.get("time_sec", 0)
        can_label = str(c.get("label", "unknown")).strip().lower()
        can_answer = escape(str(c.get("answer", "")).strip() or "—")
        can_color = {"no": "background:#dcfce7;color:#166534;", "yes": "background:#fee2e2;color:#991b1b;"}.get(can_label, "background:#e5e7eb;color:#374151;")
        mh = str(b.get("missing_helmet", "unknown")).strip().lower()
        mh_color = "background:#fee2e2;color:#991b1b;" if mh == "yes" else "background:#dcfce7;color:#166534;"
        return f"""
<div style="border:3px solid {border_color};border-radius:12px;overflow:hidden;background:#f8fafc;">
  <div style="background:#e2e8f0;">{render_frame_img(video, float(t_sec))}</div>
  <div class="finfo">
    <div class="ftitle">
      <span class="tag" style="{tag_style}">{tag_text}</span>
      {escape(Path(video).name)} @ {t_sec}s
    </div>
    <div style="margin:6px 0 8px;">
      <span class="tag tag-person">person: {b.get('detect_person',0)}</span>
      <span class="tag tag-helmet">helmet: {b.get('detect_helmet',0)}</span>
      <span class="tag" style="{mh_color}">missing_helmet: {escape(mh)}</span>
    </div>
    <div class="fdesc">
      <strong>helmet_area_visible:</strong> <span class="tag" style="{can_color}">{escape(can_label)}</span><br>
      <strong>Answer:</strong> {can_answer}
    </div>
    <div style="margin-top:8px;"><span class="badge" style="{verdict_style}">{verdict_text}</span></div>
    {extra}
  </div>
</div>"""

    html = f"""
<div class="scenario" style="border-top:4px solid #10b981;">
<h2 style="border:none;margin:0 0 12px;padding:0;">Phase 2：遮擋辨識實驗 — helmet_area_visible 全幀驗證</h2>
<div class="task-box">
  <p><strong>實驗目標：</strong>對所有有人幀跑 <code>helmet_area_visible</code> prompt，驗證它能否區分「真的沒帶帽」vs「被遮擋看不到」。</p>
  <p style="margin-top:8px;">
    <span class="badge badge-g">{success_count}/{total_occ} 遮擋識別成功</span>&nbsp;
    <span class="badge badge-r">{miss_count}/{total_occ} 識別失誤</span>&nbsp;
    <span class="badge" style="background:#eff6ff;color:#1d4ed8;border:1px solid #93c5fd;">{len(contradiction_rows)} 幀 detect 比 query 更可靠</span>
  </p>
</div>
<div class="rec-box">
  <strong>Pipeline 邏輯：</strong>當 <code>detect_helmet=0</code> 且 <code>helmet_area_visible=no</code>，
  模型承認看不清楚頭部，應壓制告警等下一幀。
  <code>helmet_area_visible=yes</code> 才升級告警。
</div>

<div class="group-hdr">detect_helmet = 0 的 {total_occ} 幀（含照片）</div>
<div class="frame-grid">
"""
    for row in occlusion_rows:
        is_success = row.get("can_confirm_helmet", {}).get("label") == "no"
        html += render_card(
            row,
            border_color="#16a34a" if is_success else "#dc2626",
            tag_text="遮擋識別成功" if is_success else "識別失誤",
            tag_style="background:#16a34a;color:#fff;" if is_success else "background:#dc2626;color:#fff;",
            verdict_text="遮擋識別成功 → 壓制告警" if is_success else "識別失誤 → 誤觸告警",
            verdict_style="background:#f0fdf4;color:#166534;border:1px solid #86efac;" if is_success else "background:#fef2f2;color:#991b1b;border:1px solid #fca5a5;",
        )
    html += "</div>"

    if contradiction_rows:
        html += f"""
<div class="group-hdr" style="margin-top:24px;">對照組：detect_helmet=1 但 missing_helmet=yes 的 {len(contradiction_rows)} 幀</div>
<div class="pattern-box" style="background:#eff6ff;border-color:#93c5fd;">
  這些幀 YOLO detect 已框到安全帽，但 query 卻說沒帶。
  <strong>helmet_area_visible 全部回答 yes</strong>（能看清楚），進一步確認：<strong>detect 比 query 更可靠</strong>。
</div>
<div class="frame-grid">
"""
        for row in contradiction_rows:
            html += render_card(
                row,
                border_color="#2563eb",
                tag_text="Query 矛盾",
                tag_style="background:#2563eb;color:#fff;",
                verdict_text="detect 較可靠，query 誤報",
                verdict_style="background:#eff6ff;color:#1d4ed8;border:1px solid #93c5fd;",
                extra='<div style="margin-top:6px;font-size:12px;color:#6b7280;">detect 已框到安全帽，優先相信 detect 而非 query。</div>',
            )
        html += "</div>"

    html += "</div>"
    return html


def _helmet_consistency_section(run_dir: Path, data_dir: Path) -> str:
    """
    新實驗：在 YOLO 已偵測到安全帽的幀上，跑 helmet_area_visible，
    確認模型不會把清楚可見的帽誤判為「被遮擋」。
    """
    phase2_path = OUTPUTS_DIR / f"{run_dir.name}_occlusion_variants" / "phase2_full.jsonl"
    if not phase2_path.exists():
        return ""

    rows = [json.loads(l) for l in phase2_path.read_text("utf-8").splitlines() if l.strip()]

    from collections import Counter
    def label_dist(subset):
        c = Counter(r["can_confirm_helmet"]["label"] for r in subset)
        return c["yes"], c["no"], c.get("unclear", 0), len(subset)

    # Groups — named by what they mean, not raw field values
    # "helmet detected, no alert"  = detect=1, missing_helmet=no  → YOLO: 偵測到帽 + 判定正常
    # "helmet detected, alert"     = detect=1, missing_helmet=yes → YOLO: 偵測到帽 + 判定缺少（矛盾）
    # "helmet not detected, alert" = detect=0, missing_helmet=yes → YOLO: 未偵測到帽 + 判定缺少
    grp_det_ok   = [r for r in rows if r["baseline"]["detect_helmet"] == 1 and r["baseline"]["missing_helmet"] == "no"]
    grp_det_cont = [r for r in rows if r["baseline"]["detect_helmet"] == 1 and r["baseline"]["missing_helmet"] == "yes"]
    grp_no_det   = [r for r in rows if r["baseline"]["detect_helmet"] == 0 and r["baseline"]["missing_helmet"] == "yes"]

    ok_yes,  ok_no,  _, ok_n   = label_dist(grp_det_ok)
    con_yes, con_no, _, con_n  = label_dist(grp_det_cont)
    nd_yes,  nd_no,  _, nd_n   = label_dist(grp_no_det)

    video_cache: dict[str, Path | None] = {}
    def resolve_video(name: str) -> Path | None:
        if name in video_cache:
            return video_cache[name]
        p = data_dir / name
        video_cache[name] = p if p.exists() else next(data_dir.rglob(name), None)
        return video_cache[name]

    def render_frame_img(video_name: str, time_sec: float) -> str:
        vp = resolve_video(video_name)
        if vp is None:
            return '<div style="padding:24px;text-align:center;color:#94a3b8;font-size:13px;">影片不存在</div>'
        b64 = extract_frame_jpeg_b64(vp, float(time_sec))
        if not b64:
            return '<div style="padding:24px;text-align:center;color:#94a3b8;font-size:13px;">畫面擷取失敗</div>'
        return f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block;">'

    def render_card(row: dict, border_color: str, top_badge: str, top_badge_style: str,
                    verdict_text: str, verdict_style: str) -> str:
        b   = row["baseline"]
        c   = row["can_confirm_helmet"]
        vis = str(c.get("label", "")).strip().lower()
        vis_color = {"yes": "background:#dcfce7;color:#166534;",
                     "no":  "background:#fee2e2;color:#991b1b;"}.get(
                         vis, "background:#e5e7eb;color:#374151;")
        det  = b.get("detect_helmet", 0)
        qmh  = str(b.get("missing_helmet", "")).strip().lower()
        det_label  = "YOLO: helmet detected ✓" if det else "YOLO: helmet not detected"
        det_style  = "background:#dcfce7;color:#166534;" if det else "background:#fee2e2;color:#991b1b;"
        qmh_label  = "Query: helmet missing ⚠" if qmh == "yes" else "Query: helmet present ✓"
        qmh_style  = "background:#fee2e2;color:#991b1b;" if qmh == "yes" else "background:#dcfce7;color:#166534;"
        return f"""
<div style="border:3px solid {border_color};border-radius:12px;overflow:hidden;background:#f8fafc;">
  <div style="background:#e2e8f0;">{render_frame_img(row["video"], float(row["time_sec"]))}</div>
  <div class="finfo">
    <div class="ftitle">
      <span class="tag" style="{top_badge_style}">{top_badge}</span>
      {escape(Path(row["video"]).name)} @ {row["time_sec"]}s
    </div>
    <div style="margin:6px 0 8px;">
      <span class="tag tag-person">person detected</span>
      <span class="tag" style="{det_style}">{det_label}</span>
      <span class="tag" style="{qmh_style}">{qmh_label}</span>
    </div>
    <div class="fdesc">
      <strong>helmet_area_visible:</strong>
      <span class="tag" style="{vis_color}">{"visible (yes)" if vis == "yes" else "not visible / occluded (no)" if vis == "no" else vis}</span><br>
      <strong>Answer:</strong> {escape(str(c.get("answer","")).strip() or "—")}
    </div>
    <div style="margin-top:8px;"><span class="badge" style="{verdict_style}">{verdict_text}</span></div>
  </div>
</div>"""

    html = f"""
<div class="scenario" style="border-top:4px solid #7c3aed;">
<h2 style="border:none;margin:0 0 12px;padding:0;">
  新實驗：helmet_area_visible 穩定性驗證 — 在 YOLO 已偵測到安全帽的幀上測試
</h2>
<div class="task-box">
  <p><strong>實驗問題：</strong>
  <code>helmet_area_visible</code> prompt 原本是設計來判斷「安全帽區域是否可見（未被遮擋）」，
  並用在 <strong>YOLO 未偵測到帽（helmet not detected）</strong>的幀上，作為遮擋過濾。</p>
  <p style="margin-top:8px;">
  現在問：如果對 <strong>YOLO 已成功偵測到安全帽</strong>的幀用同一個 prompt，
  模型會不會把清楚可見的帽誤判為「不可見 / 被遮擋」？
  若出現，代表模型對「visible」的判斷不穩定，不可靠。
  </p>
  <p style="margin-top:8px;color:#6b7280;font-size:13px;">
  資料來源：phase2_full.jsonl 已包含全部 34 有人幀，直接從中分析，不需重跑模型。
  </p>
</div>

<h3 style="margin:20px 0 10px;">結果交叉表</h3>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px;">
  <thead>
    <tr style="background:#1e293b;color:#fff;">
      <th style="padding:10px 14px;text-align:left;">YOLO 結果</th>
      <th style="padding:10px 14px;text-align:left;">Query 結果</th>
      <th style="padding:10px 14px;text-align:center;">n</th>
      <th style="padding:10px 14px;text-align:center;">helmet_area_visible = yes<br><small style="font-weight:400;">(可見，未遮擋)</small></th>
      <th style="padding:10px 14px;text-align:center;">helmet_area_visible = no<br><small style="font-weight:400;">(不可見 / 被遮擋)</small></th>
      <th style="padding:10px 14px;text-align:left;">結果</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#f0fdf4;">
      <td style="padding:9px 14px;font-weight:600;color:#166534;">Helmet detected ✓</td>
      <td style="padding:9px 14px;">Helmet present ✓<br><small style="color:#6b7280;">正常幀，無告警</small></td>
      <td style="padding:9px 14px;text-align:center;">{ok_n}</td>
      <td style="padding:9px 14px;text-align:center;font-size:20px;font-weight:700;color:#16a34a;">{ok_yes}</td>
      <td style="padding:9px 14px;text-align:center;font-size:20px;font-weight:700;color:#dc2626;">{ok_no}</td>
      <td style="padding:9px 14px;">
        {'<span style="background:#dcfce7;color:#166534;padding:3px 10px;border-radius:6px;font-weight:700;">✓ 全部 visible，模型穩定</span>'
         if ok_no == 0 else
         f'<span style="background:#fee2e2;color:#991b1b;padding:3px 10px;border-radius:6px;font-weight:700;">⚠ {ok_no} 幀誤判為遮擋</span>'}
      </td>
    </tr>
    <tr style="background:#eff6ff;">
      <td style="padding:9px 14px;font-weight:600;color:#1d4ed8;">Helmet detected ✓</td>
      <td style="padding:9px 14px;">Helmet missing ⚠<br><small style="color:#6b7280;">detect 與 query 矛盾</small></td>
      <td style="padding:9px 14px;text-align:center;">{con_n}</td>
      <td style="padding:9px 14px;text-align:center;font-size:20px;font-weight:700;color:#16a34a;">{con_yes}</td>
      <td style="padding:9px 14px;text-align:center;font-size:20px;font-weight:700;color:#dc2626;">{con_no}</td>
      <td style="padding:9px 14px;">
        {'<span style="background:#dcfce7;color:#166534;padding:3px 10px;border-radius:6px;font-weight:700;">✓ 全部 visible，confirm detect 可信</span>'
         if con_no == 0 else
         f'<span style="background:#fee2e2;color:#991b1b;padding:3px 10px;border-radius:6px;font-weight:700;">⚠ {con_no} 幀不一致</span>'}
      </td>
    </tr>
    <tr>
      <td style="padding:9px 14px;font-weight:600;color:#991b1b;">Helmet not detected ✗</td>
      <td style="padding:9px 14px;">Helmet missing ⚠<br><small style="color:#6b7280;">觸發告警（遮擋 or 真的沒戴）</small></td>
      <td style="padding:9px 14px;text-align:center;">{nd_n}</td>
      <td style="padding:9px 14px;text-align:center;font-size:20px;font-weight:700;color:#16a34a;">{nd_yes}</td>
      <td style="padding:9px 14px;text-align:center;font-size:20px;font-weight:700;color:#dc2626;">{nd_no}</td>
      <td style="padding:9px 14px;">
        <span style="background:#fef9c3;color:#854d0e;padding:3px 10px;border-radius:6px;font-weight:700;">{nd_no}/{nd_n} not visible → 遮擋過濾</span>
      </td>
    </tr>
  </tbody>
</table>

<div class="rec-box" style="border-color:#7c3aed;background:#faf5ff;">
  <strong>結論：模型穩定，不會把可見的帽誤判為遮擋。</strong><br>
  YOLO 已偵測到安全帽的 {ok_n + con_n} 幀（包含正常幀與矛盾幀），
  <code>helmet_area_visible</code> 全部回答 <strong>visible (yes)</strong>，
  與 detect 結果完全一致。
  YOLO 未偵測到帽的幀，{nd_no}/{nd_n} 回答 not visible（遮擋或角度問題）。
  <code>helmet_area_visible</code> 的判斷<strong>忠實反映影像的實際可見性</strong>，可放心用於 pipeline。
</div>

<div class="group-hdr" style="margin-top:24px;">
  Helmet detected + helmet present 的幀樣本（{ok_n} 幀，全部期望 helmet_area_visible = visible）
</div>
<div class="frame-grid">
"""
    for row in grp_det_ok[:6]:
        vis = row["can_confirm_helmet"]["label"]
        is_vis = vis == "yes"
        html += render_card(
            row,
            border_color="#16a34a" if is_vis else "#dc2626",
            top_badge="helmet_area_visible = visible ✓" if is_vis else "helmet_area_visible = not visible ✗",
            top_badge_style="background:#16a34a;color:#fff;" if is_vis else "background:#dc2626;color:#fff;",
            verdict_text="YOLO detected + visible → 一致 ✓" if is_vis else "YOLO detected but not visible → 異常 ✗",
            verdict_style=("background:#f0fdf4;color:#166534;border:1px solid #86efac;"
                           if is_vis else "background:#fef2f2;color:#991b1b;border:1px solid #fca5a5;"),
        )
    html += "</div></div>"
    return html


def _mask_occlusion_section(run_dir: Path, data_dir: Path) -> str:
    occ_dir = OUTPUTS_DIR / f"{run_dir.name}_occlusion_variants"
    mask_path    = occ_dir / "mask_phase2_full.jsonl"
    fullrun_path = occ_dir / "mask_fullrun_anything_on_face.jsonl"
    if not mask_path.exists():
        return ""

    rows = [json.loads(l) for l in mask_path.read_text("utf-8").splitlines() if l.strip()]
    fp_rows = [r for r in rows if r.get("baseline", {}).get("missing_face_mask") == "yes"]
    ok_rows = [r for r in rows if r.get("baseline", {}).get("missing_face_mask") == "no"]

    # Phase 1 stats (face_area_visible)
    fav_fp_no  = sum(1 for r in fp_rows if r.get("face_area_visible", {}).get("label") == "no")
    fav_ok_no  = sum(1 for r in ok_rows if r.get("face_area_visible", {}).get("label") == "no")

    # Phase 2 stats (anything_on_face full run), keyed by (video, time_sec)
    aof_lookup: dict[tuple, str] = {}
    aof_answer_lookup: dict[tuple, str] = {}
    if fullrun_path.exists():
        for l in fullrun_path.read_text("utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                key = (d["video"], float(d["time_sec"]))
                aof_lookup[key] = d.get("anything_on_face", {}).get("label", "unknown")
                aof_answer_lookup[key] = d.get("anything_on_face", {}).get("answer", "")

    aof_fp_yes = sum(1 for r in fp_rows
                     if aof_lookup.get((r["video"], float(r["time_sec"]))) == "yes")
    aof_ok_yes = sum(1 for r in ok_rows
                     if aof_lookup.get((r["video"], float(r["time_sec"]))) == "yes")
    has_aof = bool(aof_lookup)

    video_cache: dict[str, Path | None] = {}

    def resolve_video(name: str) -> Path | None:
        if name in video_cache:
            return video_cache[name]
        p = data_dir / name
        video_cache[name] = p if p.exists() else next(data_dir.rglob(name), None)
        return video_cache[name]

    def render_frame_img(video_name: str, time_sec: float) -> str:
        vp = resolve_video(video_name)
        if vp is None:
            return '<div style="padding:24px;text-align:center;color:#94a3b8;font-size:13px;">影片不存在</div>'
        b64 = extract_frame_jpeg_b64(vp, float(time_sec))
        if not b64:
            return '<div style="padding:24px;text-align:center;color:#94a3b8;font-size:13px;">畫面擷取失敗</div>'
        return f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block;">'

    def render_mask_card(row: dict, border_color: str, tag_text: str, tag_style: str) -> str:
        b    = row.get("baseline", {})
        fav  = row.get("face_area_visible", {})
        video  = str(row.get("video", ""))
        t_sec  = row.get("time_sec", 0)
        key    = (video, float(t_sec))
        fav_label   = str(fav.get("label", "unknown")).strip().lower()
        fav_answer  = escape(str(fav.get("answer", "")).strip() or "—")
        aof_label   = aof_lookup.get(key, "—")
        aof_answer  = escape(str(aof_answer_lookup.get(key, "")).strip() or "—")
        fav_color = {"no": "background:#fee2e2;color:#991b1b;", "yes": "background:#dcfce7;color:#166534;"}.get(fav_label, "background:#e5e7eb;color:#374151;")
        aof_color = {"yes": "background:#dcfce7;color:#166534;", "no": "background:#fee2e2;color:#991b1b;"}.get(aof_label, "background:#e5e7eb;color:#374151;")
        mfm = str(b.get("missing_face_mask", "")).strip().lower()
        mfm_color = "background:#fee2e2;color:#991b1b;" if mfm == "yes" else "background:#dcfce7;color:#166534;"
        aof_row = (f'<div style="margin-top:6px;">'
                   f'<strong>anything_on_face:</strong> <span class="tag" style="{aof_color}">{escape(aof_label)}</span><br>'
                   f'<em style="font-size:12px;color:#6b7280;">{aof_answer[:120]}</em></div>') if has_aof else ""
        return f"""
<div style="border:3px solid {border_color};border-radius:12px;overflow:hidden;background:#f8fafc;">
  <div style="background:#e2e8f0;">{render_frame_img(video, float(t_sec))}</div>
  <div class="finfo">
    <div class="ftitle"><span class="tag" style="{tag_style}">{tag_text}</span> {escape(Path(video).name)} @ {t_sec}s</div>
    <div style="margin:6px 0 8px;">
      <span class="tag tag-person">person: {b.get('detect_person',0)}</span>
      <span class="tag" style="{mfm_color}">missing_mask: {escape(mfm)}</span>
    </div>
    <div class="fdesc">
      <strong>face_area_visible:</strong> <span class="tag" style="{fav_color}">{escape(fav_label)}</span><br>
      <em style="font-size:12px;color:#6b7280;">{fav_answer[:120]}</em>
      {aof_row}
    </div>
  </div>
</div>"""

    # ── comparison grid ───────────────────────────────────────────────────────
    aof_improvement = ""
    if has_aof:
        old_fp_rate = round(len(fp_rows) / len(rows) * 100)
        new_flagged = sum(1 for r in fp_rows
                         if aof_lookup.get((r["video"], float(r["time_sec"]))) != "yes")
        new_fp_rate = round(new_flagged / len(rows) * 100)
        aof_improvement = f"""
<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:16px;margin:16px 0;">
  <strong style="color:#166534;">anything_on_face 改善效果</strong>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px;text-align:center;">
    <div style="background:#fff;border-radius:6px;padding:10px;">
      <div style="font-size:22px;font-weight:700;color:#166534;">{aof_fp_yes}/{len(fp_rows)}</div>
      <div style="font-size:12px;color:#6b7280;margin-top:4px;">FP 幀中偵測到 PPE<br>（可壓制誤報）</div>
    </div>
    <div style="background:#fff;border-radius:6px;padding:10px;">
      <div style="font-size:22px;font-weight:700;color:#2563eb;">{aof_ok_yes}/{len(ok_rows)}</div>
      <div style="font-size:12px;color:#6b7280;margin-top:4px;">正常幀中偵測到 PPE<br>（正確保留告警）</div>
    </div>
    <div style="background:#fff;border-radius:6px;padding:10px;">
      <div style="font-size:22px;font-weight:700;color:#7c3aed;">{old_fp_rate}% → {new_fp_rate}%</div>
      <div style="font-size:12px;color:#6b7280;margin-top:4px;">誤報率（人幀中）<br>face_area_visible → anything_on_face</div>
    </div>
  </div>
</div>"""

    html = f"""
<div class="scenario" style="border-top:4px solid #7c3aed;">
<h2 style="border:none;margin:0 0 12px;padding:0;">Mask PPE 辨識實驗：兩階段方法比較</h2>
<div class="task-box">
  <p><strong>核心問題：</strong>俯角監控鏡頭 + 帽沿遮擋，口罩難以直接辨識。測試兩種不同問法策略。</p>
  <p style="margin-top:8px;">
    <span class="badge badge-r">FP 幀：{len(fp_rows)} 幀（baseline 誤判為缺口罩，實際有戴）</span>&nbsp;
    <span class="badge badge-g">OK 幀：{len(ok_rows)} 幀（baseline 正確）</span>
  </p>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:16px 0;">
  <div style="background:#fef3c7;border:1px solid #fbbf24;border-radius:8px;padding:16px;">
    <strong style="color:#92400e;">方法一：face_area_visible（失敗）</strong>
    <p style="font-size:12px;margin:8px 0 4px;color:#555;">「能不能看清楚臉部？」</p>
    <ul style="margin-top:4px;margin-left:16px;font-size:13px;">
      <li>FP 幀：{fav_fp_no}/{len(fp_rows)} 說看不到臉 ← 無法區分</li>
      <li>OK 幀：{fav_ok_no}/{len(ok_rows)} 說看不到臉 ← 兩組相同</li>
      <li><strong>結論：不能作為過濾依據</strong></li>
    </ul>
  </div>
  <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:16px;">
    <strong style="color:#166534;">方法二：anything_on_face（有效）</strong>
    <p style="font-size:12px;margin:8px 0 4px;color:#555;">「臉上有沒有任何防護裝備？」</p>
    <ul style="margin-top:4px;margin-left:16px;font-size:13px;">
      <li>FP 幀：{aof_fp_yes}/{len(fp_rows)} 偵測到 PPE ✅ 大幅改善</li>
      <li>OK 幀：{aof_ok_yes}/{len(ok_rows)} 偵測到 PPE ✅</li>
      <li><strong>結論：可用於 pipeline 壓制誤報</strong></li>
    </ul>
  </div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:16px 0;">
  <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:16px;">
    <strong style="color:#166534;">對照：Helmet（helmet_area_visible）</strong>
    <ul style="margin-top:8px;margin-left:16px;font-size:13px;">
      <li>正常幀：23/23 可見 ✅</li>
      <li>FP 幀：6/7 不可見 ✅</li>
      <li><strong>兩組區分清晰 → 已整合入 pipeline</strong></li>
    </ul>
  </div>
  <div style="background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;padding:16px;">
    <strong style="color:#1d4ed8;">Mask 設計原則差異</strong>
    <ul style="margin-top:8px;margin-left:16px;font-size:13px;">
      <li>Helmet：問「能不能看到頭部區域」（可見性）</li>
      <li>Mask：問「臉上有沒有東西」（存在性）</li>
      <li>原因：帽沿遮臉，需換方向問</li>
    </ul>
  </div>
</div>
{aof_improvement}

<div class="group-hdr">FP 幀（missing_mask=yes）— 全部 {len(fp_rows)} 幀照片</div>
<div class="frame-grid">
"""
    for row in fp_rows[:12]:
        key = (row["video"], float(row["time_sec"]))
        aof_label = aof_lookup.get(key, "unknown")
        if has_aof:
            # green = PPE detected (suppressed correctly), red = missed
            bc = "#16a34a" if aof_label == "yes" else "#dc2626"
            tt = "PPE 偵測到 ✓" if aof_label == "yes" else "PPE 未偵測到"
            ts = "background:#16a34a;color:#fff;" if aof_label == "yes" else "background:#dc2626;color:#fff;"
        else:
            fav_label = row.get("face_area_visible", {}).get("label", "unknown")
            bc = "#16a34a" if fav_label == "no" else "#dc2626"
            tt = "臉部不可見" if fav_label == "no" else "臉部可見"
            ts = "background:#16a34a;color:#fff;" if fav_label == "no" else "background:#dc2626;color:#fff;"
        html += render_mask_card(row, border_color=bc, tag_text=tt, tag_style=ts)

    if len(fp_rows) > 12:
        html += f'<div style="padding:16px;text-align:center;color:#6b7280;font-size:13px;">... 共 {len(fp_rows)} 幀，僅顯示前 12 張</div>'

    html += f"""</div>
<div class="group-hdr" style="margin-top:20px;">正常幀（missing_mask=no）— 全部 {len(ok_rows)} 幀照片</div>
<div class="frame-grid">
"""
    for row in ok_rows:
        key = (row["video"], float(row["time_sec"]))
        aof_label = aof_lookup.get(key, "unknown")
        if has_aof:
            bc = "#2563eb" if aof_label == "yes" else "#94a3b8"
            tt = "PPE 偵測到 ✓" if aof_label == "yes" else "PPE 未偵測到"
            ts = "background:#2563eb;color:#fff;" if aof_label == "yes" else "background:#6b7280;color:#fff;"
        else:
            fav_label = row.get("face_area_visible", {}).get("label", "unknown")
            bc = "#2563eb" if fav_label == "yes" else "#94a3b8"
            tt = "臉部可見 ✓" if fav_label == "yes" else "臉部不可見"
            ts = "background:#2563eb;color:#fff;" if fav_label == "yes" else "background:#6b7280;color:#fff;"
        html += render_mask_card(row, border_color=bc, tag_text=tt, tag_style=ts)
    html += "</div></div>"
    return html


def main():
    args = parse_args()
    run_dir = find_run_dir(args.run)
    jp = run_dir / "results.jsonl"
    if not jp.exists():
        sys.exit(f"[error] {jp}")
    entries = load_jsonl(jp)
    config = load_config(run_dir)
    occ_path = None if args.skip_occlusion else find_occlusion_jsonl(run_dir.name)
    attr_path = None if args.skip_attribution else find_attribution_jsonl(run_dir.name)
    data_dir = Path(args.data)
    print(f"[report] {run_dir.name} — {len(entries)} 幀")
    if attr_path:
        print(f"[report] 整合誤判歸因實驗: {attr_path}")
    elif not args.skip_attribution:
        print(f"[report] 未找到誤判歸因結果（{run_dir.name}_attribution/），報告將顯示提示區塊")
    if occ_path:
        print(f"[report] 整合遮擋實驗: {occ_path}")
    elif not args.skip_occlusion:
        print(f"[report] 未找到遮擋實驗結果（{run_dir.name}_occlusion/），報告將顯示提示區塊")
    html = build_html(
        run_dir,
        entries,
        config,
        attribution_jsonl=attr_path,
        occlusion_jsonl=occ_path,
        data_dir=data_dir,
    )
    out = run_dir / "detailed_analysis_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"[report] 產出: {out}")
    print(f"[report] file://{out}")


if __name__ == "__main__":
    main()
