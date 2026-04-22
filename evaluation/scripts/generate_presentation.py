#!/usr/bin/env python3
"""
Generate HTML slide deck from evaluation results.

Usage:
  cd evaluation
  conda run -n moondream python scripts/generate_presentation.py --run 20260421T100247Z
  conda run -n moondream python scripts/generate_presentation.py --run 20260421T100247Z --preview
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR     = Path(__file__).resolve().parents[1]
OUTPUTS_DIR  = EVAL_DIR / "outputs"

# ── colours (light theme) ─────────────────────────────────────────────────────
C = {
    "bg":       "#F8FAFC",
    "card":     "#FFFFFF",
    "card2":    "#F1F5F9",
    "teal":     "#0D9488",   # darker teal, readable on white
    "blue":     "#2563EB",
    "amber":    "#D97706",
    "red":      "#DC2626",
    "text":     "#0F172A",
    "sub":      "#334155",
    "muted":    "#64748B",
    "border":   "#E2E8F0",
    "hdr_bg":   "#0F172A",   # dark header for contrast
    "hdr_text": "#FFFFFF",
    "accent":   "#0D9488",
}


# ── frame extraction ──────────────────────────────────────────────────────────
_video_cache: dict[str, Path | None] = {}
_frame_cache: dict[tuple, str | None] = {}

def _resolve_video(name: str, data_dir: Path) -> Path | None:
    if name in _video_cache:
        return _video_cache[name]
    p = data_dir / name
    _video_cache[name] = p if p.exists() else next(data_dir.rglob(name), None)
    return _video_cache[name]


def frame_b64(video_name: str, time_sec: float, data_dir: Path,
              width: int = 480, quality: int = 75) -> str | None:
    key = (video_name, time_sec, width, quality)
    if key in _frame_cache:
        return _frame_cache[key]
    try:
        import cv2
        from PIL import Image
        vp = _resolve_video(video_name, data_dir)
        if vp is None:
            _frame_cache[key] = None
            return None
        cap = cv2.VideoCapture(str(vp))
        cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            _frame_cache[key] = None
            return None
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ratio = width / img.width
        img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        result = base64.b64encode(buf.getvalue()).decode()
        _frame_cache[key] = result
        return result
    except Exception:
        _frame_cache[key] = None
        return None


def img_tag(b64: str | None, style: str = "") -> str:
    if not b64:
        return f'<div style="background:{C["card2"]};display:flex;align-items:center;justify-content:center;{style}"><span style="color:{C["muted"]};font-size:15px;">影片不存在</span></div>'
    return f'<img src="data:image/jpeg;base64,{b64}" style="display:block;width:100%;height:100%;object-fit:cover;{style}">'


# ── slide wrapper ─────────────────────────────────────────────────────────────
def slide(content: str, notes: str = "") -> str:
    note_html = f'<div class="notes">{escape(notes)}</div>' if notes else ""
    return f'<section class="slide">{content}{note_html}</section>\n'


# ── shared CSS ────────────────────────────────────────────────────────────────
BASE_CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Helvetica Neue','Arial',sans-serif;background:{C["bg"]};color:{C["text"]};}}
.slide{{
  width:1280px;height:720px;background:{C["bg"]};
  display:none;flex-direction:column;overflow:hidden;position:relative;
  flex-shrink:0;
}}
.slide.active{{display:flex;}}
/* header — dark bar at top for contrast */
.hdr{{background:{C["hdr_bg"]};padding:14px 36px 11px;border-bottom:3px solid {C["teal"]};flex-shrink:0;}}
.hdr h1{{color:{C["hdr_text"]};font-size:32px;font-weight:700;}}
.hdr .sub{{color:{C["teal"]};font-size:13px;letter-spacing:2px;margin-top:4px;font-weight:600;}}
/* body */
.body{{flex:1;min-height:0;display:flex;gap:16px;padding:16px 36px;overflow:hidden;}}
/* card */
.card{{background:{C["card"]};border-radius:10px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,0.08);min-height:0;overflow:hidden;}}
/* tags */
.tag{{display:inline-block;border-radius:4px;padding:4px 12px;font-size:14px;font-weight:700;}}
.tag-g{{background:#dcfce7;color:#166534;border:1px solid #86efac;}}
.tag-r{{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;}}
.tag-b{{background:#dbeafe;color:#1e40af;border:1px solid #93c5fd;}}
.tag-a{{background:#fef3c7;color:#92400e;border:1px solid #fcd34d;}}
/* stat box */
.stat-num{{font-size:62px;font-weight:700;line-height:1;}}
.stat-label{{font-size:17px;color:{C["muted"]};margin-top:6px;}}
/* section label */
.sec-lbl{{color:{C["teal"]};font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase;}}
/* frame card */
.fcard{{border-radius:10px;overflow:hidden;background:{C["card"]};display:flex;flex-direction:column;box-shadow:0 1px 4px rgba(0,0,0,0.08);}}
.fcard-info{{padding:10px 12px;font-size:14px;}}
/* nav */
#nav{{
  position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
  display:flex;align-items:center;gap:12px;
  background:rgba(15,23,42,0.88);border:1px solid {C["border"]};
  border-radius:24px;padding:8px 20px;z-index:999;backdrop-filter:blur(8px);
}}
#nav button{{
  background:#1e293b;color:#e2e8f0;border:none;
  border-radius:6px;padding:5px 16px;cursor:pointer;font-size:15px;
}}
#nav button:hover{{background:{C["teal"]};color:#fff;}}
#slide-num{{color:#94a3b8;font-size:14px;min-width:64px;text-align:center;}}
/* progress bar */
#progress{{position:fixed;top:0;left:0;height:4px;background:{C["teal"]};transition:width 0.2s;z-index:1000;}}
/* accent bar (left) */
.accent-l{{position:absolute;left:0;top:0;width:6px;height:100%;background:{C["teal"]};}}
.notes{{display:none;}}
/* prevent flex children from overflowing */
.slide > *,
.body > *,
.body > * > * {{ min-height:0; }}
"""

NAV_JS = """
const slides = document.querySelectorAll('.slide');
let cur = 0;
const prog = document.getElementById('progress');
const num  = document.getElementById('slide-num');
function go(n) {
  slides[cur].classList.remove('active');
  cur = Math.max(0, Math.min(slides.length-1, n));
  slides[cur].classList.add('active');
  prog.style.width = ((cur+1)/slides.length*100)+'%';
  num.textContent  = (cur+1)+' / '+slides.length;
}
document.getElementById('btn-prev').onclick = ()=>go(cur-1);
document.getElementById('btn-next').onclick = ()=>go(cur+1);
document.addEventListener('keydown', e=>{
  if(e.key==='ArrowRight'||e.key==='ArrowDown'||e.key===' ') go(cur+1);
  if(e.key==='ArrowLeft'||e.key==='ArrowUp') go(cur-1);
  if(e.key==='Home') go(0);
  if(e.key==='End')  go(slides.length-1);
});
go(0);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDES
# ═══════════════════════════════════════════════════════════════════════════════

def s_cover(run_name: str, n_frames: int, n_videos: int) -> str:
    stat_boxes = "".join(
        f'<div style="background:{C["card"]};border-radius:12px;padding:24px 28px;'
        f'border-left:5px solid {C["teal"]};box-shadow:0 2px 8px rgba(0,0,0,0.07);">'
        f'<div class="stat-num" style="color:{C["teal"]};">{v}</div>'
        f'<div class="stat-label" style="font-size:21px;">{l}</div></div>'
        for v, l in [(n_videos, "CCTV 影片"), (n_frames, "抽樣 Frames"), ("2", "PPE 目標"), ("3", "實驗階段")]
    )
    return slide(f"""
<div style="width:100%;height:100%;display:flex;position:relative;overflow:hidden;">
  <!-- left dark panel -->
  <div style="width:420px;background:{C['hdr_bg']};display:flex;flex-direction:column;
              justify-content:center;padding:40px 44px;flex-shrink:0;">
    <div style="display:inline-block;background:{C['teal']};border-radius:4px;
                padding:5px 14px;margin-bottom:16px;width:fit-content;">
      <span style="color:#fff;font-size:21px;font-weight:700;letter-spacing:2px;">
        USIG · MOONDREAM EVALUATION
      </span>
    </div>
    <h1 style="color:#fff;font-size:38px;font-weight:700;line-height:1.25;margin-bottom:10px;">
      Moondream<br>PPE 偵測<br>驗證報告
    </h1>
    <p style="color:#94a3b8;font-size:21px;line-height:1.7;">
      Helmet &amp; Mask 辨識效果<br>
      Occlusion 實驗<br>
      Prompt 策略比較
    </p>
    <div style="margin-top:20px;color:#64748b;font-size:21px;">
      {run_name} · Moondream2 2025-01-09
    </div>
  </div>
  <!-- right light panel -->
  <div style="flex:1;background:{C['bg']};display:flex;flex-direction:column;
              justify-content:center;align-items:center;gap:20px;padding:24px;">
    <div class="sec-lbl" style="margin-bottom:0;">本次實驗規模</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;width:100%;">
      {stat_boxes}
    </div>
  </div>
</div>
""")


def s_fp_overview() -> str:
    rows = [
        ("安全帽缺失 (missing_helmet)", "32.4%", "11/34", C["amber"],  "中"),
        ("口罩缺失 (missing_face_mask)", "70.6%", "24/34", C["red"],   "高"),
        ("抽菸偵測 (smoking_visible)",    "8.8%",  " 3/34", C["teal"],  "低"),
        ("異常行為 (abnormal_behavior)", "8.8%",  " 3/34", C["teal"],  "低"),
        ("倒地偵測 (fallen_person)",     "0.0%",  " 0/34", C["teal"],  "✓"),
    ]
    bar_rows = ""
    for name, pct, frac, color, risk in rows:
        pct_val = float(pct.replace("%",""))
        bar_rows += f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
  <div style="width:280px;font-size:20px;color:{C['text']};flex-shrink:0;font-weight:500;">{name}</div>
  <div style="flex:1;background:{C['card2']};border-radius:6px;height:28px;position:relative;overflow:hidden;">
    <div style="width:{pct_val}%;height:100%;background:{color};border-radius:6px;opacity:0.9;"></div>
    <span style="position:absolute;left:10px;top:5px;font-size:21px;font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.4);">{frac} 幀</span>
  </div>
  <div style="width:56px;font-size:22px;font-weight:700;color:{color};text-align:right;">{pct}</div>
  <div style="width:28px;font-size:22px;color:{C['muted']};text-align:center;">{risk}</div>
</div>"""

    return slide(f"""
<div class="hdr">
  <h1>有人幀 FP 率總覽</h1>
  <div class="sub">FALSE POSITIVE RATE · PERSON-GATED FRAMES (n=34)</div>
</div>
<div class="body" style="align-items:flex-start;padding-top:28px;">
  <div style="flex:1;">
    <div class="sec-lbl">各 Query 誤報率（有偵測到人員的幀）</div>
    {bar_rows}
    <div style="margin-top:10px;display:flex;gap:10px;">
      <div style="background:{C['card']};border-left:4px solid {C['red']};
                  border-radius:6px;padding:12px 16px;flex:1;">
        <div style="font-size:22px;color:{C['muted']};margin-bottom:4px;">最高優先改善</div>
        <div style="font-size:21px;font-weight:700;color:{C['text']};">Mask 70.6% FP</div>
        <div style="font-size:22px;color:{C['sub']};margin-top:4px;">
          幾乎等於隨機，不可直接用於告警
        </div>
      </div>
      <div style="background:{C['card']};border-left:4px solid {C['amber']};
                  border-radius:6px;padding:12px 16px;flex:1;">
        <div style="font-size:22px;color:{C['muted']};margin-bottom:4px;">中等優先改善</div>
        <div style="font-size:21px;font-weight:700;color:{C['text']};">Helmet 32.4% FP</div>
        <div style="font-size:22px;color:{C['sub']};margin-top:4px;">
          遮擋 / 角度問題導致，已有解法
        </div>
      </div>
      <div style="background:{C['card']};border-left:4px solid {C['teal']};
                  border-radius:6px;padding:12px 16px;flex:1;">
        <div style="font-size:22px;color:{C['muted']};margin-bottom:4px;">已達可用標準</div>
        <div style="font-size:21px;font-weight:700;color:{C['text']};">其他 Query ≤ 8.8%</div>
        <div style="font-size:22px;color:{C['sub']};margin-top:4px;">
          加交叉驗證規則後可直接整合
        </div>
      </div>
    </div>
  </div>
</div>
""")


def s_helmet_fp_causes(data_dir: Path, phase2_rows: list[dict]) -> str:
    occ_rows = [r for r in phase2_rows if r["baseline"]["detect_helmet"] == 0]

    # 2 success + 1 miss, shown as large 2×2-style grid (left col = 2 stacked, right col = 1 large)
    good = [r for r in occ_rows if r["can_confirm_helmet"]["label"] == "no"][:2]
    miss = [r for r in occ_rows if r["can_confirm_helmet"]["label"] == "yes"][:1]

    def frame_card(r: dict, height: int = 260) -> str:
        label = r["can_confirm_helmet"]["label"]
        is_good = label == "no"
        bc  = C["teal"] if is_good else C["red"]
        tag = "遮擋識別成功 ✓" if is_good else "識別失誤 ✗"
        b64 = frame_b64(r["video"], float(r["time_sec"]), data_dir, width=480)
        return f"""
<div style="border:3px solid {bc};border-radius:10px;overflow:hidden;
            background:{C['card']};display:flex;flex-direction:column;height:{height}px;">
  <div style="flex:1;overflow:hidden;background:{C['card2']};">
    {img_tag(b64, f"height:{height-60}px;object-fit:contain;background:{C['card2']};")}

  </div>
  <div style="height:60px;padding:8px 12px;background:{C['hdr_bg']};
              display:flex;align-items:center;justify-content:space-between;flex-shrink:0;">
    <span class="tag" style="background:{'#dcfce7' if is_good else '#fee2e2'};
          color:{bc};border:1px solid {bc};">{tag}</span>
    <span style="font-size:16px;color:#cbd5e1;font-weight:600;">
      helmet_area_visible = <strong style="color:{bc};">{label}</strong>
    </span>
  </div>
</div>"""

    return slide(f"""
<div class="hdr">
  <h1>Helmet FP 原因分析 — 遮擋 / 角度問題</h1>
  <div class="sub">HELMET · DETECT=0 FRAMES · helmet_area_visible EXPERIMENT (n=7)</div>
</div>
<div class="body" style="gap:16px;">
  <!-- left: stats -->
  <div style="width:240px;display:flex;flex-direction:column;gap:12px;flex-shrink:0;">
    <div class="card" style="border-left:4px solid {C['teal']};">
      <div style="font-size:48px;font-weight:700;color:{C['teal']};line-height:1;">6/7</div>
      <div style="font-size:17px;color:{C['sub']};margin-top:6px;">detect=0 幀<br>正確識別為遮擋</div>
    </div>
    <div class="card" style="border-left:4px solid {C['red']};">
      <div style="font-size:48px;font-weight:700;color:{C['red']};line-height:1;">1/7</div>
      <div style="font-size:17px;color:{C['sub']};margin-top:6px;">識別失誤<br>（不應壓制）</div>
    </div>
    <div class="card" style="background:{C['card2']};flex:1;">
      <div style="font-size:13px;color:{C['muted']};font-weight:700;letter-spacing:1px;margin-bottom:8px;">PIPELINE</div>
      <div style="font-size:16px;color:{C['sub']};line-height:2;">
        detect=0<br>
        ↓ helmet_area_visible<br>
        <span style="color:{C['red']};font-weight:700;">no</span> → 壓制告警<br>
        <span style="color:{C['teal']};font-weight:700;">yes</span> → 升級告警 ⚠️
      </div>
    </div>
  </div>
  <!-- right: 2×2 image grid -->
  <div style="flex:1;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;
              gap:12px;min-width:0;">
    {frame_card(good[0], 260) if len(good) > 0 else ""}
    {frame_card(miss[0], 260) if miss else (frame_card(good[1], 260) if len(good) > 1 else "")}
    {frame_card(good[1], 260) if len(good) > 1 and miss else ""}
    {frame_card(good[2] if len(good) > 2 else good[0], 260)}
  </div>
</div>
""")


def s_helmet_contradiction(data_dir: Path, phase2_rows: list[dict]) -> str:
    contra = [r for r in phase2_rows
              if r["baseline"]["detect_helmet"] > 0
              and r["baseline"]["missing_helmet"] == "yes"][:4]

    cards = ""
    for r in contra:
        label = r["can_confirm_helmet"]["label"]
        b64 = frame_b64(r["video"], float(r["time_sec"]), data_dir, width=420)
        cards += f"""
<div style="position:relative;border:2px solid {C['blue']};border-radius:10px;
            overflow:hidden;background:{C['card']};display:flex;flex-direction:column;">
  <div style="flex:1;overflow:hidden;background:{C['card2']};">
    {img_tag(b64, f"object-fit:contain;background:{C['card2']};")}
  </div>
  <div style="padding:7px 10px;background:{C['hdr_bg']};flex-shrink:0;">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
      <span class="tag" style="background:{C['card']};color:{C['blue']};border:1px solid {C['blue']};font-size:13px;">Query 矛盾</span>
      <div style="font-size:15px;color:{C['hdr_text']};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
        {escape(Path(r['video']).name[:28])} @ {r['time_sec']}s
      </div>
    </div>
    <div style="font-size:15px;color:{C['hdr_text']};margin-top:4px;line-height:1.4;">
      detect=<span style="color:{C['teal']};font-weight:700;">1</span>
      query=<span style="color:{C['red']};font-weight:700;">missing</span>
      helmet_visible=<span style="color:{C['teal']};font-weight:700;">{label}</span>
    </div>
  </div>
</div>"""

    return slide(f"""
<div class="hdr">
  <h1>Detect vs Query 矛盾案例</h1>
  <div class="sub">HELMET · DETECT=1 BUT QUERY SAYS MISSING · n=4 FRAMES</div>
</div>
<div class="body" style="flex-direction:column;gap:10px;">
  <div style="display:flex;gap:14px;flex-shrink:0;">
    <div class="card" style="border-left:4px solid {C['blue']};flex:1;padding:12px 16px;">
      <div style="font-size:20px;font-weight:700;color:{C['text']};margin-bottom:5px;">
        現象：YOLO detect 到安全帽，但 Moondream query 說工人缺少安全帽
      </div>
      <div style="font-size:19px;color:{C['sub']};line-height:1.6;">
        helmet_area_visible 全部回答 <span style="color:{C['teal']};font-weight:700;">yes</span>（能看清楚頭部）
        → 確認 <strong>detect 比 query 更可靠</strong>，query 有誤報傾向
      </div>
    </div>
    <div class="card" style="border-left:4px solid {C['teal']};flex:0 0 210px;padding:12px 14px;">
      <div style="font-size:19px;color:{C['muted']};margin-bottom:5px;">實驗結論</div>
      <div style="font-size:19px;color:{C['sub']};line-height:1.8;">
        detect 出現時 → 信任 detect<br>
        detect = 0 → 才問 query<br>
        query 優先級低於 detect
      </div>
    </div>
  </div>
  <div class="sec-lbl" style="flex-shrink:0;">矛盾幀抽樣（detect 框到帽但 query 誤報缺少）</div>
  <div style="display:grid;grid-template-columns:repeat(2,1fr);grid-template-rows:repeat(2,1fr);
              gap:10px;flex:1;min-height:0;">
    {cards}
  </div>
</div>
""")


def _load_phase2(run_id: str) -> list[dict]:
    p = OUTPUTS_DIR / f"{run_id}_occlusion_variants" / "phase2_full.jsonl"
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]


def _load_mask_data(run_id: str) -> tuple:
    occ_dir = OUTPUTS_DIR / f"{run_id}_occlusion_variants"
    phase1 = {(r["video"], float(r["time_sec"])): r
              for l in (occ_dir / "mask_phase2_full.jsonl").read_text().splitlines()
              if l.strip() for r in [json.loads(l)]}
    aof = {(r["video"], float(r["time_sec"])): r
           for l in (occ_dir / "mask_fullrun_anything_on_face.jsonl").read_text().splitlines()
           if l.strip() for r in [json.loads(l)]}
    fp_keys = [k for k, r in phase1.items() if r["baseline"]["missing_face_mask"] == "yes"]
    ok_keys = [k for k, r in phase1.items() if r["baseline"]["missing_face_mask"] == "no"]
    return phase1, aof, fp_keys, ok_keys


def s_mask_problem(data_dir: Path, mask_data: tuple) -> str:
    phase1, aof, fp_keys, ok_keys = mask_data
    fav_fp_no = sum(1 for k in fp_keys if phase1[k]["face_area_visible"]["label"] == "no")
    fav_ok_no = sum(1 for k in ok_keys if phase1[k]["face_area_visible"]["label"] == "no")
    aof_fp_yes = sum(1 for k in fp_keys if aof.get(k, {}).get("anything_on_face", {}).get("label") == "yes")
    aof_ok_yes = sum(1 for k in ok_keys if aof.get(k, {}).get("anything_on_face", {}).get("label") == "yes")

    return slide(f"""
<div class="hdr">
  <h1>Mask 偵測：兩種方法對比</h1>
  <div class="sub" style="font-size:22px;font-weight:700;letter-spacing:1px;">MASK · face_area_visible vs anything_on_face</div>
</div>
<div class="body" style="gap:16px;align-items:stretch;">
  <div style="flex:1;display:grid;grid-template-columns:1fr 1fr;gap:16px;min-width:0;">
    <!-- Method 1 FAIL -->
    <div style="background:{C['card']};border-radius:12px;padding:16px 20px;
                border-top:6px solid {C['red']};display:flex;flex-direction:column;gap:10px;min-height:0;">
      <div style="font-size:20px;font-weight:800;color:{C['red']};letter-spacing:1px;">方法一（失敗）</div>
      <div style="font-size:22px;font-weight:800;color:{C['text']};line-height:1.3;">問「能不能看到臉部？」</div>
      <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;flex:1;min-height:0;">
        <div style="background:{C['card2']};border-radius:10px;padding:12px 10px;text-align:center;">
          <div style="font-size:44px;font-weight:800;color:{C['red']};line-height:1;">{fav_fp_no}/{len(fp_keys)}</div>
          <div style="font-size:18px;font-weight:700;color:{C['sub']};margin-top:6px;line-height:1.3;">誤報幀回答<br>臉不可見</div>
        </div>
        <div style="font-size:28px;font-weight:800;color:{C['muted']};">=</div>
        <div style="background:{C['card2']};border-radius:10px;padding:12px 10px;text-align:center;">
          <div style="font-size:44px;font-weight:800;color:{C['red']};line-height:1;">{fav_ok_no}/{len(ok_keys)}</div>
          <div style="font-size:18px;font-weight:700;color:{C['sub']};margin-top:6px;line-height:1.3;">正常幀回答<br>臉不可見</div>
        </div>
      </div>
      <div style="background:{C['card2']};border-left:5px solid {C['red']};border-radius:8px;padding:10px 12px;">
        <div style="font-size:20px;font-weight:800;color:{C['red']};line-height:1.35;">兩組結果一樣 → 無法區分誤報與正常幀</div>
      </div>
    </div>
    <!-- Method 2 WIN -->
    <div style="background:{C['card']};border-radius:12px;padding:16px 20px;
                border-top:6px solid {C['teal']};display:flex;flex-direction:column;gap:10px;min-height:0;">
      <div style="font-size:20px;font-weight:800;color:{C['teal']};letter-spacing:1px;">方法二（有效）</div>
      <div style="font-size:22px;font-weight:800;color:{C['text']};line-height:1.3;">問「臉上有沒有防護裝備？」</div>
      <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;flex:1;min-height:0;">
        <div style="background:{C['card2']};border-radius:10px;padding:12px 10px;text-align:center;">
          <div style="font-size:44px;font-weight:800;color:{C['teal']};line-height:1;">{aof_fp_yes}/{len(fp_keys)}</div>
          <div style="font-size:18px;font-weight:700;color:{C['sub']};margin-top:6px;line-height:1.3;">誤報幀偵測到<br>PPE ✓</div>
        </div>
        <div style="font-size:28px;font-weight:800;color:{C['muted']};">≠</div>
        <div style="background:{C['card2']};border-radius:10px;padding:12px 10px;text-align:center;">
          <div style="font-size:44px;font-weight:800;color:{C['blue']};line-height:1;">{aof_ok_yes}/{len(ok_keys)}</div>
          <div style="font-size:18px;font-weight:700;color:{C['sub']};margin-top:6px;line-height:1.3;">正常幀保留<br>正常 PPE</div>
        </div>
      </div>
      <div style="background:{C['card2']};border-left:5px solid {C['teal']};border-radius:8px;padding:10px 12px;">
        <div style="font-size:20px;font-weight:800;color:{C['teal']};line-height:1.35;">可以分開兩組 → FP 率從 70.6% 降到 26.5%</div>
      </div>
    </div>
  </div>
  <!-- Why -->
  <div style="width:260px;background:{C['card']};border-radius:12px;padding:16px 16px;
              border-top:6px solid {C['amber']};display:flex;flex-direction:column;gap:10px;flex-shrink:0;min-height:0;">
    <div style="font-size:20px;font-weight:800;color:{C['amber']};letter-spacing:1px;">根本原因</div>
    <div style="font-size:22px;font-weight:800;color:{C['text']};line-height:1.3;">
      俯角攝影機<br>+ 帽沿遮擋下臉
    </div>
    <div style="background:{C['card2']};border-left:5px solid {C['teal']};border-radius:8px;padding:10px 10px;">
      <div style="font-size:18px;font-weight:800;color:{C['teal']};">Helmet</div>
      <div style="font-size:17px;font-weight:700;color:{C['sub']};margin-top:6px;line-height:1.4;">頭頂最清楚，問「能不能看到」有效</div>
    </div>
    <div style="background:{C['card2']};border-left:5px solid {C['amber']};border-radius:8px;padding:10px 10px;">
      <div style="font-size:18px;font-weight:800;color:{C['amber']};">Mask</div>
      <div style="font-size:17px;font-weight:700;color:{C['sub']};margin-top:6px;line-height:1.4;">下臉被帽沿遮住，改問「有沒有防護裝備」</div>
    </div>
    <div style="background:{C['card2']};border-left:5px solid {C['red']};border-radius:8px;padding:10px 10px;">
      <div style="font-size:17px;font-weight:800;color:{C['red']};line-height:1.4;">不是模型看不到人，而是原本問題設計錯了。</div>
    </div>
  </div>
</div>
""")


def s_mask_fp_photos(data_dir: Path, mask_data: tuple) -> str:
    phase1, aof, fp_keys, ok_keys = mask_data
    sample = fp_keys[:4]

    def card(k: tuple) -> str:
        r = phase1[k]
        fav = phase1[k]["face_area_visible"]["label"]
        aof_lbl = aof.get(k, {}).get("anything_on_face", {}).get("label", "—")
        aof_ans = aof.get(k, {}).get("anything_on_face", {}).get("answer", "")[:55]
        b64 = frame_b64(r["video"], float(r["time_sec"]), data_dir, width=460)
        detected = aof_lbl == "yes"
        bc = C["teal"] if detected else C["red"]
        return f"""
<div style="border:2px solid {bc};border-radius:8px;overflow:hidden;
            background:{C['card']};display:flex;flex-direction:column;">
  <div style="flex:1;overflow:hidden;background:{C['card2']};">
    {img_tag(b64, f"object-fit:contain;background:{C['card2']};")}
  </div>
  <div style="padding:7px 10px;flex-shrink:0;background:{C['hdr_bg']};">
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:4px;">
      <span class="tag" style="font-size:12px;background:{C['card']};color:{C['amber']};border:1px solid {C['amber']};">face_area_visible={fav}</span>
      <span class="tag" style="background:{C['card']};color:{bc};border:1px solid {bc};font-size:12px;">
        {f'anything_on_face=yes → 告警取消' if detected else 'anything_on_face=no → 告警保留'}
      </span>
    </div>
    <div style="font-size:13px;color:#cbd5e1;">{escape(Path(r['video']).name[:28])} @ {r['time_sec']}s</div>
  </div>
</div>"""

    cards_html = "".join(card(k) for k in sample)
    return slide(f"""
<div class="hdr">
  <h1>Mask 誤報幀實際照片（anything_on_face 驗證結果）</h1>
  <div class="sub">FP FRAMES · GREEN = FACE COVERING CONFIRMED → SUPPRESS · RED = UNCONFIRMED → ALERT REMAINS</div>
</div>
<div class="body" style="gap:12px;">
  <div style="display:grid;grid-template-columns:repeat(2,1fr);grid-template-rows:repeat(2,1fr);
              gap:12px;flex:1;min-height:0;">
    {cards_html}
  </div>
</div>
""")


def s_mask_improvement_stats(data_dir: Path, mask_data: tuple) -> str:
    phase1, aof, fp_keys, ok_keys = mask_data
    aof_fp_yes = sum(1 for k in fp_keys if aof.get(k, {}).get("anything_on_face", {}).get("label") == "yes")
    aof_ok_yes = sum(1 for k in ok_keys if aof.get(k, {}).get("anything_on_face", {}).get("label") == "yes")
    aof_fp_no  = len(fp_keys) - aof_fp_yes

    return slide(f"""
<div class="hdr">
  <h1>Mask — anything_on_face 全量結果</h1>
  <div class="sub">FULL RUN · n=34 PERSON FRAMES · FP RATE 70.6% → 26.5%</div>
</div>
<div class="body" style="gap:20px;align-items:stretch;">
  <!-- big numbers -->
  <div style="display:flex;flex-direction:column;gap:16px;flex:1;justify-content:center;">
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">
      <div style="background:{C['card']};border-radius:10px;padding:20px;text-align:center;
                  border-top:4px solid {C['teal']};">
        <div style="font-size:56px;font-weight:700;color:{C['teal']};line-height:1;">{aof_fp_yes}/{len(fp_keys)}</div>
        <div style="font-size:15px;color:{C['muted']};margin-top:8px;line-height:1.6;">
          YOLO: missing_face_mask=<b style="color:{C['red']}">yes</b><br>
          Moondream: anything_on_face=<b style="color:{C['teal']}">yes</b><br>
          → 告警取消
        </div>
      </div>
      <div style="background:{C['card']};border-radius:10px;padding:20px;text-align:center;
                  border-top:4px solid {C['blue']};">
        <div style="font-size:56px;font-weight:700;color:{C['blue']};line-height:1;">{aof_ok_yes}/{len(ok_keys)}</div>
        <div style="font-size:15px;color:{C['muted']};margin-top:8px;line-height:1.6;">
          YOLO: missing_face_mask=<b style="color:{C['teal']}">no</b><br>
          Moondream: anything_on_face=<b style="color:{C['teal']}">yes</b><br>
          → 兩者一致，無告警
        </div>
      </div>
      <div style="background:{C['card']};border-radius:10px;padding:20px;text-align:center;
                  border-top:4px solid {C['red']};">
        <div style="font-size:56px;font-weight:700;color:{C['red']};line-height:1;">{aof_fp_no}/{len(fp_keys)}</div>
        <div style="font-size:15px;color:{C['muted']};margin-top:8px;line-height:1.6;">
          YOLO: missing_face_mask=<b style="color:{C['red']}">yes</b><br>
          Moondream: anything_on_face=<b style="color:{C['red']}">no</b><br>
          → 告警保留（仍為 FP）
        </div>
      </div>
    </div>
    <!-- FP rate comparison -->
    <div style="background:{C['card']};border-radius:10px;padding:20px 24px;">
      <div style="font-size:15px;color:{C['muted']};margin-bottom:12px;font-weight:600;letter-spacing:1px;">FP 率比較（人員幀中）</div>
      <div style="display:flex;align-items:center;gap:20px;">
        <div style="flex:1;background:{C['card2']};border-radius:6px;overflow:hidden;height:36px;position:relative;">
          <div style="width:70.6%;height:100%;background:{C['red']};opacity:0.85;"></div>
          <span style="position:absolute;left:10px;top:7px;font-size:18px;font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.4);">face_area_visible</span>
        </div>
        <div style="font-size:26px;font-weight:700;color:{C['red']};width:60px;text-align:right;">70.6%</div>
      </div>
      <div style="display:flex;align-items:center;gap:20px;margin-top:10px;">
        <div style="flex:1;background:{C['card2']};border-radius:6px;overflow:hidden;height:36px;position:relative;">
          <div style="width:26.5%;height:100%;background:{C['teal']};opacity:0.85;"></div>
          <span style="position:absolute;left:10px;top:7px;font-size:18px;font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.4);">anything_on_face</span>
        </div>
        <div style="font-size:26px;font-weight:700;color:{C['teal']};width:60px;text-align:right;">26.5%</div>
      </div>
    </div>
  </div>
  <!-- limitation box -->
  <div style="width:280px;background:{C['card']};border-radius:10px;padding:20px 18px;
              border-top:4px solid {C['amber']};display:flex;flex-direction:column;gap:14px;flex-shrink:0;">
    <div style="font-size:13px;font-weight:700;color:{C['amber']};letter-spacing:1px;">殘留 FP 原因</div>
    <div style="font-size:19px;color:{C['text']};font-weight:600;">9/24 幀仍偵測不到</div>
    <div style="font-size:18px;color:{C['sub']};line-height:1.9;">
      → 工人背對鏡頭<br>
      → 臉部完全不可見<br>
      → 非 Prompt 問題<br>
      → 鏡頭角度死角
    </div>
    <div style="background:#fef3c7;border-radius:6px;padding:10px 12px;font-size:16px;color:#92400e;margin-top:4px;">
      硬體層面問題，需補充側面鏡頭覆蓋
    </div>
  </div>
</div>
""")


def _mask_frame_grid(data_dir: Path, mask_data: tuple, detected: bool) -> str:
    phase1, aof, fp_keys, _ = mask_data
    if detected:
        keys = [k for k in fp_keys if aof.get(k, {}).get("anything_on_face", {}).get("label") == "yes"][:6]
        border = C["teal"]
        tag_cls = "tag-g"
        tag_text = "anything_on_face = yes → 告警取消"
        title = "Mask — YOLO 告警但 anything_on_face = yes 的幀（告警取消）"
        sub = f"missing_face_mask=yes (YOLO) → anything_on_face=yes (Moondream) → alert cancelled · {len(keys)}/24"
    else:
        keys = [k for k in fp_keys if aof.get(k, {}).get("anything_on_face", {}).get("label") != "yes"][:6]
        border = C["red"]
        tag_cls = "tag-r"
        tag_text = "anything_on_face = no → 告警保留"
        title = "Mask — YOLO 告警且 anything_on_face = no 的幀（告警保留）"
        sub = "missing_face_mask=yes (YOLO) → anything_on_face=no (Moondream) → alert kept · 9/24 FP FRAMES REMAIN"

    def card(k: tuple) -> str:
        r = phase1[k]
        b64 = frame_b64(r["video"], float(r["time_sec"]), data_dir, width=400)
        return f"""
<div style="border:3px solid {border};border-radius:8px;overflow:hidden;
            background:{C['card']};display:flex;flex-direction:column;">
  <div style="flex:1;overflow:hidden;background:{C['card2']};">
    {img_tag(b64, "object-fit:cover;")}
  </div>
  <div style="padding:6px 8px;flex-shrink:0;background:{C['hdr_bg']};">
    <span class="tag {tag_cls}" style="font-size:12px;">{tag_text}</span>
    <div style="font-size:13px;color:#cbd5e1;margin-top:3px;">{escape(Path(r['video']).name[:28])} @ {r['time_sec']}s</div>
  </div>
</div>"""

    return slide(f"""
<div class="hdr">
  <h1>{title}</h1>
  <div class="sub">{sub}</div>
</div>
<div class="body" style="gap:10px;">
  <div style="display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(2,1fr);
              gap:10px;flex:1;min-height:0;">
    {"".join(card(k) for k in keys)}
  </div>
</div>
""")


def s_mask_detected_frames(data_dir: Path, mask_data: tuple) -> str:
    return _mask_frame_grid(data_dir, mask_data, detected=True)


def s_mask_missed_frames(data_dir: Path, mask_data: tuple) -> str:
    return _mask_frame_grid(data_dir, mask_data, detected=False)


def s_summary() -> str:
    return slide(f"""
<div class="hdr">
  <h1>實驗結論總覽</h1>
  <div class="sub">SUMMARY · HELMET + MASK · THREE EXPERIMENT PHASES</div>
</div>
<div class="body" style="flex-direction:column;gap:16px;padding-top:24px;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;flex:1;">
    <!-- helmet -->
    <div style="background:{C['card']};border-radius:10px;padding:12px 16px;
                border-top:4px solid {C['teal']};">
      <div style="font-size:22px;font-weight:700;color:{C['teal']};margin-bottom:12px;letter-spacing:1px;">
        HELMET ✓ Pipeline 已驗證
      </div>
      <div style="font-size:21px;color:{C['sub']};line-height:2;">
        detect_helmet = 0 → helmet_area_visible<br>
        &nbsp;&nbsp;&nbsp;= no → 壓制（遮擋）<span style="float:right;color:{C['teal']};">6/7</span><br>
        &nbsp;&nbsp;&nbsp;= yes → 升級告警<br>
        detect 有結果時 → 信任 detect 優先<br>
        query 只做補充確認
      </div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid {C['border']};
                  display:flex;gap:12px;">
        <div style="text-align:center;flex:1;">
          <div style="font-size:22px;font-weight:700;color:{C['teal']};">32.4%</div>
          <div style="font-size:21px;color:{C['muted']};">原始 FP 率</div>
        </div>
        <div style="text-align:center;flex:1;">
          <div style="font-size:22px;font-weight:700;color:{C['teal']};">~5%</div>
          <div style="font-size:21px;color:{C['muted']};">Pipeline 後估計</div>
        </div>
      </div>
    </div>
    <!-- mask -->
    <div style="background:{C['card']};border-radius:10px;padding:12px 16px;
                border-top:4px solid {C['amber']};">
      <div style="font-size:22px;font-weight:700;color:{C['amber']};margin-bottom:12px;letter-spacing:1px;">
        MASK ⚠ 部分改善，仍有限制
      </div>
      <div style="font-size:21px;color:{C['sub']};line-height:2;">
        face_area_visible → 無效（兩組無差異）<br>
        anything_on_face → 有效<span style="float:right;color:{C['teal']};">15/24</span><br>
        &nbsp;&nbsp;&nbsp;= yes → 告警取消 (15/24)<br>
        &nbsp;&nbsp;&nbsp;= no &nbsp;→ 告警保留 (9/24)<br>
        FP 率：70.6% → 26.5%
      </div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid {C['border']};
                  display:flex;gap:12px;">
        <div style="text-align:center;flex:1;">
          <div style="font-size:22px;font-weight:700;color:{C['red']};">70.6%</div>
          <div style="font-size:21px;color:{C['muted']};">原始 FP 率</div>
        </div>
        <div style="text-align:center;flex:1;">
          <div style="font-size:22px;font-weight:700;color:{C['amber']};">26.5%</div>
          <div style="font-size:21px;color:{C['muted']};">改善後 FP 率</div>
        </div>
      </div>
    </div>
  </div>
  <div style="background:{C['card2']};border-radius:8px;padding:12px 18px;
              border-left:4px solid {C['blue']};">
    <div style="font-size:21px;color:{C['sub']};line-height:1.8;">
      <strong style="color:{C['text']};">設計原則：</strong>
      &nbsp;Helmet 問「能不能看到頭部區域」（可見性）&nbsp;·&nbsp;
      Mask 問「臉上有沒有東西」（存在性）&nbsp;·&nbsp;
      原因：俯角鏡頭，帽沿遮臉，需針對 PPE 位置換問法
    </div>
  </div>
</div>
""")


# ═══════════════════════════════════════════════════════════════════════════════
def build_presentation(run_dir: Path, data_dir: Path) -> str:
    entries = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    n_frames = len(entries)
    videos = len({e["video"] for e in entries})

    phase2    = _load_phase2(run_dir.name)
    mask_data = _load_mask_data(run_dir.name)

    all_slides = "".join([
        s_cover(run_dir.name, n_frames, videos),
        s_fp_overview(),
        s_helmet_fp_causes(data_dir, phase2),
        s_helmet_contradiction(data_dir, phase2),
        s_mask_problem(data_dir, mask_data),
        s_mask_fp_photos(data_dir, mask_data),
        s_mask_improvement_stats(data_dir, mask_data),
        s_mask_detected_frames(data_dir, mask_data),
        s_mask_missed_frames(data_dir, mask_data),
        s_summary(),
    ])

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>Moondream PPE 評估報告</title>
<style>
{BASE_CSS}
</style>
</head>
<body>
<div id="progress" style="width:0%;"></div>
{all_slides}
<div id="nav">
  <button id="btn-prev">◀ 上一頁</button>
  <span id="slide-num">1 / 1</span>
  <button id="btn-next">下一頁 ▶</button>
  <span style="color:{C['muted']};font-size:14px;margin-left:8px;">← → 方向鍵導覽</span>
</div>
<script>{NAV_JS}</script>
</body>
</html>"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="20260421T100247Z")
    p.add_argument("--data", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--out", default="")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = OUTPUTS_DIR / args.run
    data_dir = Path(args.data)
    if not run_dir.exists():
        raise SystemExit(f"[error] run not found: {run_dir}")

    print(f"[slides] building presentation for {args.run} ...")
    html = build_presentation(run_dir, data_dir)

    out = Path(args.out) if args.out else run_dir / "presentation.html"
    out.write_text(html, encoding="utf-8")
    print(f"[slides] output: {out}")
    print(f"[slides] file://{out}")


if __name__ == "__main__":
    main()
