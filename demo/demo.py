#!/usr/bin/env python3
"""Moondream — behavior-detection demo for stakeholder briefings.

Standalone showcase, fully isolated from ``evaluation/``. Built to look like a real
monitoring tool (not a hand-drawn mock UI). Three real surfaces:

  • Window "LIVE FEED"  — the source video, played smoothly on the main thread with a
    minimal CCTV-style overlay (REC dot, filename, timecode, progress). No verdicts here:
    the live feed runs seconds ahead of what the model is analysing, so mixing them lies.
  • Window "ANALYSIS"   — one frame at a time, in a strict self-paced cadence: grab the
    current frame → analyse it (the verdict is cleared while scanning) → show the verdict
    on that same frame, held briefly (--hold, default 0.5s) → grab a fresh frame → repeat.
    The border stays one constant neutral color (no flashing); verdict color lives in the
    labels. Average latency per frame (input→output) is shown at the bottom.
  • The terminal        — a live ``rich`` results table streaming every inference
    (time · SMOKING · FALL · latency), which reads as a real engineering tool.

Inference runs on a background thread; the video never stalls waiting for the model.

Usage:
    python demo/demo.py                              # file picker (multi-select)
    python demo/demo.py --video A.mp4 B.mp4          # explicit order, loops at end
    python demo/demo.py --folder data/0520GrokVideo  # whole folder, name-sorted
    python demo/demo.py --folder ... --speed 0.5     # slow short clips down
    python demo/demo.py --folder ... --max-width 512 # ~3x faster inference on MPS
    python demo/demo.py --selftest --folder ...      # render layout preview PNGs

Keys (focus either window):  SPACE pause/resume   N next clip   Q/Esc quit
"""

from __future__ import annotations

import argparse
import math
import re
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
WINDOW_LIVE = "Moondream — LIVE FEED"
WINDOW_ANALYSIS = "Moondream — ANALYSIS"

# Lock the demo to the two behaviors already validated in evaluation/ (direct prompts).
QUERIES: dict[str, dict[str, str]] = {
    "smoking": {
        "label": "SMOKING",
        "prompt": (
            "Is any visible person smoking or holding a cigarette in this image? "
            "Answer yes or no first, then briefly say what visual evidence you see."
        ),
    },
    "fall": {
        "label": "FALL",
        "prompt": (
            "Is anyone lying on the ground, collapsed, or fallen down in this image? "
            "Answer yes or no first, then briefly say what you see."
        ),
    },
}
ORDER = ["smoking", "fall"]

# verdict label -> (display word, RGB color)
VERDICT = {
    "yes": ("DETECTED", (248, 81, 73)),
    "no": ("CLEAR", (46, 160, 67)),
    "unclear": ("UNCERTAIN", (210, 153, 34)),
    "unknown": ("—", (139, 148, 158)),
}

# Palette (RGB). cv2 helpers convert to BGR on the fly.
BG = (10, 13, 18)
INNER = (6, 8, 12)
STRIP = (16, 20, 26)
TEXT = (233, 238, 244)
DIM = (150, 158, 168)
FAINT = (96, 104, 114)
ACCENT = (56, 139, 253)
DANGER = (248, 81, 73)
OK = (46, 160, 67)
AMBER = (210, 153, 34)

# Window canvases (two side by side ≈ 1920 wide).
LW, LH = 960, 720
AW, AH = 960, 720
BWID = 14  # analysis colored border thickness (wider for projected-screen visibility)
A_TOP = BWID + 42
A_BOT = AH - BWID - 46
A_FRAME = (BWID, A_TOP, AW - BWID, A_BOT)
L_FRAME = (8, 8, LW - 8, LH - 8)

FONT_FILE = "/System/Library/Fonts/HelveticaNeue.ttc"
FONT_FALLBACK = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_INDEX = {"regular": 0, "bold": 1, "medium": 10, "light": 7}
_font_cache: dict[tuple[int, str], ImageFont.ImageFont] = {}
CV_FONT = cv2.FONT_HERSHEY_DUPLEX


def font(size: int, weight: str = "regular") -> ImageFont.ImageFont:
    key = (size, weight)
    if key in _font_cache:
        return _font_cache[key]
    try:
        f: ImageFont.ImageFont = ImageFont.truetype(FONT_FILE, size, index=FONT_INDEX[weight])
    except Exception:
        try:
            f = ImageFont.truetype(FONT_FALLBACK, size)
        except Exception:
            f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def classify_answer(answer: str) -> str:
    text = answer.strip().lower()
    if re.match(r"^(yes\b|yes[,.!\s])", text):
        return "yes"
    if re.match(r"^(no\b|no[,.!\s])", text):
        return "no"
    if re.match(r"^(unclear\b|unclear[,.!\s])", text):
        return "unclear"
    return "unknown"


def fmt_mmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def alert_level(results: dict | None) -> str:
    """red / amber / green / none from a results dict."""
    if not results:
        return "none"
    labels = {results[k]["label"] for k in results}
    if "yes" in labels:
        return "red"
    if "unclear" in labels:
        return "amber"
    if "no" in labels:
        return "green"
    return "none"


LEVEL_COLOR = {"red": DANGER, "amber": AMBER, "green": OK, "none": (40, 46, 54)}


# --------------------------------------------------------------------------
# cv2 drawing helpers (live feed — fast, technical HUD look)
# --------------------------------------------------------------------------

def _fit_paste_cv(canvas: np.ndarray, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h, w = frame_bgr.shape[:2]
    if w == 0 or h == 0:
        return
    scale = min(bw / w, bh / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = x0 + (bw - nw) // 2, y0 + (bh - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized


def _cv_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, rgb,
             thick: int = 1, anchor: str = "lt") -> tuple[int, int]:
    (tw, th), _ = cv2.getTextSize(text, CV_FONT, scale, thick)
    x, y = org
    if "r" in anchor:
        x -= tw
    if "c" in anchor:
        x -= tw // 2
    cv2.putText(img, text, (int(x), int(y)), CV_FONT, scale, (rgb[2], rgb[1], rgb[0]), thick, cv2.LINE_AA)
    return tw, th


def render_live(frame_bgr, info: dict) -> np.ndarray:
    canvas = np.empty((LH, LW, 3), np.uint8)
    canvas[:] = INNER[::-1]
    if frame_bgr is not None:
        _fit_paste_cv(canvas, frame_bgr, L_FRAME)

    # translucent top/bottom strips for legible overlay text
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (LW, 50), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, LH - 58), (LW, LH), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    # REC + filename (top-left)
    cv2.circle(canvas, (28, 26), 7, DANGER[::-1], -1)
    w, _ = _cv_text(canvas, "REC", (44, 32), 0.6, DANGER, 1)
    name = info["video_name"]
    if len(name) > 46:
        name = name[:43] + "…"
    _cv_text(canvas, name, (44 + w + 14, 32), 0.55, TEXT, 1)

    # clip index + timecode (top-right)
    i, n = info["playlist_pos"]
    _cv_text(canvas, f"CLIP {i}/{n}", (LW - 18, 32), 0.55, DIM, 1, anchor="rt")

    # progress bar + timecode + status (bottom)
    frac = 0.0 if info["video_dur"] <= 0 else min(1.0, info["video_time"] / info["video_dur"])
    bx0, bx1, by = 18, LW - 18, LH - 40
    cv2.line(canvas, (bx0, by), (bx1, by), (52, 60, 70), 4, cv2.LINE_AA)
    fill = int((bx1 - bx0) * frac)
    if fill > 2:
        cv2.line(canvas, (bx0, by), (bx0 + fill, by), ACCENT[::-1], 4, cv2.LINE_AA)
        cv2.circle(canvas, (bx0 + fill, by), 6, ACCENT[::-1], -1, cv2.LINE_AA)
    tc = f"{fmt_mmss(info['video_time'])} / {fmt_mmss(info['video_dur'])}"
    _cv_text(canvas, tc, (18, LH - 14), 0.55, TEXT, 1)
    if info["paused"]:
        _cv_text(canvas, "PAUSED", (LW - 18, LH - 14), 0.55, AMBER, 1, anchor="rt")
        cv2.rectangle(canvas, (LW - 18 - 96, LH - 24), (LW - 18 - 92, LH - 16), AMBER[::-1], -1)
        cv2.rectangle(canvas, (LW - 18 - 90, LH - 24), (LW - 18 - 86, LH - 16), AMBER[::-1], -1)
    else:
        _cv_text(canvas, "PLAYING", (LW - 18, LH - 14), 0.55, OK, 1, anchor="rt")
        pts = np.array([[LW - 18 - 104, LH - 24], [LW - 18 - 104, LH - 16], [LW - 18 - 96, LH - 20]], np.int32)
        cv2.fillPoly(canvas, [pts], OK[::-1], cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------
# PIL drawing helpers (analysis — crisp, big, readable verdicts)
# --------------------------------------------------------------------------

def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def _fit_paste_pil(canvas: Image.Image, img: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    iw, ih = img.size
    if iw == 0 or ih == 0:
        return
    scale = min(bw / iw, bh / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (x0 + (bw - nw) // 2, y0 + (bh - nh) // 2))


def _verdict_label(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, vlabel: str) -> int:
    """Large verdict plate drawn on the analysed frame. Returns next y."""
    word, color = VERDICT[vlabel]
    f_lab, f_word = font(19, "medium"), font(26, "bold")
    pad_x, pad_y, bar, gap = 16, 11, 6, 14
    w_lab = int(draw.textlength(label, font=f_lab))
    w_word = int(draw.textlength(word, font=f_word))
    h = 52
    total = bar + pad_x + w_lab + gap + w_word + pad_x
    draw.rounded_rectangle([x, y, x + total, y + h], radius=8, fill=(10, 13, 18), outline=(46, 53, 62), width=1)
    draw.rounded_rectangle([x, y, x + bar + 8, y + h], radius=8, fill=color)
    draw.rectangle([x + bar, y, x + bar + 8, y + h], fill=color)
    tx = x + bar + pad_x
    draw.text((tx, y + 8), label, font=f_lab, fill=TEXT)
    draw.text((tx + w_lab + gap, y + pad_y), word, font=f_word, fill=color)
    return y + h + 12


def render_analysis_base(av: dict, model_name: str, device: str) -> np.ndarray:
    """Static part of the analysis window for one phase (everything except the timer).

    The analysis window is a strict two-phase machine, so it can never show a verdict
    that does not belong to the frame on screen:
      • "analyzing" — the frame currently being scanned, NO verdict yet.
      • "result"    — that same frame WITH its verdict, held long enough to read.
      • "standby"   — nothing analysed yet.
    The border stays one constant neutral color in every phase, so the window never
    "suddenly changes color"; the verdict color lives in the labels alone.
    """
    phase = av["phase"]
    canvas = Image.new("RGB", (AW, AH), BG)
    draw = ImageDraw.Draw(canvas)
    border = (44, 50, 58)

    # frame area
    draw.rectangle(list(A_FRAME), fill=INNER)
    if av["frame"] is not None:
        _fit_paste_pil(canvas, _bgr_to_pil(av["frame"]), A_FRAME)
        draw = ImageDraw.Draw(canvas)

    # verdict plates ONLY in the result phase (never while analysing)
    if phase == "result":
        ly = A_FRAME[1] + 16
        for key in ORDER:
            vlabel = av["results"][key]["label"] if key in av["results"] else "unknown"
            ly = _verdict_label(draw, A_FRAME[0] + 16, ly, QUERIES[key]["label"], vlabel)
    elif phase == "standby":
        msg = "Awaiting first inference…"
        draw.text(((AW - draw.textlength(msg, font=font(20))) / 2, AH / 2 - 12),
                  msg, font=font(20), fill=FAINT)

    # constant neutral border (never flashes color)
    for i in range(BWID):
        draw.rectangle([i, i, AW - 1 - i, AH - 1 - i], outline=border, width=1)

    # top strip
    draw.rectangle([BWID, BWID, AW - BWID, A_TOP], fill=STRIP)
    draw.text((BWID + 14, BWID + 11), "ANALYSIS", font=font(15, "bold"), fill=TEXT)
    tx = BWID + 14 + draw.textlength("ANALYSIS", font=font(15, "bold")) + 12
    if phase == "analyzing":
        draw.text((tx, BWID + 13), f"scanning frame @ {fmt_mmss(av['video_time'])}",
                  font=font(14), fill=ACCENT)
    elif phase == "result":
        draw.text((tx, BWID + 13), f"frame @ {fmt_mmss(av['video_time'])}", font=font(14), fill=DIM)
    right = f"{model_name} · {device.upper()}"
    draw.text((AW - BWID - 14 - draw.textlength(right, font=font(13, "medium")), BWID + 14),
              right, font=font(13, "medium"), fill=FAINT)

    # bottom strip — average latency per frame is always shown (input→output); the result
    # phase also reports this frame's own time on the right.
    draw.rectangle([BWID, A_BOT, AW - BWID, AH - BWID], fill=STRIP)
    avg = av.get("avg")
    draw.text((BWID + 14, A_BOT + 13), "AVG / FRAME", font=font(13, "medium"), fill=FAINT)
    ax = BWID + 14 + draw.textlength("AVG / FRAME ", font=font(13, "medium")) + 6
    draw.text((ax, A_BOT + 10), f"{avg:.1f}s" if avg else "—", font=font(18, "bold"), fill=TEXT)
    if phase == "result":
        this_txt = f"this frame  {av['total']:.1f}s"
        draw.text((AW - BWID - 14 - draw.textlength(this_txt, font=font(14, "medium")), A_BOT + 14),
                  this_txt, font=font(14, "medium"), fill=DIM)

    return np.array(canvas)[:, :, ::-1].copy()  # RGB -> BGR


def _draw_analyzing_badge(img: np.ndarray, tenths: float) -> None:
    """Animated black 'ANALYZING' badge at the frame's bottom-centre. Drawn per-frame
    (blinking dot + ticking timer) so the window obviously stays alive even when several
    consecutive frames look identical and produce the same verdict."""
    bw, bh = 304, 44
    cx = (A_FRAME[0] + A_FRAME[2]) // 2
    bx0, by1 = cx - bw // 2, A_FRAME[3] - 16
    by0 = by1 - bh
    cv2.rectangle(img, (bx0, by0), (bx0 + bw, by1), (8, 8, 8), -1)            # black block
    cv2.rectangle(img, (bx0, by0), (bx0 + bw, by1), ACCENT[::-1], 2, cv2.LINE_AA)
    cy = (by0 + by1) // 2
    if int(tenths * 2) % 2 == 0:                                              # blinking dot
        cv2.circle(img, (bx0 + 26, cy), 8, ACCENT[::-1], -1, cv2.LINE_AA)
    dots = "." * (int(tenths) % 4)                                             # animated ellipsis (1s cycle)
    _cv_text(img, f"ANALYZING{dots}", (bx0 + 46, cy + 7), 0.66, TEXT, 1)
    _cv_text(img, f"{tenths:.1f}s", (bx0 + bw - 16, cy + 7), 0.66, ACCENT, 1, anchor="rt")


# Cache the analysis base; rebuild only when the displayed phase/frame changes (av["id"]).
_an_cache: dict = {"key": None, "img": None}


def render_analysis(av: dict, analyzing_tenths: float, model_name: str, device: str) -> np.ndarray:
    if av["id"] != _an_cache["key"]:
        _an_cache["key"] = av["id"]
        _an_cache["img"] = render_analysis_base(av, model_name, device)
    out = _an_cache["img"].copy()
    if av["phase"] == "analyzing" and analyzing_tenths >= 0:
        _draw_analyzing_badge(out, analyzing_tenths)
    return out


# --------------------------------------------------------------------------
# Terminal results table (rich)
# --------------------------------------------------------------------------

def _cell(label_map: dict, key: str) -> str:
    vl = label_map.get(key, "unknown")
    if vl == "yes":
        return "[bold red]● DETECTED[/]"
    if vl == "no":
        return "[green]○ clear[/]"
    if vl == "unclear":
        return "[yellow]● uncertain[/]"
    return "[dim]—[/]"


def render_table(state: dict, info: dict):
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table

    table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    table.add_column("TIME", style="dim", width=10)
    table.add_column("SMOKING", justify="center")
    table.add_column("FALL", justify="center")
    table.add_column("LATENCY", justify="right", width=9)
    for entry in list(state.get("log", []))[:12]:
        lm = {lab: vl for lab, vl in entry["items"]}
        style = "on #2a1416" if entry.get("alert") else ""
        table.add_row(entry["time"], _cell(lm, "SMOKING"), _cell(lm, "FALL"),
                      f"{entry['total']:.1f}s", style=style)

    i, n = info["playlist_pos"]
    avg = state.get("avg")
    avg_txt = f"avg {avg:.1f}s/frame" if avg else "avg —"
    head = (f"[bold]MOONDREAM[/]  behavior detection    "
            f"[dim]clip {i}/{n} · {info['model_name']} · {info['device'].upper()} · {avg_txt}[/]")
    inprog = state.get("inprogress")
    if inprog is not None:
        t = info["analyzing_tenths"]
        foot = f"[bright_blue]◐ analyzing frame @ {fmt_mmss(inprog['video_time'])} … {t:.1f}s[/]"
    elif info["paused"]:
        foot = "[yellow]❚❚ paused[/]"
    else:
        foot = "[dim]▶ streaming · waiting for next frame[/]"
    return Panel(Group(head, table, foot), border_style="grey37",
                 title="[grey50]live results[/]", title_align="left")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def choose_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


class Analyzer:
    def __init__(self, model, max_width: int = 0):
        self.model = model
        self.max_width = max_width

    @classmethod
    def load(cls, model_id: str, revision: str, max_width: int):
        from transformers import AutoModelForCausalLM

        device, dtype = choose_device()
        print(f"[model] loading {model_id} @ {revision} on {device} …")
        t0 = time.time()
        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        if revision:
            kwargs["revision"] = revision
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()
        analyzer = cls(model, max_width)
        print(f"[model] loaded in {time.time() - t0:.1f}s, warming up …")
        # Cold MPS kernels make the first real inference ~3x slower; warm them up now so
        # the first on-screen detection isn't the slow one.
        t1 = time.time()
        try:
            analyzer.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
        except Exception as exc:
            print(f"[model] warmup skipped: {exc}")
        print(f"[model] ready (warmup {time.time() - t1:.1f}s)")
        return analyzer, device

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self.max_width and image.width > self.max_width:
            scale = self.max_width / image.width
            image = image.resize((self.max_width, int(image.height * scale)), Image.BILINEAR)
        enc = self.model.encode_image(image)
        results: dict[str, dict] = {}
        for key in ORDER:
            t0 = time.time()
            answer = self.model.query(enc, QUERIES[key]["prompt"])["answer"]
            results[key] = {
                "label": classify_answer(answer),
                "answer": answer.strip(),
                "sec": round(time.time() - t0, 2),
            }
        return results


# --------------------------------------------------------------------------
# Inference thread
# --------------------------------------------------------------------------

class InferenceWorker(threading.Thread):
    """Self-paced sequential detector — this thread alone defines the cadence:

        grab the current live frame → ANALYZE (clearing any previous verdict)
        → publish the result → HOLD it for `hold` seconds → grab a fresh frame → repeat.

    The display just mirrors this thread's state, so the analysis window can never jump
    around or show a verdict that does not belong to the frame on screen.
    """

    def __init__(self, analyzer: Analyzer, hold: float):
        super().__init__(daemon=True)
        self.analyzer = analyzer
        self.hold = hold
        self._lock = threading.Lock()
        self._current: tuple | None = None     # (frame, video_time, name) — latest live frame
        self._inprogress: dict | None = None
        self._last: dict | None = None
        self._done = 0
        self._sum = 0.0                         # running totals for the average latency
        self._count = 0
        self._log: deque = deque(maxlen=20)
        self._version = 0
        self._active = threading.Event()
        self._active.set()                      # cleared while the demo is paused
        self._stop = threading.Event()

    def set_current_frame(self, frame: np.ndarray, video_time: float, name: str) -> None:
        with self._lock:
            self._current = (frame, video_time, name)

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._active.clear()
        else:
            self._active.set()

    def view_state(self) -> dict:
        with self._lock:
            avg = self._sum / self._count if self._count else None
            return {
                "inprogress": dict(self._inprogress) if self._inprogress else None,
                "last": dict(self._last) if self._last else None,
                "log": list(self._log),
                "avg": avg,
                "version": self._version,
            }

    def stop(self) -> None:
        self._stop.set()
        self._active.set()

    def run(self) -> None:
        while not self._stop.is_set():
            self._active.wait()                 # idle while paused
            if self._stop.is_set():
                break
            with self._lock:
                cur = self._current
            if cur is None:
                time.sleep(0.02)
                continue
            frame, video_time, name = cur
            frame = frame.copy()
            start = time.time()
            # Begin a fresh detection: clear the previous verdict immediately.
            with self._lock:
                self._inprogress = {"snapshot": frame, "video_time": video_time, "name": name, "start": start}
                self._last = None
                self._version += 1
            try:
                results = self.analyzer.analyze(frame)
            except Exception as exc:  # one bad frame must not crash the UI
                with self._lock:
                    self._inprogress = None
                    self._log.appendleft({"time": fmt_mmss(video_time),
                                          "items": [(QUERIES[k]["label"], "unknown") for k in ORDER],
                                          "total": 0.0, "alert": False, "error": str(exc)})
                    self._version += 1
                continue
            total = round(time.time() - start, 2)
            items, alert = [], False
            for key in ORDER:
                lab = results[key]["label"]
                items.append((QUERIES[key]["label"], lab))
                if lab == "yes":
                    alert = True
            with self._lock:
                self._done += 1
                self._sum += total
                self._count += 1
                self._last = {"id": self._done, "snapshot": frame, "video_time": video_time,
                              "name": name, "results": results, "total": total}
                self._inprogress = None
                self._log.appendleft({"time": fmt_mmss(video_time), "items": items,
                                      "total": total, "alert": alert})
                self._version += 1
            # Hold the result on screen so it is readable, then loop and grab a fresh frame.
            self._stop.wait(self.hold)


# --------------------------------------------------------------------------
# Video sources
# --------------------------------------------------------------------------

def resolve_videos(args) -> list[Path]:
    if args.video:
        vids = [Path(v) for v in args.video]
    elif args.folder:
        collected: list[Path] = []
        for f in args.folder:
            folder = Path(f)
            if not folder.is_dir():
                print(f"[error] not a folder: {folder}")
                continue
            collected.extend(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        # combine folders into one name-sorted playlist (deduped, order-stable)
        seen: set[str] = set()
        vids = []
        for p in sorted(collected, key=lambda p: p.name):
            if p.name not in seen:
                seen.add(p.name)
                vids.append(p)
    else:
        vids = _pick_videos_dialog()
    return [v for v in vids if v.exists()]


def _pick_videos_dialog() -> list[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        picked = filedialog.askopenfilenames(
            title="Select clip(s) to analyse (multi-select, plays in selected order)",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"), ("All files", "*.*")],
        )
        root.destroy()
        return [Path(p) for p in picked]
    except Exception as exc:
        print(f"[error] file dialog unavailable: {exc}")
        print("        use --video or --folder instead.")
        return []


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def run_demo(args) -> int:
    from rich.console import Console
    from rich.live import Live

    videos = resolve_videos(args)
    if not videos:
        print("[error] no playable videos.")
        return 1
    print(f"[demo] playlist ({len(videos)} clip(s)):")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v.name}")

    analyzer, device = Analyzer.load(args.model_id, args.revision, args.max_width)
    model_name = args.model_id.split("/")[-1]
    hold = max(0.0, args.hold)
    worker = InferenceWorker(analyzer, hold)
    worker.start()

    cv2.namedWindow(WINDOW_LIVE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_LIVE, LW, LH)
    cv2.moveWindow(WINDOW_LIVE, 40, 80)
    cv2.namedWindow(WINDOW_ANALYSIS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_ANALYSIS, AW, AH)
    cv2.moveWindow(WINDOW_ANALYSIS, 40 + LW + 20, 80)

    idx = 0
    paused = False
    fail_streak = 0

    console = Console()
    try:
        with Live(console=console, refresh_per_second=8, screen=False) as live:
            while True:
                path = videos[idx]
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    cap.release()
                    fail_streak += 1
                    if fail_streak >= len(videos):
                        print("[error] could not open any clip in the playlist.")
                        break
                    idx = (idx + 1) % len(videos)
                    continue
                fail_streak = 0
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not (isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0):
                    fps = 25.0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if not (isinstance(frame_count, (int, float)) and math.isfinite(frame_count) and frame_count > 0):
                    frame_count = 0
                duration = frame_count / fps if frame_count else 0.0
                frame_period = 1.0 / (fps * max(args.speed, 0.05))
                last_frame = None
                video_time = 0.0
                next_deadline = time.monotonic() + frame_period

                while True:
                    if not paused:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        last_frame = frame
                        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        # Feed the worker the freshest frame; it grabs one only when it is
                        # ready to start the next detection (it paces itself).
                        worker.set_current_frame(frame, video_time, path.name)

                    state = worker.view_state()
                    inprog = state.get("inprogress")
                    latest = state.get("last")
                    now = time.monotonic()
                    analyzing_tenths = round(now - inprog["start"], 1) if inprog else -1.0
                    # The worker defines the cadence (analyze → hold → next); we just mirror
                    # its state, so the verdict always belongs to the frame on screen.
                    if inprog is not None:
                        av = {"phase": "analyzing", "frame": inprog["snapshot"], "results": None,
                              "video_time": inprog["video_time"], "total": None,
                              "avg": state["avg"], "id": ("A", inprog["start"])}
                    elif latest is not None:
                        av = {"phase": "result", "frame": latest["snapshot"], "results": latest["results"],
                              "video_time": latest["video_time"], "total": latest["total"],
                              "avg": state["avg"], "id": ("R", latest["id"])}
                    else:
                        av = {"phase": "standby", "frame": None, "results": None,
                              "video_time": 0.0, "total": None, "avg": None, "id": ("S",)}
                    info = {
                        "video_name": path.name,
                        "video_time": video_time,
                        "video_dur": duration,
                        "playlist_pos": (idx + 1, len(videos)),
                        "model_name": model_name,
                        "device": device,
                        "paused": paused,
                        "analyzing_tenths": analyzing_tenths,
                    }
                    cv2.imshow(WINDOW_LIVE, render_live(last_frame, info))
                    cv2.imshow(WINDOW_ANALYSIS, render_analysis(av, analyzing_tenths, model_name, device))
                    live.update(render_table(state, info))

                    if paused:
                        wait_ms = 60
                    else:
                        remaining = next_deadline - time.monotonic()
                        wait_ms = max(1, int(remaining * 1000))
                        next_deadline += frame_period
                        if time.monotonic() - next_deadline > frame_period:  # fell badly behind
                            next_deadline = time.monotonic() + frame_period
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (ord("q"), 27):
                        raise KeyboardInterrupt
                    if key == ord(" "):
                        paused = not paused
                        worker.set_paused(paused)
                        next_deadline = time.monotonic() + frame_period
                    elif key == ord("n"):
                        break
                cap.release()
                idx = (idx + 1) % len(videos)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        cv2.destroyAllWindows()
    return 0


def run_selftest(args) -> int:
    videos = resolve_videos(args)
    if videos:
        cap = cv2.VideoCapture(str(videos[0]))
        cap.set(cv2.CAP_PROP_POS_MSEC, 3000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = np.full((816, 1104, 3), 40, dtype=np.uint8)
    else:
        frame = np.full((816, 1104, 3), 40, dtype=np.uint8)

    info = {
        "video_name": "grok-video-fall.mp4", "video_time": 4.5, "video_dur": 6.0,
        "playlist_pos": (1, 3), "model_name": "moondream2", "device": "mps",
        "paused": False, "analyzing_tenths": 2.3,
    }
    here = Path(__file__).resolve().parent
    live_png = here / "_selftest_live.png"
    cv2.imwrite(str(live_png), render_live(frame, info))

    # analysing phase: frame being scanned, NO verdict
    av_analyzing = {"phase": "analyzing", "frame": frame, "results": None,
                    "video_time": 5.0, "total": None, "avg": 3.0, "id": ("A", 1)}
    analyzing_png = here / "_selftest_analyzing.png"
    cv2.imwrite(str(analyzing_png), render_analysis(av_analyzing, 2.3, "moondream2", "mps"))

    # result phase: same frame WITH verdict, held
    av_result = {"phase": "result", "frame": frame,
                 "results": {"smoking": {"label": "yes"}, "fall": {"label": "no"}},
                 "video_time": 4.5, "total": 3.1, "avg": 3.0, "id": ("R", 1)}
    result_png = here / "_selftest_result.png"
    cv2.imwrite(str(result_png), render_analysis(av_result, -1.0, "moondream2", "mps"))
    print(f"[selftest] live preview:      {live_png}")
    print(f"[selftest] analyzing preview: {analyzing_png}")
    print(f"[selftest] result preview:    {result_png}")

    # one-shot terminal table sample
    from rich.console import Console
    state = {
        "inprogress": {"video_time": 5.0, "start": 0},
        "last": None,
        "avg": 3.05,
        "log": [
            {"time": "00:04.5", "items": [("SMOKING", "yes"), ("FALL", "no")], "total": 3.1, "alert": True},
            {"time": "00:03.5", "items": [("SMOKING", "yes"), ("FALL", "no")], "total": 2.9, "alert": True},
            {"time": "00:02.5", "items": [("SMOKING", "no"), ("FALL", "no")], "total": 3.0, "alert": False},
            {"time": "00:01.5", "items": [("SMOKING", "no"), ("FALL", "unclear")], "total": 3.2, "alert": False},
        ],
        "version": 1,
    }
    Console().print(render_table(state, info))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Moondream behavior-detection demo (2 windows + terminal log).")
    p.add_argument("--video", nargs="+", help="one or more video paths, played in order.")
    p.add_argument("--folder", nargs="+",
                   help="one or more folders of videos, combined into one name-sorted playlist.")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (0.5 slows short clips).")
    p.add_argument("--max-width", type=int, default=0,
                   help="downscale width sent to the model. 512 ~= 3x faster on MPS. 0 = original.")
    p.add_argument("--hold", type=float, default=0.5,
                   help="seconds to hold each completed result before grabbing a fresh frame.")
    p.add_argument("--model-id", default="vikhyatk/moondream2")
    p.add_argument("--revision", default="2025-01-09")
    p.add_argument("--selftest", action="store_true", help="render layout preview PNGs + a sample table, then exit.")
    args = p.parse_args()

    if args.selftest:
        return run_selftest(args)
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
