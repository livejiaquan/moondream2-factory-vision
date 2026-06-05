# EXP01_BASIC_FACTORY_FEASIBILITY

## Purpose

This was the first factory CCTV feasibility test for Moondream2. The goal was not to build an alarm system yet; it was to check whether a lightweight VLM can provide useful semantic evidence on tanker-operation frames where a normal object detector is not enough.

The experiment focused on these questions:

- Can Moondream describe the tanker operation scene in a stable way?
- Can it see wheel chocks near vehicle wheels?
- Can it identify visible workers and possible PPE issues such as missing helmets or face masks?
- Does it over-trigger abnormal or unsafe behavior on normal operation footage?

## Data Scope

Input data is intentionally not committed. The public code expects the local folder below:

| Local Code | Expected Path | Description |
| --- | --- | --- |
| `S01_FACTORY_CCTV` | `data/raw_videos/` | Factory/tanker CCTV clips used for feasibility testing. |

Historical runs:

- 2026-04-02: initial sparse feasibility pass, about one frame every 45 seconds.
- 2026-04-04: expanded query pass, about one frame every 30 seconds.

The second date above is still treated as experiment 1 because it used the same feasibility framing and the same factory CCTV source family.

## Method

The runner samples video frames, encodes each frame once, then runs:

- caption generation for scene-level evidence;
- yes/no VQA prompts for worker, wheel chock, helmet, face mask, and abnormal behavior checks;
- `detect()` calls for person, helmet, and wheel chock boxes;
- optional annotated images with query labels and detection boxes.

The output is designed for manual review and lightweight statistics, not for direct production alerting.

## Code

| File | Role |
| --- | --- |
| `scripts/moondream_cli.py` | Single-image CLI for caption, query, detect, and point tests. |
| `scripts/run_feasibility.py` | Video frame sampler plus safety-query runner for factory CCTV clips. |

## How to Run

```bash
source .venv/bin/activate
python experiments/01_basic_factory_feasibility/scripts/run_feasibility.py
python experiments/01_basic_factory_feasibility/scripts/run_feasibility.py --every 10 --limit 20
```

Default input:

```text
data/raw_videos/
```

Default output:

```text
experiments/01_basic_factory_feasibility/outputs/feasibility/<run_id>/
```

## Output Files

| File | Meaning |
| --- | --- |
| `results.jsonl` | Full per-frame caption, query answers, labels, and detections. |
| `summary.csv` | Flattened per-frame table for quick review. |
| `annotated/` | Optional annotated frames with boxes and query labels. |
| `run_meta.json` | Model, revision, device, input path, and sampling settings. |

## Key Findings

- Wheel chock recognition was the most stable signal in this early test family.
- Helmet-missing prompts showed some useful signal but were not reliable enough for direct alerts.
- Caption output was useful as text evidence for what the model believed was in the frame.
- Abnormal behavior stayed conservative on the available normal-operation frames.

## Interpretation Notes

This experiment is a feasibility screen. A production system must add a rule layer, worker visibility gates, repeated-frame confirmation, and human review thresholds. Do not treat a single Moondream yes/no answer as a final safety alarm.
