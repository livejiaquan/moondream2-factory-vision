# EXP03_BEHAVIOR_FALL_SMOKING

## Purpose

This experiment is the positive-sample behavior test. It is separate from the factory CCTV baseline because the input clips intentionally contain target behaviors: falling and smoking.

The goal was to answer:

- Can Moondream2 detect a fall after the person is clearly down?
- Can it detect visible smoking or a cigarette?
- Is one combined multi-event prompt reliable enough to reduce runtime?
- Can `detect()` localization help as supporting evidence?

## Data Scope

Input data is intentionally not committed.

| Local Code | Expected Path | Description |
| --- | --- | --- |
| `S03_BEHAVIOR_VIDEOS` | `data/behavior_videos/` | Short fall/smoking clips with normalized names such as `fall_01.mp4` and `smoking_01.mp4`. |

Historical run family:

- 2026-05-20 to 2026-05-26: 7 short clips, 91 sampled frames, final report/deck generated from this run family.

## Method

The experiment first extracts labeled frames from each clip, then runs frame-level prompts.

Prompt families:

- Method A: direct yes/no question for a single event.
- Method B: strict evidence question; only answer yes when visual evidence is clear.
- Method C: generic abnormal-behavior question, used as a weak broad-signal reference.
- Method D: open scene description for posture and hand/mouth evidence.
- Method E: combined fall and smoking query in one prompt to test runtime savings.

Additional localization:

- `detect person` on fall clips to verify whether the person can be localized.
- `detect cigarette` on smoking clips to test whether the small object can be localized.

## Code

| File | Role |
| --- | --- |
| `scripts/extract_behavior_frames.py` | Extract labeled frames from behavior clips. |
| `scripts/run_behavior_experiment.py` | Run direct, strict, and open-evidence prompts. |
| `scripts/run_behavior_combined_query_experiment.py` | Test one combined fall/smoking query. |
| `scripts/run_prompt_variant_experiment.py` | Compare prompt variants A-D. |
| `scripts/run_behavior_bbox_experiment.py` | Test detect-based localization for person/cigarette. |
| `scripts/generate_behavior_report.py` | Generate HTML behavior report from run output. |
| `scripts/live_behavior_demo.py` | Local stakeholder demo UI for fall/smoking clips. |

## Config

| File | Role |
| --- | --- |
| `configs/behavior_fall_smoking.json` | Frame extraction settings, query definitions, and open-evidence keywords. |

## How to Run

```bash
source .venv/bin/activate
python experiments/03_behavior_fall_smoking/scripts/extract_behavior_frames.py
python experiments/03_behavior_fall_smoking/scripts/run_behavior_experiment.py --run-dir experiments/03_behavior_fall_smoking/outputs/<run_id>
python experiments/03_behavior_fall_smoking/scripts/run_behavior_combined_query_experiment.py --run-dir experiments/03_behavior_fall_smoking/outputs/<run_id>
python experiments/03_behavior_fall_smoking/scripts/run_behavior_bbox_experiment.py --run-dir experiments/03_behavior_fall_smoking/outputs/<run_id>
python experiments/03_behavior_fall_smoking/scripts/generate_behavior_report.py --run-dir experiments/03_behavior_fall_smoking/outputs/<run_id>
```

Default input:

```text
data/behavior_videos/
```

Default output:

```text
experiments/03_behavior_fall_smoking/outputs/<run_id>/
```

## Output Files

| File | Meaning |
| --- | --- |
| `manifest.jsonl` | Extracted frame inventory with expected event labels. |
| `results.jsonl` | Per-frame caption and query outputs. |
| `summary.csv` | Flattened frame-level result table. |
| `combined_query_results.jsonl` | Method E outputs when generated. |
| `bbox_results.jsonl` | Detect/localization outputs when generated. |
| `behavior_report.html` | Human-readable experiment report. |

## Key Findings

- Fall detection became stable after the fall was visually formed; early transition frames were not a reliable trigger.
- Smoking detection worked better with direct single-event prompts than with combined prompts.
- The combined prompt reduced query count but caused cross-event contamination and poor smoking recall.
- `detect person` was useful supporting evidence for fall clips.
- `detect cigarette` was not stable enough to be used as the primary smoking signal.

## Interpretation Notes

This experiment uses positive samples, so it does not measure field precision by itself. The result supports PoC-level behavior checks, but deployment needs negative samples such as standing, sitting, crouching, holding tools, and normal hand-to-face motion.
