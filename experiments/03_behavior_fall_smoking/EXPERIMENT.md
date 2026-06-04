# EXP03_BEHAVIOR_FALL_SMOKING

Purpose: positive-sample behavior test for fall and smoking clips.

Core code:

| File | Role |
| --- | --- |
| `scripts/extract_behavior_frames.py` | Extract labeled frames from short behavior clips. |
| `scripts/run_behavior_experiment.py` | Run direct/strict/open-evidence prompts. |
| `scripts/run_behavior_combined_query_experiment.py` | Test one combined fall/smoking query. |
| `scripts/run_prompt_variant_experiment.py` | Compare prompt variants A-D. |
| `scripts/run_behavior_bbox_experiment.py` | Test detect-based localization for person/cigarette. |
| `scripts/generate_behavior_report.py` | Generate HTML behavior report from run output. |
| `scripts/live_behavior_demo.py` | Stakeholder demo UI for fall/smoking clips. |

Local input: `data/behavior_videos/`

Local output: `experiments/03_behavior_fall_smoking/outputs/`
