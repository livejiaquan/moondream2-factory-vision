# Data Placement

This directory is intentionally kept free of company images and videos in the GitHub-ready tree.

Use these local-only folders when running the experiments:

| Path | Used by | Notes |
| --- | --- | --- |
| `data/raw_videos/` | `experiments/01_basic_factory_feasibility`, `experiments/02_yolo_gap_negative_baseline` | Factory CCTV clips. Do not commit. |
| `data/behavior_videos/` | `experiments/03_behavior_fall_smoking` | Short fall/smoking clips. Use normalized names like `fall_01.mp4`, `smoking_01.mp4`. Do not commit. |

On this machine, original local files were moved to `_local_sensitive/` and are ignored by Git.
