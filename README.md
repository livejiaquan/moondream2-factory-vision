# Moondream Factory Vision Experiments

Moondream2 feasibility experiments for factory safety monitoring. The project is organized as code-first experiment folders so coworkers can inspect each test independently.

## Experiment Folders

| ID | Folder | Focus |
| --- | --- | --- |
| `EXP01_BASIC_FACTORY_FEASIBILITY` | `experiments/01_basic_factory_feasibility/` | Initial factory/tanker CCTV feasibility: caption, wheel chock, worker/PPE, helmet, abnormal behavior. |
| `EXP02_YOLO_GAP_NEGATIVE_BASELINE` | `experiments/02_yolo_gap_negative_baseline/` | Normal-operation negative baseline for YOLO-gap false positives, occlusion, and attribution. |
| `EXP03_BEHAVIOR_FALL_SMOKING` | `experiments/03_behavior_fall_smoking/` | Positive-sample fall/smoking behavior tests and demo. |

See `EXPERIMENT_MANIFEST.md` for the experiment mapping and sensitive-data inventory.

## Sensitive Data Policy

Company CCTV clips, stills, historical outputs, internal reports, and decks are not part of the public project tree. On this machine they are kept under `_local_sensitive/`, which is ignored by Git.

Local input folders expected by scripts:

- `data/raw_videos/` for factory CCTV experiments.
- `data/behavior_videos/` for fall/smoking behavior clips.

These folders are ignored and should stay local.
