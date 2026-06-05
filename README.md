# Moondream Factory Vision Experiments

Moondream2 feasibility experiments for factory safety monitoring. The project is organized as code-first experiment folders so coworkers can inspect each test independently.

## What Is in This Repo

- Core experiment code for three independent Moondream test tracks.
- Experiment notes in each `experiments/*/EXPERIMENT.md`.
- Public-safe coworker-facing decks/reports under `shareables/`.
- Placeholders only for local data; raw videos, stills, model outputs, and private working notes are excluded.

## Experiment Folders

| ID | Folder | Focus |
| --- | --- | --- |
| `EXP01_BASIC_FACTORY_FEASIBILITY` | `experiments/01_basic_factory_feasibility/` | Initial factory/tanker CCTV feasibility: caption, wheel chock, worker/PPE, helmet, abnormal behavior. |
| `EXP02_YOLO_GAP_NEGATIVE_BASELINE` | `experiments/02_yolo_gap_negative_baseline/` | Normal-operation negative baseline for YOLO-gap false positives, occlusion, and attribution. |
| `EXP03_BEHAVIOR_FALL_SMOKING` | `experiments/03_behavior_fall_smoking/` | Positive-sample fall/smoking behavior tests and demo. |

See `EXPERIMENT_MANIFEST.md` for the experiment mapping, shareable artifact inventory, and sensitive-data inventory.

## Shareable Artifacts

| Folder | Contents |
| --- | --- |
| `shareables/presentations/` | Public-safe PPTX decks, renamed with stable readable filenames. |
| `shareables/reports/` | Final report PDFs selected for coworker review. |

These files are selected for coworker review. Do not add raw run folders, private AI conversations, generated workspaces, or unreviewed local reports here.

## Sensitive Data Policy

Company CCTV clips, stills, historical outputs, private AI notes, generated workspaces, and unreviewed local reports are not part of the public project tree. On this machine they are kept under `_local_sensitive/`, which is ignored by Git.

Local input folders expected by scripts:

- `data/raw_videos/` for factory CCTV experiments.
- `data/behavior_videos/` for fall/smoking behavior clips.

These folders are ignored and should stay local.
