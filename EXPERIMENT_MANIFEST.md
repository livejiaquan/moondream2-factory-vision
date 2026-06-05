# Experiment Manifest

This project is organized around three independent experiments. The old timestamp run IDs are preserved only in local outputs; the public code uses stable experiment names.

| Public ID | Folder | What It Tests | Historical Runs |
| --- | --- | --- | --- |
| `EXP01_BASIC_FACTORY_FEASIBILITY` | `experiments/01_basic_factory_feasibility/` | Initial Moondream2 feasibility on factory/tanker CCTV: caption, wheel chock, worker/PPE, helmet missing, abnormal behavior. | 2026-04-02 first run; 2026-04-04 expanded query round. |
| `EXP02_YOLO_GAP_NEGATIVE_BASELINE` | `experiments/02_yolo_gap_negative_baseline/` | Normal-operation negative baseline for YOLO-gap scenarios: false positives, worker gate, PPE visibility, occlusion, attribution. | 2026-04-15 baseline and follow-up occlusion/attribution runs. |
| `EXP03_BEHAVIOR_FALL_SMOKING` | `experiments/03_behavior_fall_smoking/` | Positive-sample behavior test for fall and smoking clips, including direct/strict prompts, combined query, bbox check, and live demo. | 2026-05-20 to 2026-05-26 behavior-report run family. |

Shareable coworker-facing artifacts are separated from local-only reports:

| Public Artifact | Folder | Source Meaning |
| --- | --- | --- |
| `exp03_behavior_fall_smoking_20260526.pptx` | `shareables/presentations/` | Final behavior experiment slide deck. |
| `exp03_behavior_fall_smoking_20260526_report.pdf` | `shareables/reports/` | Final behavior experiment PDF report. |

Recovered but local-only artifact:

| Local Artifact | Local Folder | Why Not Public |
| --- | --- | --- |
| `moondream_factory_overview_2026_DO_NOT_PUSH.pptx` | `_local_sensitive/04_reports_and_decks/reports/private_not_pushed/` | Remote-internship Moondream report deck contains company/person/CCTV details and is excluded because this GitHub repo is public. |

Sensitive/local material is separated into `_local_sensitive/`:

| Local ID | Local Folder | Contents |
| --- | --- | --- |
| `S01_FACTORY_CCTV` | `_local_sensitive/01_factory_cctv_data/` | Factory CCTV videos, company still images, HPC-returned images. |
| `S02_RUN_OUTPUTS` | `_local_sensitive/02_run_outputs/` | Historical JSONL/CSV/HTML outputs and annotated frames. |
| `S03_BEHAVIOR_VIDEOS` | `_local_sensitive/03_behavior_videos/` | Fall/smoking demo and behavior clips. |
| `S04_REPORTS_DECKS` | `_local_sensitive/04_reports_and_decks/` | Internal reports, PPTX/PDF decks, archived intermediate report versions. |
| `S99_LEGACY_MISC` | `_local_sensitive/99_legacy_misc/` | Old examples, docs, tests, advanced/HPC/demo leftovers not part of the core public project. |
