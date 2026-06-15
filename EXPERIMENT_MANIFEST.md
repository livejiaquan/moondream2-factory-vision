# Experiment Manifest

本 repo 以穩定的 experiment ID 管理實驗，避免用歷史 timestamp run ID 當主要入口。timestamp run ID 只保留在本機輸出資料夾中。

## Experiments

| Public ID | Folder | What It Tests | Historical Runs |
| --- | --- | --- | --- |
| `EXP01_BASIC_FACTORY_FEASIBILITY` | `experiments/01_basic_factory_feasibility/` | 工廠/槽車 CCTV 初步可行性：caption、輪檔、工人/PPE、安全帽缺失、異常行為。 | 2026-04-02 first run; 2026-04-04 expanded query round. |
| `EXP02_YOLO_GAP_NEGATIVE_BASELINE` | `experiments/02_yolo_gap_negative_baseline/` | 正常作業負樣本 baseline：false positive、worker gate、PPE visibility、occlusion、attribution。 | 2026-04-15 baseline and follow-up occlusion/attribution runs. |
| `EXP03_BEHAVIOR_FALL_SMOKING` | `experiments/03_behavior_fall_smoking/` | 跌倒與抽菸正樣本：direct/strict prompts、combined query、bbox check、live demo。 | 2026-05-20 to 2026-05-26 behavior-report run family. |

## Shareable Artifacts

| Artifact | Folder | Meaning |
| --- | --- | --- |
| `exp03_behavior_fall_smoking_20260526.pptx` | `shareables/presentations/` | 跌倒/抽菸行為實驗簡報。 |
| `exp03_behavior_fall_smoking_20260526_report.pdf` | `shareables/reports/` | 跌倒/抽菸行為實驗 PDF 報告。 |

## Local-Only Material

下列資料保存在本機，並由 `.gitignore` 排除：

| Local ID | Local Folder | Contents |
| --- | --- | --- |
| `S01_FACTORY_CCTV` | `_local_sensitive/01_factory_cctv_data/` | 工廠 CCTV 影片、現場圖片、HPC 回傳圖片。 |
| `S02_RUN_OUTPUTS` | `_local_sensitive/02_run_outputs/` | 歷史 JSONL、CSV、HTML、標註圖和 run output。 |
| `S03_BEHAVIOR_VIDEOS` | `_local_sensitive/03_behavior_videos/` | 跌倒、抽菸 demo 與行為影片。 |
| `S04_REPORTS_DECKS` | `_local_sensitive/04_reports_and_decks/` | 內部報告、簡報、PDF 和中間版本。 |
| `S99_LEGACY_MISC` | `_local_sensitive/99_legacy_misc/` | 舊 workspace、範例、demo、HPC 相關暫存資料。 |

`_local_sensitive/04_reports_and_decks/reports/private_not_pushed/` 內有保留但不提交的報告或簡報，原因通常是含有公司、人物、CCTV 或未篩選內容。
