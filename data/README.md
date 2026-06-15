# Data Placement

此目錄只保留資料放置說明，不提交原始圖片、影片或實驗輸出。

## Expected Local Folders

| Path | Used by | Notes |
| --- | --- | --- |
| `data/raw_videos/` | `experiments/01_basic_factory_feasibility`, `experiments/02_yolo_gap_negative_baseline` | 工廠 CCTV 或正常作業影片。不要提交。 |
| `data/behavior_videos/` | `experiments/03_behavior_fall_smoking` | 跌倒、抽菸短影片。建議使用 `fall_01.mp4`、`smoking_01.mp4` 這類清楚檔名。不要提交。 |

## Output Locations

| Path | Meaning |
| --- | --- |
| `experiments/01_basic_factory_feasibility/outputs/` | 第一輪可行性實驗輸出。 |
| `experiments/02_yolo_gap_negative_baseline/outputs/` | YOLO-gap 負樣本 baseline 與 follow-up 輸出。 |
| `experiments/03_behavior_fall_smoking/outputs/` | 跌倒、抽菸行為實驗輸出。 |

這些 output folder 也由 Git ignore 排除。

## Local Archive

本機若需要保存原始資料、歷史輸出或內部文件，使用 `_local_sensitive/`。該資料夾不應提交到 GitHub。
