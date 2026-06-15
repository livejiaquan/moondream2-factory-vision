# Moondream Factory Vision Experiments

使用 Moondream2 評估工廠安全監控影像中的語意判斷能力。這個 repo 的重點不是訓練模型，而是把既有影片或圖片抽成 frame，使用 Moondream2 做 caption、VQA query、detect，再把結果整理成可檢查的 JSONL、CSV、標註圖與報告。

## 目標

一般 YOLO 類物件偵測適合找明確物件，例如人、車、輪胎或安全帽；但有些安全判斷需要更多語意脈絡，例如：

- 輪檔是否在車輪附近；
- 畫面中是否有可見工人；
- 工人是否疑似未戴安全帽或面罩；
- 畫面是否出現跌倒、抽菸等行為；
- 模型對現場狀態的文字描述是否能作為人工複核依據。

這些實驗用 Moondream2 產生「影像語意證據」。正式系統仍應由 rule engine 做告警決策，不應直接把單次 VLM 問答結果當成安全告警。

## 目前結論

| 能力 | 狀態 | 解讀 |
| --- | --- | --- |
| 輪檔辨識 | 穩定 | 適合做 PoC，仍需用規則確認位置與重複幀一致性。 |
| 安全帽缺失 | 有訊號 | 需要先確認 worker visible，再用多幀與 detect/caption 交叉驗證。 |
| 場景描述 | 穩定 | 適合保留成文字證據，方便人工複核。 |
| 抽菸辨識 | 不穩 | 容易受小物件、手部姿勢與管線干擾，不建議直接告警。 |
| 異常行為 | 偏保守 | 需要更多正樣本與正常姿勢負樣本。 |
| 跌倒辨識 | 正樣本可行 | 跌倒狀態形成後較穩，早期轉換幀不適合單幀判斷。 |

## 環境

建議使用 Python 3.13。若本機尚未建立虛擬環境：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

若 repo 內已經有 `.venv/`，直接啟用即可：

```bash
source .venv/bin/activate
python --version
```

主要套件：

| 套件 | 用途 |
| --- | --- |
| `torch==2.10.0` | 推理後端，Apple Silicon 會優先使用 MPS。 |
| `torchvision==0.25.0` | PyTorch 影像工具與模型相關依賴。 |
| `transformers==4.46.3` | 載入 `vikhyatk/moondream2`。 |
| `opencv-python-headless==4.13.0.92` | 讀取影片與抽幀。 |
| `Pillow==12.1.1` | 圖片讀取與標註。 |
| `numpy==2.4.3` | 影像矩陣處理。 |
| `einops==0.8.2` | Moondream remote code 可能使用的 tensor 操作依賴。 |
| `accelerate==1.13.0` | Hugging Face 模型載入相關依賴。 |

模型設定：

```text
model:    vikhyatk/moondream2
revision: 2025-01-09
device:   mps / cuda / cpu，自動選擇
dtype:    float16 on mps/cuda, float32 on cpu
```

如果本機沒有模型快取，第一次執行會從 Hugging Face 下載模型。需要 token 時可參考 `.env.example` 放置 `HF_TOKEN`，但不要提交個人 token。

## Repo 結構

```text
.
├── README.md
├── EXPERIMENT_MANIFEST.md
├── requirements.txt
├── data/
│   └── README.md
├── experiments/
│   ├── 01_basic_factory_feasibility/
│   ├── 02_yolo_gap_negative_baseline/
│   └── 03_behavior_fall_smoking/
└── shareables/
    ├── presentations/
    └── reports/
```

重要原始資料、歷史輸出和內部報告不放在 GitHub tree：

| 路徑 | 狀態 | 用途 |
| --- | --- | --- |
| `data/raw_videos/` | Git ignore | 工廠 CCTV 影片。 |
| `data/behavior_videos/` | Git ignore | 跌倒、抽菸正樣本影片。 |
| `experiments/*/outputs/` | Git ignore | 每次實驗輸出。 |
| `_local_sensitive/` | Git ignore | 本機保存的原始資料、歷史 run、內部文件。 |

## 推理流程

每個實驗的核心流程相同：

```text
Video/Image input
  -> frame sampling
  -> model.encode_image(image)
  -> caption / query / detect
  -> yes-no label parsing
  -> JSONL + CSV + annotated frames
  -> manual review / report generation
```

主要 Moondream API 用法：

| API | 在本 repo 的用途 |
| --- | --- |
| `model.caption(enc, length="normal")` | 產生場景描述，作為文字證據。 |
| `model.query(enc, prompt)` | 問 yes/no 或開放式問題，例如是否有輪檔、是否有人跌倒。 |
| `model.detect(enc, target)` | 找指定目標的 bbox，例如 `person`、`helmet`、`wheel chock`、`cigarette`。 |
| `model.point(enc, target)` | 單張圖片定位測試，主要用於互動式檢查。 |

`encode_image()` 在同一張 frame 上只做一次，後續 caption/query/detect 共用同一個 encoding，避免重複影像編碼。

## 實驗一覽

| ID | 目錄 | 目的 |
| --- | --- | --- |
| `EXP01_BASIC_FACTORY_FEASIBILITY` | `experiments/01_basic_factory_feasibility/` | 第一輪工廠 CCTV 可行性測試：caption、輪檔、工人、PPE、安全帽、異常行為。 |
| `EXP02_YOLO_GAP_NEGATIVE_BASELINE` | `experiments/02_yolo_gap_negative_baseline/` | 正常作業負樣本基線：觀察 prompt false positive、遮擋與歸因問題。 |
| `EXP03_BEHAVIOR_FALL_SMOKING` | `experiments/03_behavior_fall_smoking/` | 跌倒與抽菸正樣本測試：比較 prompt 設計、combined query、detect 輔助與 demo。 |

各實驗目錄內的 `EXPERIMENT.md` 會說明該實驗的資料範圍、方法、腳本、輸出與結果解讀。

## 腳本索引

### `experiments/01_basic_factory_feasibility`

| 腳本 | 用途 | 常用指令 |
| --- | --- | --- |
| `scripts/moondream_cli.py` | 單張圖片互動測試，支援 caption/query/detect/point/chat。 | `python experiments/01_basic_factory_feasibility/scripts/moondream_cli.py caption -i image.jpg` |
| `scripts/run_feasibility.py` | 對工廠影片抽幀後跑安全檢查，輸出 JSONL、CSV、標註圖。 | `python experiments/01_basic_factory_feasibility/scripts/run_feasibility.py --every 30` |

範例：

```bash
source .venv/bin/activate
python experiments/01_basic_factory_feasibility/scripts/run_feasibility.py
python experiments/01_basic_factory_feasibility/scripts/run_feasibility.py --every 10 --limit 20
```

### `experiments/02_yolo_gap_negative_baseline`

| 腳本 | 用途 |
| --- | --- |
| `scripts/run_baseline.py` | 跑負樣本基線，統計 expected=no 但模型回答 yes 的 false positive。 |
| `scripts/analyze_baseline.py` | 讀取 baseline run，輸出聚合統計報告。 |
| `scripts/generate_detail_report.py` | 產生 frame-level HTML 詳細報告。 |
| `scripts/run_occlusion_experiment.py` | 針對安全帽、面罩遮擋做 follow-up。 |
| `scripts/run_attribution_experiment.py` | 檢查 false positive 可能來自哪種視覺線索或 prompt 假設。 |
| `scripts/run_occlusion_phase2.py` | 用更嚴格 prompt 重跑安全帽可見性。 |
| `scripts/run_occlusion_mask.py` | 用更嚴格 prompt 重跑面罩可見性。 |
| `scripts/run_occlusion_prompt_variants.py` | 比較安全帽遮擋 prompt 變體。 |
| `scripts/run_mask_prompt_variants.py` | 比較面罩 prompt 變體。 |

範例：

```bash
source .venv/bin/activate
cd experiments/02_yolo_gap_negative_baseline
python scripts/run_baseline.py --every 10 --annotate
python scripts/analyze_baseline.py --run latest
python scripts/generate_detail_report.py --run latest
```

可選 follow-up：

```bash
python scripts/run_occlusion_experiment.py --baseline-run <RUN_ID>
python scripts/run_attribution_experiment.py --baseline-run <RUN_ID>
```

### `experiments/03_behavior_fall_smoking`

| 腳本 | 用途 |
| --- | --- |
| `scripts/extract_behavior_frames.py` | 從短影片抽取標記 frame，產生 `manifest.jsonl`。 |
| `scripts/run_behavior_experiment.py` | 跑跌倒、抽菸 direct/strict/open evidence prompts。 |
| `scripts/run_behavior_combined_query_experiment.py` | 用單一 combined query 同時問跌倒與抽菸，評估效率與互相干擾。 |
| `scripts/run_prompt_variant_experiment.py` | 比較不同 prompt family 的結果。 |
| `scripts/run_behavior_bbox_experiment.py` | 用 `detect person`、`detect cigarette` 測試定位輔助效果。 |
| `scripts/generate_behavior_report.py` | 產生行為實驗 HTML 報告。 |
| `scripts/live_behavior_demo.py` | 本機即時 demo UI，播放影片並在背景 thread 跑推理。 |

範例：

```bash
source .venv/bin/activate
python experiments/03_behavior_fall_smoking/scripts/extract_behavior_frames.py
python experiments/03_behavior_fall_smoking/scripts/run_behavior_experiment.py \
  --run-dir experiments/03_behavior_fall_smoking/outputs/<RUN_ID>
python experiments/03_behavior_fall_smoking/scripts/run_behavior_combined_query_experiment.py \
  --run-dir experiments/03_behavior_fall_smoking/outputs/<RUN_ID>
python experiments/03_behavior_fall_smoking/scripts/run_behavior_bbox_experiment.py \
  --run-dir experiments/03_behavior_fall_smoking/outputs/<RUN_ID>
python experiments/03_behavior_fall_smoking/scripts/generate_behavior_report.py \
  --run-dir experiments/03_behavior_fall_smoking/outputs/<RUN_ID>
```

Demo：

```bash
pip install -r experiments/03_behavior_fall_smoking/requirements-demo.txt
python experiments/03_behavior_fall_smoking/scripts/live_behavior_demo.py \
  --folder data/behavior_videos \
  --max-width 512
```

`live_behavior_demo.py` 會使用 OpenCV GUI 視窗與 `rich` 終端表格；若只跑批次推理，不需要安裝 demo 額外套件。

## Config 檔

| Config | 用途 |
| --- | --- |
| `experiments/02_yolo_gap_negative_baseline/configs/gap_baseline.json` | 負樣本 baseline 的 query/detect 定義。 |
| `experiments/02_yolo_gap_negative_baseline/configs/gap_baseline_v2.json` | 移除不符合現場情境項目後的 baseline 設定。 |
| `experiments/02_yolo_gap_negative_baseline/configs/occlusion_queries.json` | 遮擋追蹤問題。 |
| `experiments/02_yolo_gap_negative_baseline/configs/occlusion_variants.json` | 遮擋 prompt 變體。 |
| `experiments/02_yolo_gap_negative_baseline/configs/attribution_queries.json` | false positive 歸因問題。 |
| `experiments/03_behavior_fall_smoking/configs/behavior_fall_smoking.json` | 行為實驗的影片抽幀、query 與 evidence keyword 設定。 |

## 輸出格式

不同實驗的輸出欄位不完全相同，但共同概念如下：

| 檔案 | 用途 |
| --- | --- |
| `results.jsonl` | 每一行是一個 frame 的完整推理結果，包含 caption、queries、detections。 |
| `summary.csv` | 扁平化表格，方便用 spreadsheet 或 pandas 檢查。 |
| `run_meta.json` | run id、模型、revision、device、抽幀設定、資料來源。 |
| `annotated/` | 加上 bbox 與 query label 的標註圖。 |
| `manifest.jsonl` | 行為實驗使用，記錄抽出的 frame 與 expected event。 |
| `*.html` / `*.md` | 分析腳本產生的人工檢查報告。 |

## 判讀規則

建議把模型輸出分成三層使用：

1. **Evidence**：caption、query answer、detect bbox，保留完整原始輸出。
2. **Signal**：把 answer 開頭解析成 `yes`、`no`、`unclear`、`unknown`，做統計或初步篩選。
3. **Decision**：由 rule engine 決定是否形成告警，例如 worker gate、多幀確認、bbox count、caption keyword 與人工複核門檻。

範例 rule：

```text
helmet_missing_candidate =
  worker_visible == yes
  AND missing_helmet == yes
  AND repeated across N sampled frames
  AND not contradicted by helmet detect/caption evidence
```

這樣可以避免單一 prompt 誤判直接變成事件。

## 可分享成果

`shareables/` 放的是已整理過的簡報與報告：

```text
shareables/
├── presentations/
│   └── exp03_behavior_fall_smoking_20260526.pptx
└── reports/
    └── exp03_behavior_fall_smoking_20260526_report.pdf
```

不要把原始 CCTV、行為影片、完整輸出 frame dump、私人工作筆記或未篩選的內部報告放進這個資料夾。

## 延伸系統設計

實際部署時建議把 Moondream 放在 YOLO 或傳統影像管線旁邊，負責輸出語意證據：

```text
RTSP Camera
  -> Frame Sampler
  -> Moondream Inference Worker
  -> Rule Engine
  -> Kafka Producer
  -> vision.event.safety
```

Moondream worker 負責 caption/query/detect；Rule Engine 負責 gate、交叉驗證、重複幀確認和事件格式化。這個分層能讓模型輸出可追溯，也能降低 prompt 誤判對告警品質的影響。
