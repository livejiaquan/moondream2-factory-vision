# Evaluation — Moondream YOLO-Gap 測試

本目錄用於評估 Moondream 在 YOLO 做不到（或做得不好）的場景中的表現，
獨立於主專案，方便後期查看與追蹤。

## 背景

公司目前使用 YOLO 做安全檢測（安全帽、人員偵測等），但 YOLO 在以下場景有限制：

| YOLO Gap | 原因 | Moondream 可能的做法 |
|----------|------|---------------------|
| 抽菸 | 香菸太小，難標註訓練 | query 問答 + detect cigarette |
| 異常行為 | YOLO 只框物件，不理解行為 | query 問「是否有不安全行為」 |
| 語意 PPE 判斷 | 需關聯 person + helmet 位置 | query 語意問答 |
| 場景描述 | YOLO 無文字輸出 | caption 產生自然語言證據 |
| 開放式偵測 | YOLO 需重新訓練 | detect 任意物件名稱 |

## 目錄結構

```
evaluation/
├── README.md              ← 你正在看的這個檔案
├── configs/
│   ├── gap_baseline.json       ← 負樣本基線配置
│   ├── occlusion_queries.json  ← 遮擋／可見性實驗用 prompt
│   └── attribution_queries.json ← 誤判歸因實驗（較嚴格英文問法）
├── scripts/
│   ├── run_baseline.py         ← 負樣本基線測試
│   ├── analyze_baseline.py     ← 基線統計報告
│   ├── generate_detail_report.py ← HTML 詳細報告
│   ├── run_occlusion_experiment.py ← 遮擋／角度實驗（依賴某次 baseline 的 results.jsonl）
│   └── run_attribution_experiment.py ← 誤判歸因實驗（合規假設下區分影像 vs 模型堅稱）
└── outputs/               ← 測試結果（不推 GitHub）
    └── <run_id>/
        ├── results.jsonl      每幀完整推理結果
        ├── summary.csv        扁平化統計表
        ├── baseline_report.md 自動產出的統計報告
        ├── run_meta.json      執行參數紀錄
        └── annotated/         標註圖（選用）
```

## 策略 A：負樣本基線測試

### 概念

用現有 `data/` 的 4 支正常作業影片做密集抽幀測試。
因為資料全是「正常」的，任何被判為異常的結果都是 **False Positive（誤報）**。

這個測試能回答：**Moondream 在正常畫面上會不會亂報？**

### 執行方式

```bash
# 1. 確保虛擬環境已啟動且安裝了依賴
#    需要：torch, transformers==4.46.3, Pillow, opencv-python
source ../.venv/bin/activate

# 2. 跑基線測試（預設每 5 秒抽一幀）
python scripts/run_baseline.py

# 可自訂參數
python scripts/run_baseline.py --data ../data --every 10 --annotate --limit 20

# 3. 分析結果，產出報告
python scripts/analyze_baseline.py --run latest
```

### 參數說明

**run_baseline.py**

| 參數 | 說明 | 預設 |
|------|------|------|
| `--data` | 影片/圖片資料夾 | `../data/` |
| `--config` | 測試配置 JSON | `configs/gap_baseline.json` |
| `--every` | 抽幀間隔（秒） | 5（來自 config） |
| `--limit` | 最多處理幾幀 | 0（不限） |
| `--annotate` | 儲存標註圖 | 關閉 |

**analyze_baseline.py**

| 參數 | 說明 | 預設 |
|------|------|------|
| `--run` | run_id 或 `latest` | `latest` |

### 輸出解讀

報告會包含：

1. **Query FP 率**：每個 YOLO Gap 場景被誤報的比例
2. **Worker Gate 效果**：加入「是否有人」gate 後 FP 率的變化
3. **Detect 統計**：各物件的框數
4. **Caption 警示**：描述文字中是否出現異常關鍵詞
5. **結論與建議**：哪些場景可推進、哪些需規則保護

## 測試的場景（gap_baseline.json）

### YOLO Gap（正常畫面應全部為 No）
- `smoking_visible` — 抽菸
- `abnormal_behavior` — 異常行為
- `person_falling` — 倒地

### YOLO 增強（Moondream 用語意輔助）
- `missing_helmet` — 安全帽缺失
- `missing_face_mask` — 面罩缺失

### 正面基準（應為 Yes）
- `wheel_chock_present` — 輪擋存在

### 輔助
- `worker_visible` — 用作 Worker Gate 判斷
- `scene_summary` — 場景文字描述

## Moondream API 快速參考

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision="2025-01-09",
    trust_remote_code=True, torch_dtype=torch.float16,
).to("mps").eval()  # 或 "cuda" / "cpu"

enc = model.encode_image(pil_image)

# 四種能力
caption = model.caption(enc, length="normal")["caption"]       # 文字描述
answer  = model.query(enc, "Is there a person?")["answer"]     # 問答
objects = model.detect(enc, "person")["objects"]                # 物件偵測 → bbox
points  = model.point(enc, "person's head")["points"]          # 定位 → 座標
```

### 座標格式
- **detect**: `{"x_min": 0.12, "y_min": 0.20, "x_max": 0.41, "y_max": 0.82}`（0-1 歸一化）
- **point**: `{"x": 0.28, "y": 0.18}`（0-1 歸一化）
- 像素換算：`px_x = int(x * image.width)`

### 未來 Pipeline 接入

```
RTSP Camera → cv2.VideoCapture(url)
  → Frame Sampler（每 N 秒取一幀）
  → Moondream Inference Worker
  → Rule Engine（worker gate + 交叉驗證）
  → Kafka Producer → vision.event.safety topic
```

詳見 `codex_workspace/docs/pipeline_design.md`

## 遮擋／角度實驗（補 YOLO「框不到≠沒戴」）

**問題**：安全帽／面罩「沒偵測到」可能是背對、遮擋、遠景，YOLO 與 `detect helmet` 都會有相同限制。

**做法**：先跑完某次 baseline（產生 `results.jsonl`），再對**同一時間點**從原始影片擷取畫面（無標註框），額外問兩個問題——是否**主要因視角／遮擋／距離**而難以看清，而非畫面清楚顯示未配戴。

```bash
python scripts/run_occlusion_experiment.py --baseline-run 20260415T102259Z
```

輸出在 `outputs/<baseline_run>_occlusion/<timestamp>/`（`occlusion_results.jsonl`、`occlusion_report.md`）。可用於規則模擬：若 `missing_helmet=yes` 但 `helmet_visibility_limited=yes`，是否降級告警。

**與主報告合併**：跑完遮擋實驗後，再執行 `python scripts/generate_detail_report.py --run <baseline_run>`，會自動把該實驗（方法、統計、每幀從影片擷取的圖片與回答）寫入 `outputs/<run>/detailed_analysis_report.html` 的獨立章節。

## 誤判歸因實驗（合規假設：現場應已配戴）

**問題**：在「工人理論上都戴了安全帽／面罩」的前提下，baseline 若仍出現 `missing_*=yes` 或 detect 未框到帽，需要區分：**多少可歸因於影像**（角度、遮擋、距離導致「看不清楚」）vs **模型仍聲稱能清楚看到違規**。

**做法**：對 baseline 中挑出的誤判幀（預設與遮擋實驗同一子集邏輯），用 **較嚴格的英文問法**（見 `configs/attribution_queries.json`）詢問 Moondream，主要指標為 `clear_absence_helmet`：模型是否承認能**清楚、無歧義地**看到「未戴帽」。若為 No，較適合歸類為影像／不確定因素，便於估算「改規則／降級告警」能改善的比例；若為 Yes，則較像模型判讀與現場認知衝突。

```bash
python scripts/run_attribution_experiment.py --baseline-run 20260415T102259Z
```

輸出在 `outputs/<baseline_run>_attribution/<timestamp>/`（`attribution_results.jsonl`、`attribution_report.md`）。

**與主報告合併**：跑完後再執行 `python scripts/generate_detail_report.py --run <baseline_run>`，會在 HTML 中 **Detect 誤框章節之後、遮擋實驗章節之前** 插入「誤判歸因」統計與方法說明（遮擋實驗為較早問法，兩者可對照）。可用 `--skip-attribution` 略過該章節。

## 後續測試計畫

- **策略 B**：人工製造偽異常圖片，做正負對比
- **策略 C**：用公開資料集補充正樣本
- **策略 D**：Prompt 對抗測試
- **YOLO vs Moondream 對照**：同圖同時跑兩個模型
