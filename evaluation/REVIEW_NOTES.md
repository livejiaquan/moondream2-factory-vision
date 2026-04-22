# 報告閱讀回饋與實驗紀錄

本檔供邊讀 `detailed_analysis_report.html` 邊記下想法；也作為**最後一次程式修改**或**新實驗 run** 的備忘。

---

## 最後更新（由維護者或你手動填）

| 項目 | 內容 |
|------|------|
| 日期 | 2026-04-15 |
| 對應 baseline Run ID | `20260415T102259Z`（**舊跑仍含** `unauthorized_area`；新路徑請重跑 baseline） |
| 主報告 HTML | `outputs/20260415T102259Z/detailed_analysis_report.html` |
| 程式／設定最近異動摘要 | 自 `gap_baseline.json` **移除** `unauthorized_area`（現場無此情境）；`generate_detail_report` 已對齊。 |

---

## 閱讀回饋（依時間往下加）

> 你口頭或訊息裡的想法可請協作者貼在這裡，或你自己直接編輯下方的清單。

- **2026-04-15**：不需要 `unauthorized_area`（現場沒有「未授權區域」這類情境）。
- **2026-04-15**：Prompt 寫法需再驗證是否最適合 **Moondream2**（見下方待辦）。

---

## 待辦／下次實驗想驗證的事項

> 從回饋收斂出的行動項目。

- [ ] **Prompt 針對 Moondream2 的適用性**：對照官方／社群範例、`query` 最佳實踐；可規劃**小規模對照實驗**（同一批幀、僅改 prompt）量 FP / 一致性，再定稿各 `query_checks` 英文題。
- [ ] 移除 `unauthorized_area` 後請 **重跑** `run_baseline.py` 產出新 `run_id`，再按需跑 occlusion／attribution 與 `generate_detail_report.py`，舊 HTML 才會與設定一致。

---

## 新實驗跑完後請更新

1. 將上表「日期」「Run ID」「主報告路徑」改成新的一輪。
2. 若有新發現，可在「閱讀回饋」加一則註明 *YYYY-MM-DD · run_id*。

**常用指令（專案根目錄下 `evaluation/`）：**

```bash
# 新基線
python scripts/run_baseline.py --every 10 --annotate   # 參數依需要

# 可選：遮擋／歸因（需先有 baseline run id）
python scripts/run_occlusion_experiment.py --baseline-run <RUN_ID>
python scripts/run_attribution_experiment.py --baseline-run <RUN_ID>

# 重產詳報
python scripts/generate_detail_report.py --run <RUN_ID>
```
