# Demo — Moondream Behavior Detection

獨立的展示介面（主管簡報用），跟 `evaluation/` 完全隔離。設計成**像一個真實的監控工具**，
而不是手繪的假介面。錄影展示時用三個真實的畫面：

| 畫面 | 內容 |
|------|------|
| 視窗 **LIVE FEED** | 來源影片，主執行緒流暢播放。極簡 CCTV 角標（REC、檔名、時間碼、進度條）。**不疊判定** —— live feed 比模型正在看的影格快好幾秒，混在一起會誤導。 |
| 視窗 **ANALYSIS** | 一次處理一張、嚴格循序：抓當前影格 → 分析（分析中**清空、不顯示任何判定**，只有底部 ANALYZING 徽章）→ 完成後在同一幀疊上大字判定（SMOKING / FALL）→ **停留 `--hold` 秒（預設 0.5s）** → 抓新的一張，如此循環。邊框固定中性色（**不閃色**），顏色只在判定文字（紅 DETECTED / 綠 CLEAR）。底部顯示每張影像的平均處理時間（input→output）。 |
| 終端機 | `rich` 即時結果表格，每次推論一行（時間 · SMOKING · FALL · 延遲），讀起來像真的工程工具。 |

推論在背景執行緒進行，影片不會因為等模型而卡住。

## 執行

```bash
source ../.venv/bin/activate

# 跳出檔案選擇視窗（可多選，依選取順序播放）
python demo.py

# 指定影片，依序播放，播完循環（適合「跌倒 → 抽菸」連播）
python demo.py --video ../data/0520GrokVideo/grok-video-fall.mp4 \
               "../data/0520GrokVideo/grok-video-d5938d27-e6b2-4e23-88e2-61424118cebc (1).mp4"

# 整個資料夾依檔名排序
python demo.py --folder ../data/0520GrokVideo

# 多個資料夾合併成一個播放清單（依檔名排序、去重）
python demo.py --folder ../data/0520GrokVideo ../data/0520GrokVideo_2

# 短片放慢播放，讓偵測節奏跟得上
python demo.py --folder ../data/0520GrokVideo --speed 0.5

# 加速推論：降低送模型的解析度（~3 倍）
python demo.py --folder ../data/0520GrokVideo --max-width 512

# 結果定格更久（預設 1.5s）
python demo.py --folder ../data/0520GrokVideo --hold 2.5
```

啟動後把兩個視窗擺好（程式預設並排在左上），錄影時把這兩個視窗 + 跑程式的終端機一起框進去即可。

## 操作鍵（焦點放在任一視窗）

| 鍵 | 功能 |
|----|------|
| SPACE | 暫停 / 繼續 |
| N | 跳下一部影片 |
| Q / Esc | 離開 |

## 設計重點

- **三個真實畫面，零假殼**：不在單一視窗裡手繪假的標題列 / 假面板 —— 用真的 OS 視窗 + 真的終端機。
- **播放與推論分離**：影片在主執行緒以原生 FPS 流暢播放；Moondream 在背景執行緒推論。
- **循序、連貫、不跳動**：背景執行緒自己定節奏（分析一張 → 停 `--hold` 秒 → 抓新的一張），顯示端只反映它的狀態。一次只處理一張、依序呈現，不會跳來跳去，也不會顯示不屬於當前畫面的判定。
- **分析中清空**：每次都是重新偵測，分析中絕不殘留上次結果。判定只在完成時、和它自己那一幀一起出現。
- **不閃色**：邊框固定中性色，避免「整個畫面突然變色」。顏色只在判定文字。
- **ANALYZING 徽章**：分析中時，影像底部中央有黑色徽章（閃爍圓點 + `ANALYZING…` + 跳動秒數），即使連續幾幀畫面相近也明顯在運作。
- **平均處理時間**：ANALYSIS 底部與終端機都顯示每張影像的平均 input→output 時間，主管直接看到真實速度。
- **辨識項目**：抽菸、跌倒（direct 問法，沿用 `evaluation/` 的測試 prompt）。

## 效能備註（Apple MPS，實測）

| 送模型解析度 | 單次偵測延遲（encode + 2 query） |
|---|---|
| 原始（1104px） | ~8.5s（encode ~8s 為瓶頸） |
| `--max-width 512` | **~3s** |

- query 本身只佔 ~0.5s，瓶頸在 `encode_image`；高解析度會觸發 tiling 而變慢。
- **增加辨識任務幾乎不增加延遲**（成本在編碼，不在 query 數量）。
- 啟動約 15s 載入 + 暖機（暖機在載入畫面就跑完，避免第一次偵測卡在 MPS 冷啟動）。
- 要更低延遲需更強硬體（如 HPC H100 / moondream3）。

## 版面預覽

```bash
python demo.py --selftest --folder ../data/0520GrokVideo
```

不載入模型，輸出 `_selftest_live.png`、`_selftest_analyzing.png`（分析中、無判定）、`_selftest_result.png`（完成、有判定），並在終端機印一張範例結果表。
