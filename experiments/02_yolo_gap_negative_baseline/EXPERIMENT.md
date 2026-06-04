# EXP02_YOLO_GAP_NEGATIVE_BASELINE

Purpose: evaluate false positives on normal factory operation footage for safety cases that YOLO does not handle well.

Core code:

| File | Role |
| --- | --- |
| `scripts/run_baseline.py` | Dense negative-baseline inference over factory clips. |
| `scripts/analyze_baseline.py` | CSV/report statistics. |
| `scripts/generate_detail_report.py` | Detailed HTML analysis with frame evidence. |
| `scripts/run_occlusion_experiment.py` | Helmet/mask visibility and occlusion follow-up. |
| `scripts/run_attribution_experiment.py` | False-positive attribution under compliant-site assumption. |
| `scripts/run_occlusion_phase2.py` | Full-run strict helmet visibility prompt. |
| `scripts/run_occlusion_mask.py` | Full-run strict mask visibility prompt. |
| `scripts/run_occlusion_prompt_variants.py` | Prompt variants for helmet occlusion. |
| `scripts/run_mask_prompt_variants.py` | Prompt variants for mask visibility. |

Local input: `data/raw_videos/`

Local output: `experiments/02_yolo_gap_negative_baseline/outputs/`
