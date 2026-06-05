# EXP02_YOLO_GAP_NEGATIVE_BASELINE

## Purpose

This experiment separates normal-operation false-positive testing from the first feasibility test. The question here is narrower: when the site is assumed to be compliant or normal, which Moondream prompts still incorrectly produce safety-event signals?

This is the YOLO-gap baseline because the target cases are semantic or context-heavy checks that a basic YOLO detector usually cannot solve by itself.

## Data Scope

Input data is intentionally not committed.

| Local Code | Expected Path | Description |
| --- | --- | --- |
| `S01_FACTORY_CCTV` | `data/raw_videos/` | Normal factory/tanker operation clips used as negative baseline footage. |

The latest historical baseline family started on 2026-04-15. Old timestamp run IDs are kept only in local outputs.

## Method

The baseline runner samples factory footage more densely than experiment 1 and evaluates configured query and detection checks. The important assumption is that these clips are treated as negative examples for selected safety events. A `yes` answer on a negative-only check is therefore a false-positive candidate that needs review.

Follow-up scripts test whether the false positive came from:

- helmet or mask occlusion;
- poor worker visibility;
- ambiguous wording in the prompt;
- model hallucination under a compliant-site assumption;
- detect/count evidence that disagrees with a yes/no answer.

## Code

| File | Role |
| --- | --- |
| `scripts/run_baseline.py` | Main negative-baseline inference over factory clips. |
| `scripts/analyze_baseline.py` | CSV/report statistics for a completed baseline run. |
| `scripts/generate_detail_report.py` | HTML analysis with frame-level evidence and follow-up links. |
| `scripts/run_occlusion_experiment.py` | Helmet/mask visibility and occlusion follow-up. |
| `scripts/run_attribution_experiment.py` | False-positive attribution under compliant-site assumption. |
| `scripts/run_occlusion_phase2.py` | Strict helmet visibility prompt over a full run. |
| `scripts/run_occlusion_mask.py` | Strict mask visibility prompt over a full run. |
| `scripts/run_occlusion_prompt_variants.py` | Prompt variants for helmet occlusion. |
| `scripts/run_mask_prompt_variants.py` | Prompt variants for mask visibility. |

## Configs

| File | Role |
| --- | --- |
| `configs/gap_baseline.json` | Baseline query/detect definitions. |
| `configs/gap_baseline_v2.json` | Follow-up baseline settings after removing unsupported site scenarios. |
| `configs/occlusion_queries.json` | Focused occlusion prompts. |
| `configs/occlusion_variants.json` | Prompt wording variants. |
| `configs/attribution_queries.json` | Attribution checks for false-positive review. |

## How to Run

```bash
source .venv/bin/activate
cd experiments/02_yolo_gap_negative_baseline
python scripts/run_baseline.py --every 10 --annotate
python scripts/analyze_baseline.py --run latest
python scripts/generate_detail_report.py --run latest
```

Optional follow-up:

```bash
python scripts/run_occlusion_experiment.py --baseline-run <RUN_ID>
python scripts/run_attribution_experiment.py --baseline-run <RUN_ID>
```

Default output:

```text
experiments/02_yolo_gap_negative_baseline/outputs/<run_id>/
```

## Output Files

| File | Meaning |
| --- | --- |
| `results.jsonl` | Full per-frame model output. |
| `summary.csv` | Flattened labels, answers, detections, and caption keyword hits. |
| `baseline_report.md` | Aggregate false-positive summary. |
| `detailed_analysis_report.html` | Human-review report with frame evidence when generated. |
| `annotated/` | Optional annotated frames. |

## Key Findings

- Normal-operation footage is required before any positive demo result can be trusted.
- Helmet and mask prompts are sensitive to visibility and occlusion, so they need a worker-visible gate.
- Unsupported site concepts should be removed from the config instead of kept as vague prompts.
- Moondream is useful as a semantic evidence generator, but the alarm decision needs a rule engine and repeated-frame confirmation.

## Interpretation Notes

This experiment does not prove a detector is accurate in the field. It identifies false-positive risk and prompt failure modes on negative footage. A useful production design should combine this baseline with positive samples and explicit rule logic.
