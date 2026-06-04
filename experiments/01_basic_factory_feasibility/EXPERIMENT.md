# EXP01_BASIC_FACTORY_FEASIBILITY

Purpose: initial factory CCTV feasibility test for Moondream2.

Core code:

| File | Role |
| --- | --- |
| `scripts/moondream_cli.py` | Single-image CLI for caption/query/detect/point. |
| `scripts/run_feasibility.py` | Video frame sampler and safety-query runner. |

Local input: `data/raw_videos/`

Local output: `experiments/01_basic_factory_feasibility/outputs/`

Historical scope:

- 2026-04-02: initial sparse feasibility run.
- 2026-04-04: expanded query round, treated as part of experiment 1.

The original detailed text reports contain local source paths and are kept under `_local_sensitive/`.
