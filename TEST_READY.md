# Legacy Test Readiness Document: Retracted & Archived

> ⚠️ **RETRACTION & ARCHIVAL NOTICE**
> **Status**: RETRACTED / INVALIDATED (2026-09-03)
> The claims in this document regarding "Tier 4: Real-World Scenarios [ Speedup >= 25x, +0.00% VRAM ]" and production readiness were based on a scratch-trained compact prototype ($D=256$) that suffered from total semantic collapse (0.0% accuracy, $H=0.00$ bits entropy, 13-token repetition loops). Unit test passes on synthetic arrays do not constitute production readiness.
>
> All historical findings have been archived in:
> **[`results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md`](results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md)**
>
> The verified test runner for active code is:
> ```bash
> pytest projects/parallel_latent_reasoner/tests/ -v
> ```
> For authoritative project roadmap and evidence status, see:
> **[`/Volumes/Storage/qan_transformers/PROJECT.md`](../../PROJECT.md)** and **[`EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md)**.
