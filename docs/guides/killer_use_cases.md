# Guide: Target Architectural Blueprints & Migration Plan

> ⚠️ **ARCHITECTURAL SCOPE & PROTOTYPE RETRACTION NOTICE**
> **Status**: CONCEPTUAL BLUEPRINTS / EXPERIMENTAL ROADMAP (2026-09-03)
> The integration recipes and code snippets in this guide previously claimed operational tool routing and semantic extraction using the compact prototype (`compact_test` with `prlr_latent_adapter.npz`). As documented in [`COMPACT_MODEL_FAILURE_REPORT.md`](../../results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md), that prototype exhibited **0.0% task accuracy**, **$H=0.00\text{ bits}$ entropy collapse**, and repetitive looping emissions (`<<<<<<<<<<<<<<<<`) due to the broken pooled-vector decoder loop.
>
> The examples below illustrate the **target architectural interfaces** currently being rebuilt on top of a verified pretrained **Gemma 2B** backbone with a causal decoder head and masked cross-entropy training (Phases R1–R6 in `PROJECT.md`).

---

## 1. Motivation: Latent Deliberation vs. Autoregressive CoT

In conventional LLM-based agent systems, sequential reasoning requires emitting long strings of tokens:
- **Autoregressive CoT**: Emits hundreds of intermediate `<thought>` tokens, incurring $O(N)$ KV-cache memory expansion and high end-to-end latency (seconds per step).
- **Parallel Latent Deliberation (PRLR)**: Executes iterative continuous updates across $M=16$ continuous working memory slots in GPU SRAM/unified memory before causal decoding, maintaining $+0.00\%$ KV-cache expansion during thought iterations.

---

## 2. Target Use Cases (Planned Pretrained Capabilities)

### Target Lane 1: Autonomous Agent Tool & Action Routing
- **Objective**: Parallel deliberation over candidate tool definitions to output structured JSON tool invocations.
- **Implementation Status**: Under active redevelopment under R6 (Solver-Backed Domain Lane) and R3 (Causal/Structured Decoder). The legacy 256D adapter is retired.

### Target Lane 2: Multi-Constraint Satisfaction & Optimization
- **Objective**: Resolving complex constraint sets (budget, latency, resource quotas) via parallel slot relaxation.
- **Implementation Status**: Benchmarked under R6 with solver-backed ground truths.

### Target Lane 3: Constant-Memory Edge Deliberation
- **Objective**: Fixed-memory working state on embedded Apple Silicon GPUs without KV-cache expansion during deliberation sweeps.
- **Verified Property**: Zero KV-cache growth during recurrent unrolls is verified by unit tests (`test_stress_stability.py`).

---

## 3. Authoritative Architecture & Evidence References

For verified measurements, current status, and active project requirements, please refer to:
- **[`/Volumes/Storage/qan_transformers/PROJECT.md`](../../../../PROJECT.md)** — Active Roadmap and Engineering Specifications (R0–R9)
- **[`EVIDENCE_STATUS.md`](../../EVIDENCE_STATUS.md)** — Comprehensive Evidence Status Matrix
- **[`COMPACT_MODEL_FAILURE_REPORT.md`](../../results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md)** — Full Retraction and Failure Analysis of the Compact Prototype
