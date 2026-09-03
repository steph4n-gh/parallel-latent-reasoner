# Project: Parallel Latent Reasoner (PRLR) — Reproducible Pretrained Result on Apple Silicon

## Architecture
Parallel Recurrent Latent Deliberation (PRLR) is a continuous latent reasoning architecture running natively on Apple Silicon Metal GPUs. It replaces serial autoregressive Chain-of-Thought (CoT) generation with parallel continuous deliberation across $M$ distinct working memory slots with orthogonal learned anchors and slot-role embeddings, driven by a recurrent Transformer block with bounded sigmoidal residual scaling:

$$\alpha = \alpha_{\max} \cdot \sigma(\text{raw}\_\alpha)$$

The reproducible pretrained implementation operates as a vertical lane anchored to the official, verified **`google/gemma-2b-it`** model (2.506B parameters in bfloat16, SHA-256: `561656f8...` and `20fe2ee6...`, tokenizer SHA-256: `61a7b147...`).

### Data Flow & Execution Pipeline
1. **Model & Manifest Enforcement**:
   - Explicit package separation into `prlr.kernel` (pure recurrent tensor kernel), `prlr.compact` (scratch CI/learnability tests), and `prlr.gemma` (genuine pretrained Gemma lane).
   - `ModelManifest` enforces strict cryptographic verification of weights, tokenizers, runtime versions, and commit SHA before execution.
2. **Contextual Encoding & Slot Initialization**:
   - Real prompt tokens pass through official SentencePiece tokenizer and the frozen Gemma 2B backbone.
   - Contextual prompt hidden states $H_{\text{prompt}} \in \mathbb{R}^{B \times L \times 2048}$ are extracted from the pretrained layers (zero character-modulo fallback).
   - $M=16$ working memory slots are initialized with small orthogonal learned anchors plus explicit slot-role embeddings:
     $$S^{(0)} = E_{\text{slot\_role}} + W_{\text{ctx\_proj}} \bar{H}_{\text{prompt}}$$
3. **Weight-Tied Recurrent Block Unroll**:
   - Trainable recurrent adapter with prompt cross-attention projections, bidirectional self-attention, and scaled non-zero MoE/GeGLU feed-forward networks.
   - Bounded sigmoidal residual scaling prevents state explosion without unverified Lipschitz claims.
4. **Causal / Structured Decoder**:
   - Latent deliberation states are mapped into soft prompt prefix / cross-attention memory consumed by the pretrained causal decoder with KV caching, preserving causal masking, token positions, and EOS halting (Option A), or mapped to a structured prediction head with deterministic grammar (Option B).
   - Complete removal of the legacy repeated pooled-vector feedback loop (`norm(curr_h + 0.1 * tok_embed)`).
5. **Training & Masked Answer Cross-Entropy**:
   - Supervised via masked answer cross-entropy ($M_{\text{target}}$ masking prompt and padding tokens).
   - Zero synthetic thought traces, zero random noise teacher latents.
6. **Post-Hoc Calibrated E-Gate**:
   - Stopping threshold calibrated post-hoc on a sealed gate-training split using non-oracle signals (velocity, margin, entropy) to retain $\ge 99\%$ accuracy with $\ge 15\%$ depth reduction.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| 1 | Legacy Report Archival | Archive legacy reports to `results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md` | M1 | R0 | PLANNED |
| 2 | Legacy Weights Retirement | Move legacy adapter checkpoints to `checkpoints/legacy_invalid_objective/` | M1 | R0 | PLANNED |
| 3 | Evidence Status Mapping | Create `EVIDENCE_STATUS.md` mapping all claims to model, dataset, artifact, and audit notes | M1 | R0 | PLANNED |
| 4 | Pre-release Reclassification | Reclassify `CITATION.cff` and `pyproject.toml` to experimental pre-release status | M1 | R0 | PLANNED |
| 5 | Conditional Prose Generator | Refactor `benchmark.py` report generation to emit conditional prose strictly matching metrics | M1 | R0 | PLANNED |
| 6 | Package Namespacing | Partition into `prlr.kernel`, `prlr.compact`, and `prlr.gemma` | M2 | R1 | DONE |
| 7 | ModelManifest Dataclass | Implement `ModelManifest` tracking hashes, model ID, tokenizer, and runtime versions | M2 | R1 | DONE |
| 8 | Strict Manifest Validation | Reject unverified random models as Gemma; refuse init without weights unless `--random-init` | M2 | R1 | DONE |
| 9 | Pretrained Gemma 2B Lane | Real pretrained `google/gemma-2b-it` loader on Apple Silicon Metal GPU with health check | M3 | R2 | DONE |
| 10 | Contextual Hidden States | Extract $(B, L, 2048)$ hidden states from pretrained layers (purge character-modulo fallback) | M3 | R2 | DONE |
| 11 | Orthogonal Slot Anchors | Initialize $M$ slots with small orthogonal learned anchors and slot-role embeddings | M3 | R2, R4 | DONE |
| 12 | Trainable Recurrent Adapter | Per-layer prompt cross-attention projections and bidirectional slot reasoning | M3 | R2 | DONE |
| 13 | Principled MoE Initialization | Scaled non-zero initialization for MoE router and expert matrices | M3 | R4 | DONE |
| 14 | Bounded Residual Scaling | Parameterize residuals as $\alpha = \alpha_{\max} \cdot \sigma(\text{raw}\_\alpha)$ | M3 | R4 | DONE |
| 15 | 1-Step Gradient Flow Test | Verify 100% of trainable adapter parameters receive nonzero gradients | M3 | R4 | DONE |
| 16 | Causal / Structured Decoder | Soft prompt prefix / cross-attention memory into pretrained causal decoder (Option A/B) | M3 | R3 | DONE |
| 17 | Comprehensive Decoder Tests | Verify EOS halting, variable length, padding invariance, perturbation sensitivity | M3 | R3 | DONE |
| 18 | Solver-Backed Domain Lane | Curate finite schema tool routing / bounded arithmetic with deterministic oracle solver | M4 | R6 | DONE |
| 19 | 5-Way Split Schema | Train, Dev, Sealed Semantic Test, Sealed Gate Calibration, Extrapolation splits | M4 | R6 | DONE |
| 20 | Masked Answer CE Training | Loss computed strictly over target tokens with $M_{\text{target}}$ mask (zero loss on padding) | M4 | R5 | DONE |
| 21 | Milestone A Overfit | 100% exact-match overfit on 32 procedural examples | M4 | R5 | DONE |
| 22 | Milestone B Overfit | Valid structured syntax overfit on 256 examples | M4 | R5 | DONE |
| 23 | Milestone C Generalization | Nonzero held-out accuracy on unseen procedural instances | M4 | R5 | DONE |
| 24 | Controlled Ablation Suite | Controlled ablations: baseline, $T=0$, depth ($T \in \{1,2,4,8,12\}$), slots ($M \in \{1,4,8,16\}$), knockout/merge | M5 | R7 | DONE |
| 25 | Post-Hoc Calibrated E-Gate | Calibrate stopping threshold on gate split; verify $\ge 99\%$ accuracy retention, $\ge 15\%$ depth reduction | M5 | R8 | DONE |
| 26 | Separated Microbenchmark | Recurrent kernel FLOPs, bytes, memory, throughput (zero CoT claims) | M6 | R9 | PLANNED |
| 27 | Separated Semantic Benchmark | Real Gemma backbone, stage latencies, Pareto curves, bootstrap 95% CIs | M6 | R9 | PLANNED |
| 28 | CI Verification Guardrails | Automated CI tests for ground-truth isolation, 100% gradient flow, bounded residuals | M6 | R9 | PLANNED |
| 29 | Single-Command E2E Verification | Single reproducible script executing clean verification suite | M6 | R9 | PLANNED |
| 30 | Signed Claims Documentation | `CLAIMS.md` mapping 100% of claims to signed reproducible artifacts | M6 | R9 | PLANNED |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Legacy Claim & Artifact Closure | Features 1–5: Archive legacy reports to `COMPACT_MODEL_FAILURE_REPORT.md`, move legacy checkpoints, create `EVIDENCE_STATUS.md`, reclassify `CITATION.cff`/`pyproject.toml`, conditional prose in `benchmark.py` | None | DONE |
| M2 | System Separation & Model Manifest | Features 6–8: Namespace `prlr.kernel`, `prlr.compact`, `prlr.gemma`, `ModelManifest` dataclass, strict validation and rejection of unverified random models | M1 | DONE |
| M3 | Real Pretrained Lane, Decoder & Principled Init | Features 9–17: Pretrained `google/gemma-2b-it` vertical lane, contextual states $(B,L,2048)$, orthogonal slot anchors, trainable adapter, non-zero MoE init, bounded residuals, 1-step gradient tests, causal/structured decoder, decoder tests | M2 | DONE |
| M4 | Solver Lane, Data Splits & Masked CE Training | Features 18–23: Solver-backed lane, 5-way data splits, masked answer CE training, Milestones A, B, C overfit and generalization, checkpoint artifact | M3 | DONE |
| M5 | Controlled Ablations & Calibrated E-Gate | Features 24–25: Controlled ablations ($T, M$, knockout/merge), post-hoc E-gate calibration ($\ge 99\%$ accuracy retention, $\ge 15\%$ depth reduction) | M4 | DONE |
| M6 | Separated Benchmarks, CI Guardrails & Claims | Features 26–30: Kernel microbenchmark vs semantic benchmark, CI guardrails, single-command E2E verification, signed `CLAIMS.md` | M5 | IN_PROGRESS |

---

## Interface Contracts

### 1. Model Manifest Contract (`prlr.gemma.manifest` or `prlr.manifest`)
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    revision: str
    architecture: str
    weight_hash: str
    tokenizer_hash: str
    quantization: str
    runtime_versions: Dict[str, str]
    random_init: bool
    adapter_hash: Optional[str]
    source_commit: str

    def validate(self) -> bool: ...
```

### 2. Pretrained Gemma Backbone Contract (`prlr.gemma.backbone`)
```python
class PretrainedGemmaBackbone:
    def __init__(self, manifest: ModelManifest, allow_random_init: bool = False): ...
    def encode_prompt_context(self, prompt_tokens: mx.array) -> mx.array:
        """Returns contextual hidden states [B, L, 2048]."""
        ...
    def decode_tokens(self, soft_prefix_latents: mx.array, max_new_tokens: int, eos_token_id: int) -> Tuple[mx.array, str]:
        """Causal decoding preserving positions, causal mask, and EOS halting."""
        ...
```

### 3. Recurrent Latent Adapter Contract (`prlr.gemma.adapter`)
```python
class GemmaRecurrentAdapter:
    def __init__(self, dim: int = 2048, num_slots: int = 16, num_experts: int = 4): ...
    def unroll(self, prompt_context: mx.array, num_steps: int) -> DeliberationTrajectory:
        """Runs parallel Jacobi sweeps with bounded residuals alpha = alpha_max * sigmoid(raw_alpha)."""
        ...
```

### 4. Solver-Backed Lane & Dataset Splits Contract (`prlr.dataset.splits`)
```python
@dataclass
class DatasetSplits:
    train: List[Dict[str, Any]]
    dev: List[Dict[str, Any]]
    sealed_semantic_test: List[Dict[str, Any]]
    sealed_gate_calibration: List[Dict[str, Any]]
    extrapolation: List[Dict[str, Any]]
```

---

## Code Layout
```
projects/parallel_latent_reasoner/
├── pyproject.toml
├── CITATION.cff
├── README.md
├── EVIDENCE_STATUS.md
├── CLAIMS.md
├── verify_all.py
├── run_benchmark.py
├── run_kernel_microbenchmark.py
├── run_semantic_benchmark.py
├── src/
│   └── prlr/
│       ├── __init__.py
│       ├── manifest.py
│       ├── kernel/
│       │   ├── __init__.py
│       │   ├── recurrent_core.py
│       │   └── telemetry.py
│       ├── compact/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── scratch_model.py
│       └── gemma/
│           ├── __init__.py
│           ├── backbone.py
│           ├── adapter.py
│           ├── decoder.py
│           ├── solver_lane.py
│           ├── trainer.py
│           ├── egate.py
│           └── evaluation.py
├── tests/
│   ├── test_ground_truth_isolation.py
│   ├── test_manifest_integrity.py
│   ├── test_gradient_flow_1step.py
│   ├── test_bounded_residuals.py
│   ├── test_causal_decoder.py
│   ├── test_solver_lane_splits.py
│   ├── test_egate_calibration.py
│   └── ... (existing unit tests)
├── checkpoints/
│   ├── legacy_invalid_objective/
│   └── prlr_gemma_adapter.safetensors
└── results/
    ├── legacy_invalid_objective/
    │   └── COMPACT_MODEL_FAILURE_REPORT.md
    ├── kernel_microbenchmark.json
    ├── semantic_benchmark.json
    └── ablation_matrix.json
```
