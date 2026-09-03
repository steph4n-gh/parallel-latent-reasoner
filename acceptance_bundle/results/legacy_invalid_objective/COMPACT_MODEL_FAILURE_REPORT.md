# Archived Failure Analysis & Invalidation Report: Legacy PRLR Compact-Model Prototype

**Status**: RETRACTED & RETIRED (Legacy Invalid Objective)  
**Date of Retirement**: 2026-09-03  
**Original Report Dates**: 2026-09-02 to 2026-09-03  
**Evaluated Architecture**: MLX Scratch-Trained Compact Model ($D=256$, Slots $M=16$, Recurrent Depth $T=8$)  
**Legacy Adapter Artifact**: `checkpoints/legacy_invalid_objective/prlr_latent_adapter.npz`  
**Superceded By**: PRLR Pretrained Gemma 2B Lane (`PROJECT.md`, R0–R9)

---

## 1. Retraction & Invalidation Notice

This report serves as the formal archival record and failure analysis for the legacy Parallel Latent Reasoner (PRLR) compact-model prototype.

All prior claims of "frontier-grade accuracy", "diverse, non-degenerate token distributions", "zero repetitive token looping", and "22x speedup vs Autoregressive Chain-of-Thought" emitted in legacy benchmark reports are **hereby formally retracted and classified as invalid**.

### 1.1 Summary of Measured Empirical Failure
The legacy compact model prototype evaluated across the 25 curated cognitive tasks (`cognitive_suite.py`) exhibited total semantic failure:
- **Multi-Domain Reasoning Accuracy**: **0.0%** (0 out of 25 tasks passed). Target specification: $\ge 80.0\%$. Status: **FAIL**.
- **Information-Theoretic Shannon Entropy**: **$H = 0.00\text{ bits}$** across all emitted tokens. Target specification: $H \ge 1.0\text{ bits}$. Status: **FAIL**.
- **Max 4-Gram Token Repetition**: **13 consecutive repetitive 4-grams**. Target specification: $< 2$. Status: **FAIL**.
- **Emitted Text Output**: Across all tasks, the model collapsed into degenerate, repetitive symbol emissions such as `<<<<<<<<<<<<<<<<` or constant padding tokens.

### 1.2 Root Cause Analysis of Prototype Failure

1. **Broken Pooled-Vector Decoder Loop**:
   The discrete token generation loop across `models.py:MLXCompactGemmaModel.generate()`, `pipeline.py:PRLRPipeline`, and `eval_harness.py` used an unprincipled pooled-vector update:
   $$\mathbf{h}_{\text{curr}} = \text{RMSNorm}\left(\mathbf{h}_{\text{curr}} + 0.1 \cdot \text{Embed}(y_s)\right)$$
   This formulation completely lacked causal autoregressive attention, token position embeddings, KV caching, and EOS termination halting. The readout was incapable of preserving syntactic structure or conditionality.

2. **Unmasked Padding Loss in BPTT Training**:
   In `trainer.py:PRLRBPTTTrainer`, cross-entropy loss was calculated over padded token batches without a target attention mask ($M_{\text{target}}$). Consequently, the model received strong supervisory signals rewarding the prediction of padding tokens, directly incentivizing repetition loops and entropy collapse.

3. **Degenerate Working Memory Slot Initialization**:
   In `models.py:MLXPreludeProjection`, working memory slot embeddings were initialized to absolute zeros (`mx.zeros((1, M, D))`). All 16 slots began as identical copies of the pooled prompt representation. As confirmed by tensor inspection of `prlr_latent_adapter.npz`, slot vectors remained collinear throughout training (mean cross-slot difference $< 10^{-4}$), destroying parallel hypothesis representation.

4. **Zero-Initialized MoE Parameters**:
   In `models.py:MLXGemmaMoE`, expert gate weights, up-projection weights, and down-projection weights were initialized to all-zeros (`mx.zeros`), blocking gradient flow during backpropagation.

5. **Character-Modulo Fallback Tokenization**:
   In `dataset.py` and `pipeline.py`, prompt encoding fell back to `[ord(c) % vocab_size for c in prompt]`, which destroyed semantic token identities and subword structures.

6. **Violations of Non-Negotiable Evidence Rules**:
   - **Rule 4 Violation**: Serial recurrent microbenchmarks over 200 steps were mislabeled "Autoregressive CoT", claiming a false "22x wall-clock speedup".
   - **Rule 5 Violation**: 1-layer randomly initialized models with $D=3840$ and $D=2816$ were mislabeled as "Gemma 12B Q4" and "Gemma 26B A4B MoE" without verified weights or tokenizers.
   - **Rule 6 Violation**: Simulated CoT latencies in demonstration tools used hardcoded multipliers (e.g. 2.85 ms/token).
   - **Rule 8 Violation**: Markdown report generators emitted celebratory prose asserting "frontier-grade accuracy" and "diverse token distributions" despite the tabular metrics showing 0.0% accuracy and $H=0.00\text{ bits}$.

---

## 2. Consolidation & Archive of Legacy Reports

### 2.1 Archival of `BENCHMARK_REPORT.md` & `results/BENCHMARK_REPORT_LARGE_GEMMA4.md`

#### Measured Verification Gates Table (Original Data)
| Empirical Verification Gate | Target Specification | Measured Result | Status |
|---|:---:|:---:|:---:|
| **Multi-Domain Reasoning Accuracy** | $\ge 80.0\%$ | **0.0%** | ❌ FAIL |
| **Reasoning Phase Wall-Clock Speedup** | $\ge 15.0\times$ | **22.0x** | ⚠️ INVALID BASELINE |
| **Deliberation Phase Latency** | $\le 500.0\text{ ms}$ | **1.9 ms** | ✅ PASS (Microbenchmark Only) |
| **Peak Resident VRAM Memory** | $\le 6.0\text{ GB}$ | **0.04 GB** (43.2 MB) | ✅ PASS |
| **Thought Phase KV-Cache Expansion** | $+0.00\%$ (Constant $M=16$) | **+0.00%** | ✅ PASS |
| **Information-Theoretic Shannon Entropy** | $H \ge 1.0\text{ bits}$ | **H = 0.00 bits** | ❌ FAIL |
| **Max 4-Gram Token Repetition** | $< 2$ (No Repetition Loops) | **13** | ❌ FAIL |

#### Cognitive Domain Accuracy Breakdown (Original Data)
| Cognitive Domain | Cases | CoT Acc | PRLR Acc | Delib Latency | Speedup | Mean Entropy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Multi Constraint (MCS)** | 5 | 0.0% | **0.0%** | 2.0 ms | 22.0x | H=0.00 |
| **Winograd Schema (WSD)** | 5 | 0.0% | **0.0%** | 1.5 ms | 25.9x | H=0.00 |
| **Semantic Denoising (SDN)** | 5 | 0.0% | **0.0%** | 1.9 ms | 20.3x | H=0.00 |
| **Multi Clue Synthesis (CMS)** | 5 | 0.0% | **0.0%** | 1.8 ms | 21.4x | H=0.00 |
| **Action Tool Routing (ATR)** | 5 | 0.0% | **0.0%** | 2.1 ms | 20.3x | H=0.00 |
| **OVERALL TOTAL** | **25** | 0.0% | **0.0%** | **1.9 ms** | **22.0x** | **H=0.00** |

#### Verbatim Degenerate Emission Samples from Original Runs
- **Case `mcs_01_spacecraft_payload`**:
  - Expected: `"Beta, Gamma"`
  - Mode 1 Emitted: `<<<<<<<<<<<<<<<<`
  - Mode 2 Emitted: `<<<<<<<<<<<<<<<<`
  - Entropy: $H = 0.00\text{ bits}$ | Max 4-gram repetition: 13
- **Case `wsd_01_physical_affordance_trophy`**:
  - Expected: `"the trophy"`
  - Mode 1 Emitted: `<<<<<<<<<<<<<<<<`
  - Mode 2 Emitted: `<<<<<<<<<<<<<<<<`
  - Entropy: $H = 0.00\text{ bits}$ | Max 4-gram repetition: 13
- **Case `sdn_01_angry_customer_return`**:
  - Expected: `{"action": "REFUND", "order_id": "QX-99281", ...}`
  - Mode 1 Emitted: `<<<<<<<<<<<<<<<<`
  - Mode 2 Emitted: `<<<<<<<<<<<<<<<<`
  - Entropy: $H = 0.00\text{ bits}$ | Max 4-gram repetition: 13
- **Case `cms_01_whodunit_alibi_deduction`**:
  - Expected: `"Mrs. Peacock"`
  - Mode 1 Emitted: `<<<<<<<<<<<<<<<<`
  - Mode 2 Emitted: `<<<<<<<<<<<<<<<<`
  - Entropy: $H = 0.00\text{ bits}$ | Max 4-gram repetition: 13
- **Case `atr_01_financial_portfolio_rebalancer`**:
  - Expected: `{"tool_id": "T4", "tool_name": "rebalance_portfolio_weights", ...}`
  - Mode 1 Emitted: `<<<<<<<<<<<<<<<<`
  - Mode 2 Emitted: `<<<<<<<<<<<<<<<<`
  - Entropy: $H = 0.00\text{ bits}$ | Max 4-gram repetition: 13

---

### 2.2 Archival & Audit of `TEST_READY.md`
- **Original Claim**:
  - Claimed test suite status "READY / 100% PASSING" with 54 passing tests.
  - Section 3 claimed "Tier 4: Real-World Application Scenarios [ E2E Dual-Mode Transcripts, Speedup >= 25x, +0.00% VRAM ]".
- **Audit Finding**:
  - The 54 unit tests validated MLX array operations, probe calculations, and rubric parser functions on synthetic inputs. None of the tests validated semantic correctness or end-to-end task solving.
  - The "Speedup >= 25x" claim was derived entirely from comparing parallel sweeps against serial iterations on a 256-dim scratch model.
  - The document omitted disclosure of the 0.0% task accuracy and entropy collapse.

---

### 2.3 Archival & Audit of `TEST_INFRA.md`
- **Original Claim**:
  - Specified a 4-tier test pyramid asserting Tier 4 real-world validation of dual-mode transcripts, 25 cognitive tasks as native benchmarks, and Gemma 12B/26B presets.
- **Audit Finding**:
  - The 25 cognitive tasks had hardcoded ground-truth answers without generative diversity or solver verification.
  - The Gemma 12B and 26B presets were purely geometric tensor configurations instantiated with random normal weights without official Google weights or SentencePiece tokenizers.
  - The evaluation harness compared an invalid pooled-vector loop against hardcoded reasoning traces.

---

### 2.4 Archival & Audit of `README_SPACES.md`
- **Original Claim**:
  - Frontmatter metadata: `short_description: 20x Faster Continuous Latent Deliberation on Apple Silicon`.
- **Audit Finding**:
  - The claim of "20x Faster" violated Non-Negotiable Evidence Rule 9 ("No speedup claim may be made unless output quality is matched under a documented non-inferiority criterion").
  - The compact prototype produced garbage tokens (`<<<<<<<<<<<<<<<<`), meaning speedup was measured over null output.

---

### 2.5 Archival & Audit of `docs/guides/killer_use_cases.md`
- **Original Claim**:
  - Promoted three "Killer Use Cases" with ready-to-run code snippets claiming:
    1. Sub-3ms Autonomous Agent Tool Routing: Outputting `refund_order(order_id="ORD-991", amount=74.50)` in 2.14 ms using `checkpoints/prlr_latent_adapter.npz` on `compact_test`.
    2. Conversational Intent Denoising: Outputting `download_invoice(account_id="ACC-410", year="2025")` in 1.85 ms.
    3. Multi-Constraint Satisfaction: Selecting portfolio allocations.
- **Audit Finding**:
  - Running `compact_test` with `checkpoints/prlr_latent_adapter.npz` on these prompts does **not** output `refund_order(...)` or `download_invoice(...)`; it outputs `<<<<<<<<<<<<<<<<` with 0% accuracy.
  - The examples presented hypothetical future capabilities as working software, in direct violation of `docs/documentation_policy.md` and repository integrity rules.

---

## 3. Relocation of Legacy Adapter Weights

All adapter weights trained under the invalid objective have been quarantined to `projects/parallel_latent_reasoner/checkpoints/legacy_invalid_objective/`:
- `prlr_latent_adapter.npz` (256D compact adapter, step 8)
- `prlr_latent_adapter.safetensors`
- `prlr_latent_adapter_step_3.npz`
- `prlr_latent_adapter_step_4.npz`
- `prlr_latent_adapter_step_8.npz`
- `prlr_latent_adapter_step_12.npz`
- `prlr_latent_adapter_step_24.npz`
- `prlr_latent_adapter_step_36.npz`
- `prlr_math_adapter.npz`

These weights must never be loaded as defaults or claimed as capable of language generation.

---

## 4. Replacement Charter & Next Steps

The repository has transitioned to the verified roadmap defined in `PROJECT.md`:
1. **R1 (Milestone 2)**: Namespacing into `prlr.kernel`, `prlr.compact`, and `prlr.gemma` with cryptographic `ModelManifest`.
2. **R2 (Milestone 3)**: Official pretrained `google/gemma-2b-it` backbone on Apple Silicon Metal GPU with real contextual hidden states and orthogonal slot anchors.
3. **R3 (Milestone 3)**: Causal/structured decoder replacing the pooled-vector feedback loop.
4. **R4 (Milestone 3)**: Principled non-zero initialization and bounded sigmoidal residual scaling.
5. **R5 (Milestone 4)**: Masked answer cross-entropy training ($M_{\text{target}}$ masking).
6. **R6 (Milestone 4)**: Solver-backed procedural datasets with clean train/dev/sealed test splits.
7. **R7 (Milestone 5)**: Rigorous controlled ablations on depth and memory slots.
8. **R8 (Milestone 5)**: Post-hoc calibrated E-gate retaining $\ge 99\%$ accuracy with $\ge 15\%$ depth reduction.
9. **R9 (Milestone 6)**: Separated kernel microbenchmarks and semantic benchmarks, CI guardrails, and signed `CLAIMS.md`.
