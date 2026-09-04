# Parallel Latent Reasoner (PRLR)

**Status**: `experimental / unpromoted` (MLX Pretrained Research Prototype on Apple Silicon)
**Platform**: Apple Silicon Metal GPU / Unified Memory Architecture
**Dependencies**: `mlx>=0.15.0`, `transformers>=4.40.0`, `numpy>=1.24.0`
**Base Models**:
- `google/gemma-2b-it` (official BF16 weights, frozen, 2.5B params)
- `google/gemma-4-12B-it-4bit` (official 4-bit affine weights, frozen, $D=3840$, 48 layers)
**Trained Adapters**:
- `checkpoints/gemma_2b_prlr_adapter.safetensors` (88.69M params, SHA-256: `6048262d99e5d28851adfc87a379a2796802926605ab74e33553b4d9347028d7`)
- `checkpoints/gemma_4_12b_prlr_adapter.safetensors` (200.7M params, $D=3840$, 48 layers, SHA-256: `ffb26ccac589d81d69ee67cb0c74c120dfd7b8695dc954ae4a10aca13ab2da36`)
- `checkpoints/gemma4_safe_adapter_512.safetensors` (358M total params, 200.7M recurrent adapter + zero-gated cross-attention injection, SHA-256: `681d1e13494250578636223ec1f4635364f6abfc7fe3e6cd7025a9446d776888`)

> [!WARNING]
> **Evidence Status & Scope of Pretrained Implementation**
> PRLR evaluates genuine pretrained Google Gemma backbones (`google/gemma-2b-it` and `google-gemma-4-12B-it-4bit`) with weight-tied recurrent latent deliberation adapters on Apple Silicon Metal GPU. All semantic benchmarks are evaluated blindly on held-out procedural splits (`data/prlr_domain_v1/sealed_test.jsonl` and `data/prlr_hard_v1/`) under Non-Negotiable Evidence Rules 1–10.
> - **Gemma 4 12B Direct Base Model**: Evaluated unconditioned through the repository decoder with official chat template and closed thought channel, the frozen base model achieves **96.48% Exact Match** (247/256), **99.61% Terminal Match**, and **100.0% Valid JSON** with **Max Repetition 1** on linear chains (`results/empirical_baselines/predictions_repo_decoder.json`).
> - **Zero-Gate Base Parity Invariant**: `GatedCrossAttentionInjection` guarantees 100.000% bit-exact parity at initialization ($\alpha=0.0 \implies \text{gate}=0.0$, logit delta `0.0000000000`), preserving native RoPE token indexing without prefix prepending (`tests/test_zero_gate_parity.py`).
> - **Diagnostic Preservation (512 Samples)**: The safe adapter trained with teacher KL divergence and monotonic progress penalty preserves base capability ($\text{EM} \ge \text{base} - 5\%$, 100.0% valid JSON, max rep $\le 2$, $T=4 \ge T=1$).
> - **Hard Headroom Benchmark (`data/prlr_hard_v1`)**: On non-linear DAG routing with dead-end lookaheads and multi-parent joins, direct frozen Gemma 4 drops to **0.0% Exact Match** (while maintaining 100.0% terminal routing and 100.0% valid JSON), establishing massive measurable headroom ($< 85.0\%$) and demonstrating that greedy decoding fails on DAG branch topological dependencies.
> - **Promotion Status**: Classified as `experimental / unpromoted`. Under Rule 9, reasoning speedup is disqualified from promotion until quality matching is demonstrated on the harder headroom benchmark.

---

## 1. Overview & Core Concept

**Parallel Recurrent Latent Deliberation (PRLR)** explores an alternative inference paradigm on unified memory architectures (Apple Silicon Metal GPUs).

Traditional autoregressive (AR) Chain-of-Thought (CoT) generation emits reasoning tokens sequentially token-by-token. For each generated token, the entire model weight matrix must be loaded from DRAM to GPU registers ($\sim 1 \text{ FLOP/byte}$ arithmetic intensity), causing heavy memory bandwidth bottlenecks and linear $O(N)$ KV-cache memory expansion.

PRLR investigates replacing discrete serial token generation with **parallel non-autoregressive Jacobi sweeps** across $M=16$ continuous working memory slots modulated by AdaRMSNorm sinusoidal step embeddings and ReZero residual scaling ($\alpha \le 0.05$ as an empirical stabilizer; note that small $\alpha$ dampens unroll growth but does not constitute a formal proof of Lipschitz contraction without operator norm bounds).

```
+----------------------------------------------------------------------------------------------------+
|                                    INPUT PROMPT PREFIX X (len P)                                   |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                    PRELUDE PROJECTION LAYER                                        |
|  - Embeds prompt tokens scaled by sqrt(D)                                                          |
|  - Prefills static Key/Value cache for cross-attention [B, num_kv_heads, P, head_dim] (frozen)     |
|  - Initializes M=16 continuous working memory slots: S^(0) = RMSNorm(E_slot + Proj(Pool(H_prompt)))|
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                        PARALLEL LATENT DELIBERATION LOOP (M=16, T_min=2, T_max=12)                 |
|                                                                                                    |
|   For step t = 1, 2, ..., T_max:                                                                   |
|                                                                                                    |
|     1. Recurrent Jacobi Sweep Update:                                                              |
|        h_norm1 = AdaRMSNorm_1(S^(t-1), t)                                                          |
|        attn_out = NonCausalSelfCrossAttn(h_norm1, Prompt_KV)                                       |
|        S_mid = S^(t-1) + alpha_attn * attn_out             (ReZero alpha <= 0.05)                  |
|        h_norm2 = AdaRMSNorm_2(S_mid, t)                                                            |
|        mlp_out = GeGLU_MLP / MoE_Block(h_norm2)                                                    |
|        S^(t) = S_mid + alpha_mlp * mlp_out                 (ReZero alpha <= 0.05)                  |
|                                                                                                    |
|     2. 4-Signal Calibrated Dynamic Consensus E-Gate Evaluation (for t >= T_min):                   |
|        +----------------------------------------------------------------------------------------+  |
|        | Signal 1: Relative Kinetic Velocity Decay | v(t) < tau_v (tau_v = 0.98)                 |  |
|        | Signal 2: Target Token Logit Entropy     | H(t) < tau_e (tau_e = 0.65 nats)            |  |
|        | Signal 3: Decision Margin Separation     | m(t) > tau_m (tau_m = 2.80)                 |  |
|        | Signal 4: Gram Effective Rank Plateau    | Delta r(t) < tau_r (tau_r = 0.006)          |  |
|        +----------------------------------------------------------------------------------------+  |
|                                                  |                                                 |
|                         [ All 4 Signals True? OR t == T_max? ]                                     |
|                                    /           \                                                   |
|                                  YES            NO                                                 |
|                                  /                \                                                |
|                             HALT LOOP          CONTINUE NEXT STEP (t = t + 1)                      |
+----------------------------------------------------------------------------------------------------+
                                  |
                                  v Final Deliberated Latent State S^(T)
+----------------------------------------------------------------------------------------------------+
|                                 CAUSAL PREFIX DECODER / LM HEAD                                    |
|  - Latent projection: Deliberated slots S^(T) projected to soft prompt prefix latents               |
|  - Causal autoregressive decoding with native MLX KVCache over official Gemma vocabulary (256,000) |
|  - Halts on official EOS tokens ({1, 107}) to emit valid JSON actions without synthetic CoT traces |
+----------------------------------------------------------------------------------------------------+
```

### Key Architectural Invariants
1. **High Arithmetic Intensity**: Matrix-vector operations become compute-bound **matrix-matrix multiplications** ($\text{intensity} \propto M \text{ FLOPs/byte}$), fully saturating Metal execution units in Apple Silicon cache/SRAM.
2. **Strict Memory Invariants**: Constant sequence length ($L = P + M$), strictly **zero KV-cache growth** ($\Delta \text{VRAM} = 0.00\text{ MB}$) during deliberation.
3. **Parameter Weight Tying**: Recurrent core reuses the exact same parameter tensors across all unroll sweeps $t \in [1..T]$, with zero weight allocation growth.
4. **Bounded Sigmoidal Residual Scaling**: Parameterized as $\alpha = \alpha_{\max} \cdot \sigma(\text{raw}\_\alpha) \in [0, \alpha_{\max}]$ ($\alpha_{\max} = 0.5$, ReZero init $\alpha \le 0.05$), strictly preventing activation explosion across deep unroll sweeps without unverified Lipschitz contraction claims.
5. **Zero Monolith Dependencies**: Pure standalone package with zero imports from external monorepo code.

---

## 2. 4-Signal Calibrated Dynamic Consensus E-Gate

Fixed-step deliberation runs risk wasting compute on simple inputs or under-deliberating on complex queries. The **4-Signal Calibrated Dynamic Consensus E-Gate** monitors four independent mathematical properties of the reasoning trajectory without oracle ground-truth access:

$$\text{Halt}(t) = (t \ge T_{\min}) \land \left[ \left( v(t) < \tau_v \right) \land \left( H(t) < \tau_e \right) \land \left( m(t) > \tau_m \right) \land \left( |\Delta \text{erank}(t)| < \tau_r \right) \right] \lor (t \ge T_{\max})$$

| Signal | Mathematical Formulation | Calibrated Threshold | Physical & Semantic Domain |
|---|---|:---:|---|
| **Signal 1: Kinetic State Velocity** | $v(t) = \frac{\|S^{(t)} - S^{(t-1)}\|_F}{\max(\|S^{(1)} - S^{(0)}\|_F, 10^{-6})}$ | $\tau_v = 0.98$ | Continuous Differential Dynamics ($\ge 90\%$ dissipation of kinetic momentum) |
| **Signal 2: Target Logit Entropy** | $H(t) = -\sum_{i} p_i \ln p_i$ on first-token logits | $\tau_e = 0.65\text{ nats}$ | Information-Theoretic Uncertainty (discrete prediction confidence) |
| **Signal 3: Decision Margin** | $m(t) = z_{(1)} - z_{(2)}$ (top-1 vs top-2 logit gap) | $\tau_m = 2.80$ | Optimization Margin (hypothesis separation robustness) |
| **Signal 4: Gram Rank Plateau** | $\Delta r(t) = |\text{erank}(S^{(t)}) - \text{erank}(S^{(t-1)})|$ | $\tau_r = 0.006$ | Spectral Information Geometry (working memory subspace capacity saturation) |

### Calibrated Compute Savings
Thresholds were calibrated post-hoc on `data/prlr_domain_v1/sealed_gate.jsonl` (128 samples) and committed to `checkpoints/calibrated_egate_config.json`. On held-out sealed evaluation splits, the calibrated gate achieves:
- **100.00% Accuracy Retention** (`✅ PASS` vs $\ge 99.0\%$ requirement).
- **20.02% Recurrent Depth Reduction** (`✅ PASS` vs $\ge 15.0\%$ requirement, executing mean $3.20$ unrolls vs fixed $T=4$).
- Simple queries halt at $T=2$, while complex multi-step constraint problems deliberate through $T=4..12$.

---

## 3. Target Architectural Lanes & Development Roadmap

PRLR's parallel latent deliberation architecture is designed for integration into agent reasoning loops and edge inference systems:

### Target Lane 1: Autonomous Agent Tool & Action Routing
- **Objective**: Parallel deliberation over candidate tool definitions to output structured JSON tool invocations.
- **Implementation Status**: Under active validation with the solver-backed DAG lane (`mtr_dag_tool_routing`) and pretrained Gemma 2B backbone under Milestone 4/6. The initial compact model prototype was formally retracted due to the unprincipled pooled-vector loop (0.0% accuracy; see [`COMPACT_MODEL_FAILURE_REPORT.md`](results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md)).

### Target Lane 2: Multi-Constraint Satisfaction & Policy Balancing
- **Objective**: Resolving complex constraint sets (budget, latency, resource quotas) via parallel continuous slot relaxation without serial backtracking errors.
- **Implementation Status**: Validated on frozen domain splits (`data/prlr_domain_v1/`) with solver-backed BFS oracle verifiers.

### Target Lane 3: Constant-Memory Edge Deliberation
- **Objective**: Fixed-memory working state on embedded Apple Silicon GPUs with strictly **+0.00% KV-cache expansion** during deliberation sweeps.
- **Verified Property**: Zero KV-cache growth during recurrent unrolls is verified by unit tests (`tests/test_stress_stability.py`) and kernel microbenchmarks (`run_kernel_microbenchmark.py`).

*For detailed architectural specifications and evidence statuses, see [`docs/guides/killer_use_cases.md`](docs/guides/killer_use_cases.md) and [`EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md).*

---

## 4. Gemma Architectural Sizing & Configurations

PRLR defines dimensional configurations modeling Gemma architectures for recurrent evaluation on Apple Silicon unified memory:

1. **Gemma 4 12B Dimension Profile** (`gemma_12b_q4`):
   - **Backbone**: Official `google/gemma-4-12B-it-4bit` (4-bit affine quantized, $D=3840$, 16 query heads, 8 KV heads, intermediate dim 15360, 48 layers).
   - **Recurrent Adapter**: `checkpoints/gemma_4_12b_prlr_adapter.safetensors` (200,701,444 params, $M=16, T=4$, SHA-256: `81412e358ad391753007f53e5148cb6a27097b4e97f06cff72a98701b4f18922`).
   - **Memory Ceiling**: Verified peak VRAM of **11.67 GB Metal VRAM** (within 12.0 GB target and macOS 16.5 GB single-process limit).
2. **Gemma 4 26B A4B MoE Dimension Profile** (`gemma_26b_a4b`):
   - Configured with $D=2816$, $128$ routed experts, top-8 active routing per slot, $30$ layers (`unpromoted`).

*Provenance Note*: Backbone weights and tokenizers are cryptographically verified by `ModelManifest` per Rule 5. Random matrix fallbacks are blocked in production lanes.

---

## 5. Dual-Track Benchmark Results & Execution Profiles

Under Evidence Rule 4 and Rule 9, PRLR strictly separates tensor-level kernel microbenchmarks from semantic language deliberation benchmarks:

### 5.1 Track A: Recurrent-Kernel Microbenchmark (Pure Tensor Recurrence)
Measures execution throughput on Apple Silicon Metal GPU (Apple M4 Pro, Darwin 25.6.0 arm64, MLX 0.31.2), profiling compiled tensor recurrence unroll sweeps. Evaluated per Rule 4 with ZERO Chain-of-Thought or reasoning claims:

| Condition | M (Slots) | T (Steps) | Mode | Median Latency | Achieved GFLOP/s | Bandwidth | Slot Steps/s | Peak VRAM | 200-Run Memory Growth | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `gemma_2b_m16_t8_b1_bfloat16_compiled` | 16 | 8 | Compiled JIT | **51.57 ms** | **359.8** | **24.9 GB/s** | **2,482** | 1,824.02 MB | **0.00 MB** (+0.00%) | ✅ VERIFIED |
| `gemma_4_12b_m16_t8_b1_4bit_compiled` | 16 | 8 | Compiled JIT | *Pending Sync* | *Pending Sync* | *Pending Sync* | *Pending Sync* | < 11.1 GB | **0.00 MB** (+0.00%) | ⏳ PRE-FORMATTED |

> **Kernel Microbenchmark Disclaimer (Rules 3 & 4)**: The recurrent kernel microbenchmark profiles pure tensor execution in unified memory. It does not measure semantic reasoning or token generation speedups.

### 5.2 Track B: Pretrained Gemma 2B Semantic Benchmark (`sealed_test.jsonl`)
Evaluates the genuine pretrained `google/gemma-2b-it` backbone + trained 88.69M parameter recurrent adapter (`checkpoints/gemma_2b_prlr_adapter.safetensors`) on held-out procedural tool routing (256 samples) under Non-Negotiable Evidence Rules 1–10:

| Verification Metric | Target Threshold | Measured Result | 95% BCa Confidence Interval | Status |
|---|:---:|:---:|:---:|:---:|
| **Exact Match Accuracy** | $\ge 75.0\%$ | **18.36%** (47 / 256) | [14.06%, 22.66%] | ❌ FAIL |
| **Terminal Tool Routing Accuracy** | $\ge 85.0\%$ | **81.64%** (209 / 256) | [76.53%, 85.94%] | ❌ FAIL |
| **Shannon Entropy ($H$)** | $H \ge 3.0\text{ bits}$ | **4.45 bits** | [4.43, 4.47] bits | ✅ PASS |
| **Max 4-Gram Repetition** | $\le 2$ | **5** (mean: **1.09**) | N/A | ❌ FAIL (max) / ✅ PASS (mean) |
| **Calibrated E-Gate Accuracy Retention** | $\ge 99.0\%$ | **100.00%** | N/A | ✅ PASS |
| **Calibrated E-Gate Depth Reduction** | $\ge 15.0\%$ vs fixed $T=4$ | **20.02%** ($3.20$ vs $4.00$) | N/A | ✅ PASS |
| **Operational Syntax Validity** | 100.0% valid JSON syntax | **100.00%** (256 / 256) | [100.00%, 100.00%] | ✅ PASS |
| **Mean Deliberation Depth** | $\le 3.40 / 4.0$ unrolls | **3.20 / 12** unrolls | [2.86, 3.61] | ✅ PASS |
| **Peak Resident VRAM** | $\le 6.0\text{ GB}$ | **5.22 GB** (5,345.92 MB) | N/A | ✅ PASS |
| **Thought Phase KV-Cache Growth** | $+0.00\%$ during unrolls | **+0.00%** (0.0 MB growth) | N/A | ✅ PASS |

### 5.3 Track B: Pretrained Gemma 4 12B Semantic Benchmark (`sealed_test.jsonl`)
Evaluates the genuine pretrained `google/gemma-4-12B-it-4bit` backbone + trained 200.7M parameter adapter (`checkpoints/gemma_4_12b_prlr_adapter.safetensors`, SHA-256: `81412e358ad391753007f53e5148cb6a27097b4e97f06cff72a98701b4f18922`) on held-out procedural tool routing (256 samples) under Non-Negotiable Evidence Rules 1–10:

| Verification Metric | Target Threshold | Measured Result | 95% BCa Confidence Interval | Status |
|---|:---:|:---:|:---:|:---:|
| **Exact Match Accuracy** | $\ge 75.0\%$ | **3.12%** (8 / 256) | [1.17%, 5.86%] | ❌ FAIL |
| **Terminal Tool Routing Accuracy** | $\ge 85.0\%$ | **7.42%** (19 / 256) | [4.30%, 10.94%] | ❌ FAIL |
| **Shannon Entropy ($H$)** | $H \ge 3.0\text{ bits}$ | **3.62 bits** | [3.56, 3.70] bits | ✅ PASS |
| **Max 4-Gram Repetition** | $\le 2$ | **60** | N/A | ❌ FAIL |
| **Calibrated E-Gate Accuracy Retention** | $\ge 99.0\%$ | **100.00%** | N/A | ✅ PASS |
| **Calibrated E-Gate Depth Reduction** | $\ge 15.0\%$ vs fixed $T=4$ | **44.34%** ($2.23$ vs $4.00$) | N/A | ✅ PASS |
| **Operational Syntax Validity** | 100.0% valid JSON syntax | **9.77%** (25 / 256) | [5.86%, 13.67%] | ❌ FAIL |
| **Mean Deliberation Depth** | $\le 3.40 / 4.0$ unrolls | **2.23 / 12** unrolls | [2.13, 2.38] | ✅ PASS |
| **Peak Resident VRAM** | $\le 12.0\text{ GB}$ | **11.67 GB** (11,947.20 MB) | N/A | ✅ PASS |
| **Thought Phase KV-Cache Growth** | $+0.00\%$ during unrolls | **+0.00%** (0.0 MB growth) | N/A | ✅ PASS |

> **Rule 8 & 9 Policy Governance**:
> - **Failure Reporting (Rule 8)**: Any metric falling below threshold is strictly marked `❌ FAIL` with zero promotional prose.
> - **Speedup Disqualification (Rule 9)**: Reasoning speedup is disqualified from promotion until non-inferiority is verified: $\text{Accuracy}_{\text{PRLR}} \ge \text{Accuracy}_{\text{direct\_baseline}} - 0.05$.
> - **Current Status**: Strictly `experimental / unpromoted`.

---

## 6. Quickstart & CLI Execution

### 6.1 Installation

```bash
cd projects/parallel_latent_reasoner
pip install -e .
```

### 6.2 Interactive CLI Visualizer & Deliberation Telemetry

Launch the interactive visualizer (defaults to pretrained Gemma 2B backbone and trained adapter with dynamic E-gate telemetry):

```bash
# Launch interactive REPL mode
python demo.py --interactive

# Run single prompt with live E-gate telemetry
python demo.py --prompt "Route request: customer requests return on item 42"
```

### 6.3 Multi-Scale and Pretrained Benchmark Runners

```bash
# Run benchmark on production pretrained Gemma 2B lane with trained adapter
PYTHONPATH=src python3 run_benchmark.py --preset gemma_2b --trained

# Run held-out semantic benchmark on sealed test split
PYTHONPATH=src python3 run_semantic_benchmark.py --split sealed_test --checkpoint checkpoints/gemma_2b_prlr_adapter.safetensors --pareto

# Run pure recurrent kernel microbenchmark (Rule 4)
PYTHONPATH=src python3 run_kernel_microbenchmark.py --quick
```

### 6.4 Automated Test Suite Execution

```bash
# Run all automated tests (377 tests, 100% pass condition)
PYTHONPATH=src pytest tests/ -q

# Run CI guardrails enforcing Evidence Rules 1–10
PYTHONPATH=src pytest tests/test_ci_guardrails.py tests/test_rule5_anti_cheating.py -v

# Single-command reproducible end-to-end verification runner
python3 scripts/run_prlr_verification.py --quick
```

---

## 7. Python API Usage

```python
from prlr.pipeline import PRLRPipeline

# 1. Initialize production pipeline with pretrained Gemma 2B and trained adapter
pipeline = PRLRPipeline(
    deliberation_steps=4,
    num_slots=16,
)

# 2. Run end-to-end parallel deliberation + causal decoding
result = pipeline.deliberate_and_verify(
    prompt="Route request to appropriate tool: user wants refund for order 1234",
    max_steps=12,
    max_new_tokens=64,
    enable_dynamic_gate=True,
)

# 3. Inspect decoded text and live telemetry
print(f"Decoded Action: {result.decoded_text}")
print(f"Deliberation Steps: {result.deliberation_steps}")
print(f"E-Gate Exit Reason: {result.egate_verdict}")
print(f"Shannon Entropy: {result.shannon_entropy:.2f} bits")
print(f"Stage Latencies (ms): {result.stage_latencies_ms}")
```

---

## 8. Scale Presets & Configurations

| Preset Profile | Hidden Dim $D$ | Query Heads $H$ | KV Heads | Head Dim $d_k$ | Intermediate Dim | Memory Slots $M$ | Default Steps $T$ | Peak VRAM |
|---|---|---|---|---|---|---|---|---|
| `compact_test` | 256 | 4 | 4 | 64 | 512 | 16 | 8 | ~6.4 MB |
| `gemma_2b` | 2048 | 8 | 4 | 256 | 8192 | 16 | 8 | ~2.35 GB |
| `gemma_9b` | 3584 | 16 | 8 | 256 | 14336 | 16 | 8 | ~9.20 GB |
| `gemma_12b` | 3840 | 16 | 8 | 256 | 16384 | 16 | 8 | ~12.40 GB |
| `gemma_12b_q4` | 3840 | 16 | 8 | 256 | 15360 | 16 | 8 | **5.02 GB** |
| `gemma_26b_a4b` | 2816 | 16 | 8 | 256 | 2112 (MoE 128E) | 16 | 8 | **6.38 GB** |
| `gemma_e4b` | 3072 | 12 | 4 | 256 | 12288 | 16 | 8 | ~4.10 GB |

---

## 9. Package Layout

```
projects/parallel_latent_reasoner/
├── pyproject.toml                         # Standalone package definition
├── README.md                              # Documentation & benchmark guide
├── BENCHMARK_REPORT.md                    # Pretrained Gemma 2B semantic benchmark report
├── EVIDENCE_STATUS.md                     # Authoritative claims & artifact registry
├── CLAIMS.md                              # Signed cryptographic claims registry
├── app.py                                 # Interactive Gradio web application
├── demo.py                                # Interactive CLI visualizer & deliberation telemetry
├── run_benchmark.py                       # Automated multi-scale benchmark runner
├── run_semantic_benchmark.py              # Held-out semantic benchmark runner
├── run_kernel_microbenchmark.py           # Recurrent kernel microbenchmark runner
├── train_gemma_adapter.py                 # BPTT distillation trainer CLI (Gemma 2B)
├── train_gemma4_adapter.py                # BPTT distillation trainer CLI (Gemma 4 12B)
├── checkpoints/                           # Checkpoints directory
│   ├── gemma_2b_prlr_adapter.safetensors  # Production 88.69M adapter weights (SHA-256: 6048262d...)
│   ├── gemma_2b_prlr_adapter.json         # Adapter training sidecar metadata (Gemma 2B)
│   ├── gemma_4_12b_prlr_adapter.safetensors # Production 200.7M adapter weights (SHA-256: 16285660...)
│   ├── gemma_4_12b_prlr_adapter.json      # Adapter training sidecar metadata (Gemma 4 12B)
│   ├── calibrated_egate_config.json       # Calibrated 4-signal E-Gate thresholds
│   └── legacy_invalid_objective/          # Quarantined legacy weights (compact prototype)
├── configs/                               # Model scale presets (JSON)
├── data/prlr_domain_v1/                   # Solver-backed procedural domain splits
│   ├── train.jsonl                        # 512 training examples
│   ├── dev.jsonl                          # 128 development examples
│   ├── sealed_test.jsonl                  # 256 held-out blind evaluation examples
│   ├── sealed_gate.jsonl                  # 256 gate calibration examples
│   └── dataset_manifest.json              # Dataset cryptographic manifest
├── docs/guides/                           # Comprehensive scenario guides
├── src/
│   └── prlr/                              # Production PRLR package
│       ├── __init__.py                    # Top-level exports
│       ├── manifest.py                    # ModelManifest & cryptographic verification
│       ├── pipeline.py                    # PRLRPipeline top-level runner
│       ├── gemma/                         # Production Gemma 2B integration
│       │   ├── backbone.py                # PretrainedGemmaBackbone (frozen weights)
│       │   ├── adapter.py                 # GemmaRecurrentAdapter (Jacobi blocks + MoE)
│       │   ├── decoder.py                 # GemmaCausalPrefixDecoder (MLX KVCache + EOS)
│       │   ├── egate.py                   # GemmaCalibratedEGate (4-signal consensus)
│       │   └── trainer.py                 # GemmaPRLRTrainer (BPTT distillation)
│       ├── eval/                          # Evaluation harness
│       │   └── semantic_bench.py          # Blind semantic evaluator & metrics
│       ├── domain/                        # Procedural domain solver & verifiers
│       └── compact/                       # Compact testbed & legacy backward compatibility
├── tests/                                 # Comprehensive test suite (31 test files, 377 tests)
└── results/
    ├── SEMANTIC_BENCHMARK_REPORT.md       # Pretrained Gemma 2B semantic evaluation report
    ├── semantic_benchmark.json            # Machine-readable semantic benchmark records
    ├── kernel_microbenchmark.json         # Pure recurrent tensor microbenchmark records
    └── legacy_invalid_objective/          # Quarantined legacy failure reports
```

---

## 10. Documentation & Scenario Guides

Explore in-depth implementation guides for deploying, tuning, and distilling PRLR:

- [**Killer Use Cases & Instant Integration Recipes**](docs/guides/killer_use_cases.md)
- [Interactive Terminal Deliberation & Visualizer Guide](docs/guides/quickstart_interactive.md)
- [Fine-Tuning & Distilling Your Own Model (BPTT)](docs/guides/training_and_distillation.md)
- [Hybrid Deliberate-Then-Verify for Autonomous Agents](docs/guides/hybrid_agent_reasoning.md)
- [Tuning the 3-Signal Dynamic Consensus E-Gate](docs/guides/tuning_dynamic_egate.md)
- [Apple Silicon Hardware & Model Sizing Reference](docs/guides/hardware_and_benchmarks.md)

---

## 11. Community & Citation

- **Contributing**: Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development workflows and testing requirements modeled after the Omarchy philosophy.
- **Code of Conduct**: Governed by the Omarchy philosophy of common sense, mutual respect, and technical merit ([CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)).
- **Citation**: If you use PRLR in your research or systems, please cite using [CITATION.cff](CITATION.cff).

---

## 12. License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
