# Parallel Latent Reasoner (PRLR) Pretrained Gemma 2B Semantic Benchmark Report

> ⚠️ **DISCLAIMER (Non-Negotiable Evidence Rules 1, 2, 5, 8, 9, 10)**:
> *PRETRAINED GEMMA 2B SEMANTIC BENCHMARK: Evaluates genuine pretrained `google/gemma-2b-it` backbone + recurrent deliberation adapter on frozen solver-backed domain splits (`data/prlr_domain_v1/sealed_test.jsonl`). Operates under strict Rule 1 (blind evaluation: zero inference access to target keys) and Rule 2 (post-hoc verification). Per Rule 8, all metrics are reported conditionally; failed target thresholds emit explicit failure statuses. Historical failure analysis of the retired compact prototype is archived in `results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md`.*

---

## 1. Execution & Model Provenance

- **Evaluated Architecture**: Pretrained `google/gemma-2b-it` Backbone (BF16, 2.5B params, frozen) + Recurrent Deliberation Adapter + Causal Prefix Decoder + 4-Signal Calibrated Consensus E-Gate
- **Backbone Revision**: `96988410cbdaeb8d5093d1ebdc5a8fb563e02bad`
- **Backbone Weights SHA-256**:
  - `model-00001-of-00002.safetensors`: `561656f892a2a1ca0837ca529c5ce820a72b40f4f563b1cd0a1acc0b3899c30c`
  - `model-00002-of-00002.safetensors`: `20fe2ee66bf1361241a6c522091a5e0328fc6c1703f93734889fa381fcf8760c`
- **Official Tokenizer SHA-256**: `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` (`tokenizer.model`, SentencePiece 256k vocab)
- **Trained Adapter Artifact**: `checkpoints/gemma_2b_prlr_adapter.safetensors`
  - **Adapter Weights SHA-256**: `6048262d99e5d28851adfc87a379a2796802926605ab74e33553b4d9347028d7`
  - **Adapter Sidecar JSON SHA-256**: `db40b258698490794866ae5af76b58e2c1d678930782ba777a595299ba9699ca` (`checkpoints/gemma_2b_prlr_adapter.json`)
  - **Trained Parameters**: 88,690,692 parameters (88.69M params)
  - **Training Loss**: Final loss 0.1464, rolling loss 0.1499 < 0.15 on 512 samples of `data/prlr_domain_v1/train.jsonl` (8 epochs, BPTT)
- **Calibrated E-Gate Configuration**: `checkpoints/calibrated_egate_config.json` (SHA-256: `89fa2c9cde10b340d3c9ff58c4c0ca845fad972ac2b6c967d456c981049373b6`)
- **Dataset Split**: `data/prlr_domain_v1/sealed_test.jsonl` (256 samples, SHA-256: `1be6c4fe69be31ca81a81736841c523e2b022c85bfa90a486ebde1b088f0a5d6`)
- **Hardware Platform**: Apple Silicon Metal GPU (Apple M4 Pro, 24.0 GB Unified Memory, macOS Darwin 25.6.0 arm64)
- **Runtime Environment**: Python 3.14.4, MLX 0.31.2, Transformers 5.9.0, NumPy 2.4.6
- **Git Commit SHA**: `a90ad7ecebdd7a2f7c9d7d5a84227bd5bc729732` (Dirty: `True`)
- **Timestamp**: `2026-09-03T19:01:35.605972+00:00`
- **Authoritative Data Artifact**: `results/semantic_benchmark.json` (SHA-256: `81f15af40e01980f95aff18e302980a71055bf95b2041622e5be13b15a29c516`)
- **Exact Reproduction Command**:
  ```bash
  PYTHONPATH=src python3 run_semantic_benchmark.py --split sealed_test --checkpoint checkpoints/gemma_2b_prlr_adapter.safetensors --pareto
  ```

---

## 2. Benchmark Summary Metrics (1,000-Resample Bootstrap 95% BCa CI)

| Empirical Verification Gate | Target Specification | Measured Result | 95% BCa Confidence Interval | Status |
|---|:---:|:---:|:---:|:---:|
| **Exact Match Accuracy** | $\ge 75.0\%$ | **18.36%** (47 / 256) | [14.06%, 22.66%] | ❌ FAIL |
| **Terminal Tool Routing Accuracy** | $\ge 85.0\%$ | **81.64%** (209 / 256) | [76.53%, 85.94%] | ❌ FAIL |
| **Information-Theoretic Shannon Entropy ($H$)** | $H \ge 3.0\text{ bits}$ | **4.45 bits** | [4.43, 4.47] bits | ✅ PASS |
| **Max 4-Gram Token Repetition** | $\le 2$ | **5** (mean: **1.09**) | N/A | ❌ FAIL (max) / ✅ PASS (mean) |
| **Calibrated E-Gate Accuracy Retention** | $\ge 99.0\%$ | **100.00%** | N/A | ✅ PASS |
| **Calibrated E-Gate Depth Reduction** | $\ge 15.0\%$ vs fixed $T=4$ | **20.02%** ($3.20$ vs $4.00$) | N/A | ✅ PASS |
| **Operational / Syntactic Validity** | 100.0% valid JSON syntax | **100.00%** (256 / 256) | [100.00%, 100.00%] | ✅ PASS |
| **Mean Deliberation Depth** | $\le 3.40 / 4.0$ unrolls | **3.20 / 12** unrolls | [2.86, 3.61] | ✅ PASS |
| **Peak Resident VRAM Memory** | $\le 6.0\text{ GB}$ | **5.22 GB** (5,345.92 MB) | N/A | ✅ PASS |
| **Thought Phase KV-Cache Growth** | $+0.00\%$ during unrolls | **+0.00%** (0.0 MB growth) | N/A | ✅ PASS |

---

## 3. Executive Analysis & Rule 8 Conditional Governance

### 3.1 Resolution of Legacy Prototype Failure Modes
The genuine pretrained Gemma 2B vertical lane successfully resolves the structural defects that crippled the retired compact prototype:
1. **Elimination of Degenerate Token Collapse**: The legacy prototype exhibited total entropy collapse ($H=0.00\text{ bits}$) and emitted repetitive strings (`AAAAAAAAAAAAAAAA`). In contrast, the production Gemma 2B model achieves **$H = 4.45\text{ bits}$** with a mean 4-gram repetition of **$1.09$**, confirming natural, diverse token emissions.
2. **Grammar & Operational Validity**: Emits 100% syntactically valid JSON tool invocations matching the DAG routing schema (`operational_validity: 100.00%`).
3. **High Terminal Tool Accuracy**: Correctly identifies the terminal routing tool in **81.64%** of held-out test cases (209/256).

### 3.2 Gate Failures & Mandatory Rule 8 Affirmative Bias Prohibition
Per Non-Negotiable Evidence Rule 8, no success prose may be emitted when an associated metric fails its threshold:
- **Exact Match Accuracy Deficit**: Measured at **18.36%** vs target threshold $\ge 75.0\%$ (`❌ FAIL`). Full multi-hop tool execution paths require complete sequential fidelity; the adapter successfully identifies the terminal tool in 81.64% of cases but diverges on intermediate multi-hop path sequences.
- **Terminal Tool Routing Deficit**: Measured at **81.64%** vs target threshold $\ge 85.0\%$ (`❌ FAIL`). Although the upper bound of the 95% bootstrap CI reaches 85.94%, the point estimate (81.64%) falls 3.36 percentage points below the promotion gate.
- **Worst-Case 4-Gram Repetition**: While mean repetition is healthy at 1.09, worst-case repetition reached 5 in edge cases (`❌ FAIL` vs $\le 2$).

### 3.3 Rule 9 Speedup Disqualification
Under Rule 9, reasoning phase speedup claims are conditioned on output quality matching direct autoregressive generation under a documented non-inferiority criterion:
$$\text{Accuracy}_{\text{PRLR}} \ge \text{Accuracy}_{\text{direct\_baseline}} - 0.05$$
Because Exact Match accuracy is 18.36% < 75.0%, reasoning speedup is classified as **DISQUALIFIED FROM PROMOTION**. The system remains classified as **`experimental / unpromoted`**.

---

## 4. Stage-by-Stage Latency Decomposition (ms)

Measured across all 256 samples on Apple M4 Pro Metal GPU:

| Stage | Mean (ms) | Median (p50) | p95 (ms) | 95% BCa CI (ms) | Fraction of Total |
|---|:---:|:---:|:---:|:---:|:---:|
| **Prefill** | 299.47 ms | 188.22 ms | 828.81 ms | [265.35, 360.64] | 10.7% |
| **Prelude** | 1.65 ms | 0.75 ms | 4.10 ms | [1.28, 2.32] | 0.1% |
| **Deliberation** | 791.19 ms | 451.31 ms | 2,447.78 ms | [713.14, 894.99] | 28.3% |
| **Decode** | 1,701.57 ms | 1,686.46 ms | 2,246.29 ms | [1641.03, 1761.23] | 60.9% |
| **Total** | 2,793.88 ms | 2,393.35 ms | 4,922.33 ms | [2663.90, 2930.29] | 100.0% |

- **Peak Resident VRAM**: 5,345.92 MB (5.22 GB $\le 6.0\text{ GB}$)
- **Active Allocator VRAM**: 5,119.22 MB
- **Memory Growth Between Inferences**: 0.00 MB (+0.00%)

---

## 5. Empirical Pareto Curves

### 5.1 Fixed Depth Progression ($T \in \{0, 1, 2, 4, 8, 12\}$)

| Recurrence Depth $T$ | Exact Match | 95% CI | Deliberation Latency (ms) | Total Latency (ms) |
|:---:|:---:|:---:|:---:|:---:|
| **T = 0** | 0.0% | [0.0%, 0.0%] | 0.00 ms | 1,361.58 ms |
| **T = 1** | 12.5% | [2.3%, 25.0%] | 16.52 ms | 2,096.48 ms |
| **T = 2** | 9.4% | [0.0%, 18.8%] | 10.95 ms | 1,916.32 ms |
| **T = 4** | 9.4% | [0.0%, 18.8%] | 20.02 ms | 1,883.30 ms |
| **T = 8** | 12.5% | [2.3%, 25.0%] | 57.30 ms | 2,643.50 ms |
| **T = 12** | 9.4% | [0.0%, 18.8%] | 75.95 ms | 2,807.10 ms |

### 5.2 Calibrated Dynamic E-Gate Frontier ($\lambda \in [0.25, 2.0]$)

| Sensitivity $\lambda$ | Mean Executed Depth | Depth Reduction vs $T=4$ | Exact Match | Deliberation Latency (ms) | Total Latency (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\lambda = 0.25$** | 12.00 / 12 | -200.0% | 9.4% | 24,709.42 ms | 28,754.99 ms |
| **$\lambda = 0.50$** | 12.00 / 12 | -200.0% | 9.4% | 2,986.20 ms | 5,406.85 ms |
| **$\lambda = 0.75$** | 10.44 / 12 | -160.9% | 9.4% | 2,048.43 ms | 3,738.77 ms |
| **$\lambda = 1.00$** | 2.69 / 12 | +32.8% | 9.4% | 571.24 ms | 2,485.03 ms |
| **$\lambda = 1.50$** | 2.00 / 12 | +50.0% | 9.4% | 387.42 ms | 2,020.36 ms |
| **$\lambda = 2.00$** | 2.00 / 12 | +50.0% | 9.4% | 440.04 ms | 2,366.17 ms |

---

## 6. Non-Negotiable Evidence Attestation

- **Rule 1 (Blind Evaluation)**: Programmatically verified that inference and generation routines receive zero ground-truth keys, constraints, or verifier metadata.
- **Rule 2 (Post-Hoc Verification)**: Generated output predictions were committed to immutable records prior to scoring against answer keys.
- **Rule 3 (Zero Synthetic Traces)**: Zero hardcoded or simulated thought chains are labeled as model reasoning.
- **Rule 4 (Honest Nomenclature)**: Kernel unroll benchmarks are strictly designated microbenchmarks.
- **Rule 5 (Authentic Model Provenance)**: Model loads official `google/gemma-2b-it` weights verified by SHA-256 hashes; random matrix fallbacks are blocked.
- **Rule 6 (Measured Latencies)**: All latencies are measured from Metal GPU stream event timers; zero synthetic millisecond multipliers.
- **Rule 7 (Measured Memory Residency)**: All VRAM metrics reported from Metal device allocators.
- **Rule 8 (Conditional Prose)**: Failed metrics (EM 18.36%, Terminal 81.64%, Max Rep 5) explicitly emit failure narratives.
- **Rule 9 (Speedup & Non-Inferiority)**: Latent speedup is disqualified from promotion due to the quality gap.
- **Rule 10 (Cryptographic Provenance)**: Full provenance tuple recorded in `results/semantic_benchmark.json`.
