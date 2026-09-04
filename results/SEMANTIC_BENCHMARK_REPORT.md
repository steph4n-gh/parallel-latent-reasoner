# Pretrained google/gemma-4-12B-it-4bit Semantic Benchmark Report

> ⚠️ **DISCLAIMER (Non-Negotiable Evidence Rules 1, 2, 5, 8, 9, 10)**:  
> *PRETRAINED GEMMA SEMANTIC BENCHMARK: Evaluates genuine pretrained Gemma backbone + recurrent deliberation adapter on frozen solver-backed domain splits. Operates under strict Rule 1 (blind evaluation) and Rule 2 (post-hoc verification).*

---

## 1. Execution & Model Provenance

- **Model**: `google/gemma-4-12B-it-4bit`
- **Weight SHA-256**: `{'model-00001-of-00002.safetenso`
- **Dataset Split**: `sealed_test` (256 samples)
- **Hardware**: `Apple M4 Pro` (24.0 GB RAM)
- **Git Commit**: `a90ad7ecebdd7a2f7c9d7d5a84227bd5bc729732` (Dirty: `True`)
- **Timestamp**: `2026-09-04T03:07:14.833520+00:00`

---

## 2. Benchmark Summary Metrics (1,000-Resample Bootstrap 95% BCa CI)

| Metric | Value | 95% BCa Confidence Interval | Target Threshold | Status |
|---|:---:|:---:|:---:|:---:|
| **Exact Match Accuracy** | 3.12% | [1.17%, 5.86%] | >= 75.0% | ❌ FAIL |
| **Terminal Tool Routing Accuracy** | 7.42% | [4.30%, 10.94%] | >= 85.0% | ❌ FAIL |
| **Shannon Entropy (H)** | 3.62 bits | [3.56, 3.70] bits | >= 3.0 bits | ✅ PASS |
| **Max 4-Gram Repetition** | 60 | N/A | <= 2 | ❌ FAIL |
| **Calibrated E-Gate Accuracy Retention** | 100.00% | N/A | >= 99.0% | ✅ PASS |
| **Calibrated E-Gate Depth Reduction** | 44.34% | N/A | >= 15.0% vs fixed T=4 | ✅ PASS |
| **Operational Validity** | 9.77% | [5.86%, 13.67%] | N/A | Evaluated |
| **Mean Deliberation Depth** | 2.23 / 12 | [2.13, 2.38] | <= 3.40 / 4.0 | ✅ PASS |

---

## 3. Stage-by-Stage Latency Decomposition (ms)

| Stage | Mean (ms) | Median (p50) | p95 | 95% BCa CI (ms) | Fraction of Total |
|---|:---:|:---:|:---:|:---:|:---:|
| **Prefill** | 1058.05 ms | 1011.07 ms | 1514.04 ms | [1033.69, 1088.23] | 15.4% |
| **Prelude** | 5.44 ms | 3.30 ms | 12.19 ms | [4.89, 6.37] | 0.1% |
| **Deliberation** | 2532.99 ms | 2277.42 ms | 4679.65 ms | [2420.15, 2708.63] | 36.8% |
| **Decode** | 3279.06 ms | 3283.22 ms | 3472.59 ms | [3250.88, 3299.66] | 47.7% |
| **Total** | 6875.54 ms | 6616.40 ms | 9172.49 ms | [6747.57, 7068.87] | 100.0% |

---

## 4. Empirical Pareto Curves

### 4.1 Fixed Depth Progression (T in {0, 1, 2, 4, 8, 12})

| Recurrence Depth T | Exact Match | 95% CI | Deliberation Latency (ms) | Total Latency (ms) |
|:---:|:---:|:---:|:---:|:---:|
| T=0 | 0.0% | [0.0%, 0.0%] | 0.00 ms | 4618.16 ms |
| T=1 | 0.0% | [0.0%, 0.0%] | 118.46 ms | 4635.27 ms |
| T=2 | 3.1% | [0.0%, 9.4%] | 60.58 ms | 4635.43 ms |
| T=4 | 0.0% | [0.0%, 0.0%] | 63.83 ms | 4522.79 ms |
| T=8 | 0.0% | [0.0%, 0.0%] | 86.32 ms | 4573.26 ms |
| T=12 | 6.2% | [0.0%, 15.6%] | 109.69 ms | 4571.12 ms |

### 4.2 Calibrated Dynamic E-Gate Frontier (lambda in [0.25, 2.0])

| Sensitivity lambda | Mean Executed Depth | Depth Reduction | Exact Match | Deliberation Latency (ms) | Total Latency (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| lambda=0.25 | 12.00 / 12 | -200.0% | 6.2% | 12889.09 ms | 17307.77 ms |
| lambda=0.5 | 7.34 / 12 | -83.6% | 12.5% | 8008.22 ms | 12265.27 ms |
| lambda=0.75 | 4.56 / 12 | -14.1% | 3.1% | 4803.18 ms | 8887.59 ms |
| lambda=1.0 | 2.06 / 12 | 48.4% | 3.1% | 2185.03 ms | 6231.92 ms |
| lambda=1.5 | 2.00 / 12 | 50.0% | 3.1% | 2088.45 ms | 6129.16 ms |
| lambda=2.0 | 2.00 / 12 | 50.0% | 3.1% | 2131.93 ms | 6192.53 ms |

---

## 5. Non-Negotiable Evidence Attestation
- **Rule 1 (Blind Evaluation)**: Programmatically verified that inference functions received zero ground-truth keys.
- **Rule 2 (Post-Hoc Verification)**: Output predictions were sealed prior to scoring against answer keys.
- **Rule 5 (Verified Model Weights)**: Loaded verified weights from official google/gemma-4-12B-it-4bit repository.
- **Rule 8 (Conditional Prose)**: Metric outcomes reported truthfully without affirmative bias.
- **Rule 9 (Speedup & Non-Inferiority)**: Latent deliberation paired with calibrated accuracy retention.
- **Rule 10 (Cryptographic Provenance)**: Machine-readable artifact records commit SHA, hashes, and raw prediction records.

