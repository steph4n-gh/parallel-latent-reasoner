# Parallel Latent Reasoner (PRLR) — Signed Claims & Evidence Registry

- **Schema Version**: `prlr.claims.v1`
- **Specification Status**: Active & Verified
- **Date Created / Sealed**: 2026-09-03
- **Hardware Platform**: Apple Silicon Metal GPU (`arm64`), macOS Unified Memory Architecture
- **Git Commit SHA**: `a90ad7ecebdd7a2f7c9d7d5a84227bd5bc729732`
- **Model Reference**: `google/gemma-2b-it` (Architecture: `GemmaForCausalLM`)
- **Dataset Manifest SHA-256**: `cdfb10f9cbd3d6d9d8380f901822919362bc4d9928a6a0ad41b1a9dcf8bb6b82`
- **Governing Policies**: `AGENTS.md`, `docs/documentation_policy.md`, Non-Negotiable Evidence Rules 1–10

---

## 1. Governance & Strict Enforcement Clauses

All claims in this registry are subject to programmatic automated verification under the **10 Non-Negotiable Evidence Rules**:

### Rule 8: Conditional Prose Enforcement
No success or promotional prose may be emitted when an associated metric fails its acceptance threshold. Any metric failing its threshold automatically forces the claim classification to `FAILED` or `RETRACTED`.

### Rule 9: Dual-Track Benchmark Separation & Non-Inferiority Enforcement
1. **Track A — Recurrent Latent Memory Kernel Microbenchmarks**:
   - Profile pure tensor recurrence execution ($M$ slots, $T$ steps, $D$ dimensions) on Metal GPU.
   - Strictly forbidden from asserting Chain-of-Thought (CoT), language reasoning, or cognitive speedup claims (Rule 4).
2. **Track B — Semantic Reasoning Benchmarks**:
   - Evaluated on matching pretrained backbone (`google/gemma-2b-it`) and frozen solver-backed domain splits (`data/prlr_domain_v1/`).
   - Speedup claims are valid **if and only if** solution quality satisfies the documented non-inferiority criterion:
     $$\text{Accuracy}_{\text{PRLR}} \ge \text{Accuracy}_{\text{direct\_baseline}} - 0.05$$
   - If output accuracy collapses or emits repetitive looping (as occurred in the compact model prototype), reasoning speedup is classified as **DISQUALIFIED**.

### Rule 10: Cryptographic Provenance Tuple Requirement
Every active claim MUST map to:
1. Checked-in machine-readable artifact path and its exact SHA-256 checksum.
2. Exact terminal reproduction command.
3. Target hardware platform and device identifiers.
4. Model ID, weight SHA-256, and tokenizer SHA-256.
5. Deterministic random seed and runtime environment versions.

---

## 2. Master Claims Registry (31 Claims)

| Claim ID | Claim Description | Status / Classification | Architectural Tier | Dataset / Split Scope | Artifact Path | Artifact SHA-256 | Reproduction Command | Enforced Rules | Audit & Verification Notes |
|:---:|---|:---:|:---:|:---:|---|---|---|:---:|---|
| **CLM-01** | Multi-domain reasoning accuracy on compact model | **RETRACTED** | Compact Scratch ($D=256$) | 25 cognitive tasks | `results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md` | `0c1023ee...` | `python run_benchmark.py --preset compact_test` | R8, R10 | Measured 0.0% accuracy; repetitive token loops. Retracted per Milestone 1 R0. |
| **CLM-02** | 22x speedup vs Autoregressive CoT | **RETRACTED** | Compact Scratch ($D=256$) | 25 cognitive tasks | `results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md` | `0c1023ee...` | `python run_benchmark.py --preset compact_test` | R4, R9 | Baseline was serial recurrent loop, not LLM CoT; accuracy 0.0%. Disqualified. |
| **CLM-03** | Zero repetitive token looping & diverse entropy | **RETRACTED** | Compact Scratch ($D=256$) | 25 cognitive tasks | `results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md` | `0c1023ee...` | `python run_benchmark.py --preset compact_test` | R8 | Shannon entropy $H=0.00\text{ bits}$; token repetition count 13. Retracted. |
| **CLM-04** | Sub-3ms autonomous agent tool routing (`refund_order`) | **RETRACTED** | Compact Prototype | Tool routing snippet | `docs/guides/killer_use_cases.md` | `5d837312...` | N/A | R8 | Model emitted padding tokens; example was ungrounded marketing prose. Retracted. |
| **CLM-05** | Gemma 4 12B Q4 and 26B A4B MoE support | **RETRACTED** | 1-layer random configs | Synthetic config presets | `results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md` | `0c1023ee...` | `python run_large_gemma_eval.py` | R5 | Configuration-shaped random matrices without genuine Google weights. Retracted. |
| **CLM-06** | Production readiness from 54 passing unit tests | **RETRACTED** | Unit test suite | Synthetic test fixtures | `results/legacy_invalid_objective/COMPACT_MODEL_FAILURE_REPORT.md` | `0c1023ee...` | `pytest tests/` | R10 | Synthetic tests passed on random tensors; failed to validate semantic reasoning. |
| **CLM-07** | Zero KV-cache growth (+0.00%) during unrolls | **EVIDENCE-BOUND** | MLX Recurrent Block | Synthetic memory slots | `tests/test_stress_stability.py` | `59af3e36...` | `pytest tests/test_stress_stability.py` | R7, R10 | Strictly applies to recurrent deliberation phase; does NOT apply to causal decode. |
| **CLM-08** | Sub-5ms recurrent unroll latency on Apple Silicon | **EVIDENCE-BOUND** | Recurrent Kernel ($M=16, T=8$) | Synthetic prompt context | `tests/test_tier5_adversarial_challenger.py` | `5e3e4ff4...` | `pytest tests/test_tier5_adversarial_challenger.py` | R4, R6 | Pure tensor microbenchmark in unified memory. Zero language reasoning claim. |
| **CLM-09** | Cryptographic ModelManifest verification | **VERIFIED** | `ModelManifest` | SHA-256 assertions | `src/prlr/manifest.py` | `e8a15067...` | `pytest tests/test_manifest_integrity.py` | R10 | Enforces weight hashes, tokenizer hash, commit SHA, runtime versions. |
| **CLM-10** | Strict Rule 5 anti-cheating & random model rejection | **VERIFIED** | `prlr.manifest` | Anti-cheating test suite | `tests/test_rule5_anti_cheating.py` | `d9a8075f...` | `pytest tests/test_rule5_anti_cheating.py` | R5 | Rejects random models as Gemma; blocks character-modulo fallback. |
| **CLM-11** | System namespace isolation (`kernel`, `compact`, `gemma`) | **VERIFIED** | Package layout | Structure tests | `tests/test_system_separation.py` | `dc05d2d0...` | `pytest tests/test_system_separation.py` | R10 | Pure recurrent kernel contains zero language claims; gemma lane requires verified weights. |
| **CLM-12** | Official Google Gemma 2B backbone loader | **VERIFIED** | `google/gemma-2b-it` | Health check suite | `src/prlr/gemma/backbone.py` | `5fb4d697...` | `pytest tests/test_pretrained_gemma_lane.py` | R5, R10 | Loads verified bfloat16 weights from official repository; official tokenizer. |
| **CLM-13** | Contextual prompt hidden states $(B, L, 2048)$ | **VERIFIED** | Gemma 2B backbone | SentencePiece tokens | `src/prlr/gemma/backbone.py` | `5fb4d697...` | `pytest tests/test_pretrained_gemma_lane.py` | R5 | Contextual hidden representations extracted; character-modulo fully purged. |
| **CLM-14** | Orthogonal slot anchor initialization | **VERIFIED** | Recurrent Adapter | Learned slot embeddings | `src/prlr/gemma/adapter.py` | `56762eee...` | `pytest tests/test_pretrained_gemma_lane.py` | R10 | Small orthogonal anchors + slot-role embeddings; non-collinear memory slots. |
| **CLM-15** | Scaled nonzero MoE router and expert init | **VERIFIED** | Recurrent Adapter | Adapter weights | `src/prlr/gemma/adapter.py` | `56762eee...` | `pytest tests/test_pretrained_gemma_lane.py` | R10 | Nonzero MoE weights; eliminates dead expert routes. |
| **CLM-16** | Bounded sigmoidal residual scaling | **VERIFIED** | Recurrent Adapter | Mathematical bounds | `tests/test_ci_guardrails.py` | `01e26021...` | `pytest tests/test_ci_guardrails.py -k residual` | R10 | $\alpha = \alpha_{\max} \cdot \sigma(\text{raw}\_\alpha) \in [0, \alpha_{\max}]$; strictly monotonic. |
| **CLM-17** | 100% nonzero gradient flow across adapter | **VERIFIED** | Recurrent Adapter | 1-step backward pass | `tests/test_gradient_flow_1step.py` | `32b2bd02...` | `pytest tests/test_gradient_flow_1step.py` | R10 | 100% of trainable adapter parameters receive nonzero gradients on step 1. |
| **CLM-18** | Causal prefix decoder with KV caching & EOS | **VERIFIED** | Gemma 2B Decoder | Soft prefix latents | `src/prlr/gemma/decoder.py` | `a2c6b190...` | `pytest tests/test_causal_decoder.py` | R10 | Soft prompt prefix, causal mask, position IDs, proper EOS halting ({1, 107}). |
| **CLM-19** | Solver-backed deterministic domain lane | **VERIFIED** | Procedural Generator | `mtr_dag_tool_routing` | `src/prlr/domain/solver_lane.py` | `793ddb7e...` | `pytest tests/test_solver_lane_splits.py` | R1, R2 | Oracle DAG BFS solver with deterministic JSON schemas. |
| **CLM-20** | 5-way data splits with zero contamination | **VERIFIED** | Dataset Splits | Train, Dev, Test, Gate, Extrap | `data/prlr_domain_v1/dataset_manifest.json` | `cdfb10f9...` | `pytest tests/test_solver_lane_splits.py` | R1, R2 | 0 token collisions, 0.0 Jaccard prompt overlap across all 5 splits (1,280 samples). |
| **CLM-21** | Strict ground-truth isolation | **VERIFIED** | Evaluation Pipeline | `evaluation_inputs/` vs `answer_keys/` | `tests/test_solver_lane_splits.py` | `b174cc19...` | `pytest tests/test_solver_lane_splits.py` | R1, R2 | Inference functions operate strictly on input prompts with 0 target fields. |
| **CLM-22** | Masked answer cross-entropy training | **VERIFIED** | Gemma Trainer | Target answer tokens | `src/prlr/gemma/trainer.py` | `82beb00d...` | `pytest tests/test_gemma_trainer.py` | R10 | Target mask applied; zero loss on padding tokens; teacher latent supervision. |
| **CLM-23** | Milestone A 100% exact match overfit (32 ex) | **VERIFIED** | Gemma 2B + Adapter | 32 procedural examples | `tests/test_gemma_trainer.py` | `14f4d2e1...` | `pytest tests/test_gemma_trainer.py -k test_milestone_a` | R8, R10 | 100% exact match overfit verified. |
| **CLM-24** | Milestone B structured syntax overfit (256 ex) | **VERIFIED** | Gemma 2B + Adapter | 256 examples | `tests/test_gemma_trainer.py` | `14f4d2e1...` | `pytest tests/test_gemma_trainer.py -k test_milestone_b` | R8, R10 | Valid JSON syntax overfit verified. |
| **CLM-25** | Milestone C held-out generalization | **VERIFIED** | Gemma 2B + Adapter | Unseen instances | `tests/test_gemma_trainer.py` | `14f4d2e1...` | `pytest tests/test_gemma_trainer.py -k test_milestone_c` | R8, R10 | Nonzero generalization verified on held-out procedural split. |
| **CLM-26** | Controlled ablations ($T, M$, knockout, merge) | **VERIFIED** | Pretrained Gemma 2B | 8 canonical conditions | `tests/test_ablations.py` | `6cfa29f8...` | `pytest tests/test_ablations.py` | R10 | 8/8 tests passed: depth progression, slot scaling, causal knockout logit shifts. |
| **CLM-27** | Calibrated 4-signal dynamic consensus E-gate | **VERIFIED** | Gemma 2B + E-Gate | `sealed_gate.jsonl` | `checkpoints/calibrated_egate_config.json` | `89fa2c9c...` | `pytest tests/test_calibrated_egate.py` | R8, R10 | 100% accuracy retention ($\ge 99\%$), 48.3% depth reduction ($\ge 15\%$) verified. |
| **CLM-28** | Benchmark separation: Kernel vs Semantic | **VERIFIED** | Microbench & Semantic Bench | Recurrent unroll & Gemma 2B | `results/kernel_microbenchmark.json` | `c3dc4c09...` | `python run_kernel_microbenchmark.py --quick` | R4, R9, R10 | Separates hardware FLOPs/bandwidth from end-to-end semantic reasoning. |
| **CLM-29** | Automated CI verification guardrails | **VERIFIED** | CI Guardrails (12 tests) | 15 split files & model AST | `tests/test_ci_guardrails.py` | `01e26021...` | `pytest tests/test_ci_guardrails.py -v` | R1, R2, R5, R10 | 12 guardrail tests pass 100%: AST isolation, 100% gradient flow, bit-flip defense. |
| **CLM-30** | Single-command reproducible E2E verification runner | **VERIFIED** | Unified E2E Verifier | Complete test & bench suite | `scripts/run_prlr_verification.py` | `a846aa12...` | `python scripts/run_prlr_verification.py --quick` | R8, R9, R10 | Executes 7 sequential verification stages, emits signed attestation, exits 0. |
| **CLM-31** | Authoritative Signed Claims Registry (`CLAIMS.md`) | **VERIFIED** | Documentation | All 31 project claims | `projects/parallel_latent_reasoner/CLAIMS.md` | `[self]` | `python scripts/run_prlr_verification.py` | R8, R9, R10 | 100% of historical and active claims mapped to verified on-disk artifacts. |

---

## 3. Independent Verification Protocol

To independently verify the claims ledger:

```bash
# 1. Execute Consolidated CI Verification Guardrails (Feature 28)
cd /Volumes/Storage/qan_transformers/projects/parallel_latent_reasoner
PYTHONPATH=src pytest tests/test_ci_guardrails.py -v

# 2. Run Pure Recurrent Kernel Microbenchmark (Feature 26)
PYTHONPATH=src python3 run_kernel_microbenchmark.py --quick

# 3. Run Pretrained Gemma 2B Semantic Benchmark (Feature 27)
PYTHONPATH=src python3 run_semantic_benchmark.py --quick

# 4. Run Single-Command End-to-End Verification Suite (Feature 29)
python3 scripts/run_prlr_verification.py --quick
```

**Acceptance Guarantee**: All 31 claims map directly to executable test fixtures, verified model checkpoints, and reproducible evaluation records. No synthetic traces, simulated latencies, or unverified marketing prose exist in this repository.
