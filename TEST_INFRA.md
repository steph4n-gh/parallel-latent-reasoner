# Test Infrastructure Specification: Large Gemma 4 Empirical Benchmark Suite

**Project**: Parallel Latent Reasoner (PRLR) — Large Gemma 4 Empirical Benchmark Suite  
**Target Architecture**: MLX on Apple Silicon Metal GPU (Unified Memory)  
**Specification Version**: `1.0.0`  
**Test Runner**: `pytest projects/parallel_latent_reasoner/tests/ -v`  

---

## 1. Executive Summary & Verification Methodology

The test infrastructure for the **Large Gemma 4 Empirical Benchmark Suite** validates parallel continuous latent deliberation (PRLR) on Apple Silicon Metal GPUs across dense (Gemma 4 12B Q4) and mixture-of-experts (Gemma 4 26B A4B MoE) model scales.

Testing follows an opaque-box, 4-tier verification hierarchy ensuring functional correctness, numerical stability, deterministic rubric evaluation, and empirical hardware residency constraints.

```
+----------------------------------------------------------------------------------------------------+
|                                    4-TIER TEST PYRAMID ARCHITECTURE                                |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   Tier 4: Real-World Scenarios        [ E2E Dual-Mode Transcripts, Speedup >=25x, +0.00% VRAM ]    |
|   Tier 3: Cross-Feature Combinations  [ 3-Signal E-Gate + Large Presets + Probes + Rubrics ]       |
|   Tier 2: Boundary & Corner Cases     [ Empty Inputs, Degenerate Ranks, M=64 Slots, Deep Unroll ]  |
|   Tier 1: Feature Coverage            [ 25 Cognitive Tasks in 5 Domains, Model Configs, Harness ]  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Tier 1: Feature Coverage

Tier 1 exercises every discrete feature, dataclass, scoring rubric, and configuration preset in the PRLR large model suite.

### 2.1 Native Cognitive Domain Benchmark Suite (25 Test Cases across 5 Domains)

Every test case is defined with immutable identifiers, domain classification, input prompts, expected ground truth, and programmatic scoring rubrics.

| Domain | Task ID | Description & Target Output | Verifier Type | Expected Output Derivation |
|---|---|---|---|---|
| **Multi-Constraint Satisfaction (MCS)** | `mcs_01_spacecraft_payload` | 5-instrument payload optimization (Mass <= 40kg, Power <= 110W, Data >= 30Mbps, Zones A & B). Target: `"Beta, Gamma"` | `regex_constraint` | Mathematical integer constraint satisfaction maximizing data rate |
| | `mcs_02_constrained_pangrammatic_sentence` | 7-word sentence starting with -ly adverb, ending in plural noun, containing k, x, z, zero 'o'. | `regex_constraint` | Lexical constraint solver verifying all 5 grammar/orthographic rules |
| | `mcs_03_budget_itinerary_optimizer` | 4-day conference itinerary (Cost <= $600, CO2 <= 50kg, <= 1 flight, contains A1 or C1). Target: `"A1, B2, C1, D2"` | `exact_match` | Multi-objective combinatorial optimization schedule |
| | `mcs_04_cryptarithm_modular_logic` | 4-variable modular arithmetic system with distinct digits 1-9. Target: `"W=5, X=4, Y=2, Z=6"` | `regex_constraint` | Deterministic solution to algebraic system: $W+X=Y+Z+1$, $WX=YZ+8$, $W>X, Y<Z$ |
| | `mcs_05_microservice_traffic_shaper` | 3-path network QoS routing (BW >= 450Mbps, Avg Latency <= 20ms, Loss <= 0.6%, Cost <= $45/hr, includes P3). Target: `"P2, P3, P5"` | `exact_match` | Network flow linear optimization solver |
| **Winograd Schema & Disambiguation (WSD)** | `wsd_01_physical_affordance_trophy` | Trophy too large to fit in suitcase. Question: What was too large? Target: `"the trophy"` | `exact_match` | Winograd Schema physical containment affordance |
| | `wsd_02_reversal_trophy_small` | Trophy could not fit in suitcase because it was too small. Question: What was too small? Target: `"the suitcase"` | `exact_match` | Polarity inversion physical affordance binding |
| | `wsd_03_corporate_negotiation_fiduciary` | Apex sued Summit rather than Vertex because they breached contract. Who breached? Target: `"Summit Cargo"` | `exact_match` | 3-agent legal co-reference resolution |
| | `wsd_04_pharmacotherapy_drug_interaction` | Lisinopril prescribed over Metoprolol because it lowers ACE activity. What lowers ACE? Target: `"Lisinopril"` | `exact_match` | Biomedical pharmacological mechanism binding |
| | `wsd_05_legal_indemnity_clause` | Landlord indemnifies Tenant provided they did not cause defect through gross negligence. Who must not? Target: `"The Tenant"` | `exact_match` | Legal reciprocal indemnity co-reference |
| **Semantic Denoising & Intent (SDN)** | `sdn_01_angry_customer_return` | Extract structured refund parameters beneath sarcastic headphone rant. | `json_schema` | JSON object: `{"action": "REFUND", "order_id": "QX-99281", "product": "QuantumX Pro Headphones", "payment_target": "ORIGINAL_PAYMENT"}` |
| | `sdn_02_devops_slack_outage` | Isolate rollback target and version from chaotic multi-speaker incident chat. | `json_schema` | JSON object: `{"target_service": "payments-worker", "operation": "ROLLBACK", "target_version": "v3.0.9"}` |
| | `sdn_03_meeting_transcript_action_item` | Filter meeting banter to extract assignee, task description, and deadline. | `json_schema` | JSON object: `{"assignee": "Rachel", "deadline": "Thursday 5 PM", "task_description": "Patch PDF export Unicode bug"}` |
| | `sdn_04_sarcastic_database_query` | Extract true SQL update statement parameters beneath sarcastic hypothetical deletion remarks. | `json_schema` | JSON object: `{"statement_type": "UPDATE", "target_table": "transactions", "filter_id": "TXN-884102", "set_status": "REFUNDED"}` |
| | `sdn_05_rambling_travel_flight_search` | Extract origin, destination, date, and cabin class from rambling stream-of-consciousness text. | `json_schema` | JSON object: `{"origin_airport": "BOS", "destination_airport": "SFO", "departure_date": "2026-10-12", "cabin_class": "ECONOMY"}` |
| **Cross-Context Clue Synthesis (CMS)** | `cms_01_whodunit_alibi_deduction` | 5 suspects, 5 locations, 4 alibi statements. Identify thief leaving vintage fountain pen at Gallery. Target: `"Mrs. Peacock"` | `exact_match` | Deductive elimination matrix over disjoint alibis |
| | `cms_02_distributed_supply_chain_bottleneck` | 5-tier supply chain reports. Identify root cause supplier suffering power grid failure. Target: `"Supplier Beta"` | `exact_match` | Causal graph dependency traversal |
| | `cms_03_genealogy_kinship_deduction` | 4-hop lineage records (Charles=father of Arthur, Edward=father of Charles & Fiona, Fiona=mother of Brenda). Relationship of Brenda to Arthur? Target: `"First Cousin"` | `exact_match` | Kinship graph traversal (Degree 4 consanguinity) |
| | `cms_04_microservice_distributed_trace_diagnosis` | Asynchronous service logs across Gateway, Order, Payment, and RedisLock. Identify primary root-cause failure. Target: `"RedisLock"` | `exact_match` | Distributed trace causal chain root-cause analysis |
| | `cms_05_biochemical_pathway_inhibition` | 4-step metabolic cascade with competitive inhibitor for Enzyme 2. Outcome on Compound E? Target: `"Decreases"` | `exact_match` | Biochemical pathway dynamic flux deduction |
| **Action & Tool Routing (ATR)** | `atr_01_financial_portfolio_rebalancer` | Select portfolio rebalancing tool from 5-function financial catalog and extract target allocations and slippage. | `json_schema` | JSON object: `{"tool_id": "T4", "tool_name": "rebalance_portfolio_weights", "extracted_parameters": {"portfolio_id": "Fund-7", "max_slippage_bps": 15}}` |
| | `atr_02_cloud_infrastructure_provisioning` | Route brute-force attack mitigation request to WAF IP blocklist tool with ACL and CIDRs. | `json_schema` | JSON object: `{"tool_id": "T4", "tool_name": "update_waf_ip_blocklist", "target_acl": "acl-prod-us-east-1"}` |
| | `atr_03_biomedical_genomics_lookup` | Route BRCA2 mutation query to ClinVar tool with variant identifier and genome assembly. | `json_schema` | JSON object: `{"tool_id": "T1", "tool_name": "query_clinvar_variant", "variant_identifier": "NM_000059.3:c.5946del"}` |
| | `atr_04_smart_home_multimodal_dispatch` | Route dual-room heating request to HVAC controller tool with target temperature and mode. | `json_schema` | JSON object: `{"tool_id": "T1", "tool_name": "adjust_hvac_zones", "target_temp": 72.0, "mode": "heat"}` |
| | `atr_05_ecommerce_fulfillment_routing` | Route urgent warehouse SKU pick request to automated robotics dispatcher with priority. | `json_schema` | JSON object: `{"tool_id": "T3", "tool_name": "dispatch_warehouse_picker", "warehouse_id": "Warehouse-West", "priority": "HIGH"}` |

### 2.2 Large Model Architectural Presets

Validates shape, head-count, intermediate dimension, vocabulary, and MoE routing fields for large Gemma 4 configurations.

- `GemmaLatentConfig.gemma_12b_q4()`:
  - Hidden dimension: $D = 3840$
  - Query heads: $H = 16$
  - KV heads: $H_{\text{kv}} = 8$ (Grouped Query Attention ratio 2:1)
  - Head dimension: $d_k = 256$
  - Intermediate dimension: $D_{\text{mlp}} = 15360$ (or $16384$)
  - Layers: $L = 48$ (or $40$)
  - Memory Slots: $M = 16$
  - Deliberation Steps: $T = 8$ ($T_{\min} = 2, T_{\max} = 12$)
  - Peak Memory Bound: $\le 8.5\text{ GB}$ base residency.

- `GemmaLatentConfig.gemma_26b_a4b()`:
  - Hidden dimension: $D = 2816$ (or $3584$)
  - Query heads: $H = 16$, KV heads: $H_{\text{kv}} = 8$
  - Total Experts: $N_{\text{experts}} = 128$
  - Active Experts per Token: $K_{\text{active}} = 8$
  - MoE Intermediate Dimension: $D_{\text{moe}} = 704$
  - Peak Memory Bound: $\le 16.5\text{ GB}$ single-process residency.

### 2.3 Dual-Mode Evaluation Harness

Validates `EvaluationSampleResult`, `LargeGemmaDualEvaluator`, metric aggregations, speedup factors ($t_{\text{CoT}} / t_{\text{PRLR}}$), throughput calculations, and schema compliance (`prlr.gemma4_suite.v1`).

---

## 3. Tier 2: Boundary & Corner Cases

Tier 2 exercises edge-case inputs, mathematical singularities, and extreme parameter bounds.

1. **Empty & Whitespace Inputs**:
   - Evaluator handles empty strings `""`, whitespace `"   "`, and control characters safely without division-by-zero or indexing errors.
2. **Malformed & Adversarial Rubric Inputs**:
   - Evaluates JSON schema verifier against truncated JSON, invalid types (string instead of int), extra/missing fields, markdown code-fence wrappers (` ```json ... ``` `), and raw text with embedded JSON.
   - Evaluates regex and exact match verifiers against uppercase/lowercase variants, leading/trailing punctuation, and adversarial distractor substrings.
3. **Degenerate SVD States & Collinear Collapse**:
   - Rank-1 state matrices where all $M$ slots are identical: $\text{erank}(S) = 1.0000 \pm 10^{-4}$.
   - Orthogonal states where all $M$ slots are mutually orthogonal: $\text{erank}(S) = M.0000 \pm 10^{-4}$.
   - Near-singular states with small numerical perturbations ($\epsilon = 10^{-12}$) produce stable entropy values without `NaN` or `Inf`.
4. **Maximum Slot Scale & Deep Recurrent Unrolls**:
   - Scales memory slots from $M=16$ up to $M=64$.
   - Unrolls recurrent core to $T=64$ and $T=128$ steps; verifies ReZero residual scaling ($\alpha \le 0.05$) guarantees Lipschitz continuity ($\|S^{(T)}\| / \|S^{(0)}\| \le 1.25$).
5. **Score & Percentage Boundary Invariants**:
   - Verifies scoring bounds: $\text{score} \in [0.0, 1.0]$.
   - Verifies speedup metric: $\text{speedup} > 0.0$.
   - Verifies memory growth: $\Delta \text{VRAM} \le 0.05\text{ MB}$ ($+0.00\%$).

---

## 4. Tier 3: Cross-Feature Combinations

Tier 3 exercises interactions across multiple subsystems operating simultaneously.

1. **3-Signal E-Gate + Large Model Configs + SVD Probes**:
   - Executes multi-step unroll with `DynamicConsensusEGate` on `gemma_12b_q4` and `gemma_26b_a4b` configs.
   - Monitors simultaneous relative velocity decay ($v(t)/v(1) < 0.10$), Coda discrete prediction consensus ($\hat{y}^{(t)} == \hat{y}^{(t-1)}$), and SVD effective rank plateau ($|\Delta \text{erank}| < 0.005$).
2. **Cognitive Suite Integration with E-Gate Compute Dynamics**:
   - Verifies that simple cognitive queries (e.g. single entity Winograd schemas) trigger early convergence at $T \le 3$, while complex multi-constraint optimization problems deliberate deeper ($T \ge 6$).
3. **Telemetry Trajectory & Limit-Cycle Detection**:
   - Verifies `analyze_deliberation_trajectory` and `detect_limit_cycle` accurately capture trajectory properties across all cognitive task domains.

---

## 5. Tier 4: Real-World Application Scenarios

Tier 4 validates full end-to-end user workflows, transcript generation, and artifact serialization.

1. **End-to-End Cognitive Evaluation & Scoring**:
   - Executes full 25-task evaluation run across all 5 cognitive domains.
   - Verifies that each task yields a complete `EvaluationSampleResult` with both Mode 1 (AR CoT) and Mode 2 (PRLR) metrics.
2. **Side-by-Side Transcript Formatting**:
   - Formats evaluation results into side-by-side comparison transcripts with prompt, CoT thought trace, CoT answer, PRLR latent deliberation telemetry, and PRLR direct answer.
3. **Benchmark JSON Artifact Validation (`prlr.gemma4_suite.v1`)**:
   - Serializes complete suite results into JSON and validates top-level schema keys: `$schema`, `metadata`, `summary_metrics`, `domain_breakdown`, and `test_case_records`.
   - Verifies JSON roundtrip loading and data integrity.
4. **Hardware & Memory Residency Compliance**:
   - Verifies memory footprint remains within the $\le 16.5\text{ GB}$ single-process macOS unified memory envelope.

---

## 6. Verification Commands & Acceptance Criteria

### Test Runner Command
```bash
pytest projects/parallel_latent_reasoner/tests/ -v
```

### Acceptance Targets
- **Test Pass Rate**: 100% (0 failures, 0 errors, 0 regressions).
- **Domain Coverage**: 25/25 cognitive tasks verified across 5 domains.
- **Model Presets**: Gemma 4 12B Q4 and 26B A4B MoE verified.
- **Speedup Target**: $\ge 25.0\times$ reasoning phase speedup in comparative evaluations.
- **Memory Invariant**: $+0.00\%$ memory growth during continuous latent deliberation unrolls.
