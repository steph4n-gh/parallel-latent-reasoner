"""Comprehensive Test Suite for Native Cognitive Domain Benchmark Suite.

Verifies:
1. All 25 cognitive domain test cases are well-formed across 5 domains.
2. Domain filtering and summary statistics work correctly.
3. Deterministic scoring rubrics evaluate accurately on passing, failing, and adversarial responses.
4. JSON schema, regex constraint, exact match, and specialized multi-constraint verifiers.
"""

import json
import pytest

from parallel_latent_reasoner.cognitive_suite import (
    CognitiveTestCase,
    DomainType,
    EvaluationResult,
    VerifierType,
    get_domain_summary,
    get_test_case_by_id,
    load_cognitive_benchmark_suite,
    verify_test_case_result,
)


def test_suite_loading_and_counts():
    """Verify loading the full benchmark suite returns at least 25 cases across 5 domains."""
    suite = load_cognitive_benchmark_suite()
    assert len(suite) >= 25, f"Expected at least 25 test cases, got {len(suite)}"

    domain_counts: dict[str, int] = {}
    for case in suite:
        dom_val = case.domain.value if isinstance(case.domain, DomainType) else str(case.domain)
        domain_counts[dom_val] = domain_counts.get(dom_val, 0) + 1

    expected_domains = [
        "multi_constraint",
        "winograd_schema",
        "semantic_denoising",
        "multi_clue_synthesis",
        "action_tool_routing",
    ]
    for dom in expected_domains:
        assert dom in domain_counts, f"Missing domain {dom} in loaded suite"
        assert domain_counts[dom] >= 5, f"Domain {dom} has fewer than 5 test cases ({domain_counts[dom]})"


def test_domain_filtering():
    """Verify domain filtering returns only test cases for the specified domain."""
    for dom in DomainType:
        filtered_enum = load_cognitive_benchmark_suite(domain=dom)
        filtered_str = load_cognitive_benchmark_suite(domain=dom.value)
        assert len(filtered_enum) == len(filtered_str)
        assert len(filtered_enum) >= 5
        for case in filtered_enum:
            case_dom = case.domain.value if isinstance(case.domain, DomainType) else str(case.domain)
            assert case_dom == dom.value


def test_get_domain_summary():
    """Verify get_domain_summary returns complete metadata and accurate counts."""
    summary = get_domain_summary()
    assert "_total_test_cases" in summary
    assert summary["_total_test_cases"] >= 25

    for dom in DomainType:
        assert dom.value in summary
        dom_info = summary[dom.value]
        assert dom_info["count"] >= 5
        assert len(dom_info["test_case_ids"]) == dom_info["count"]
        assert len(dom_info["titles"]) == dom_info["count"]


def test_get_test_case_by_id():
    """Verify retrieving test cases by ID (case-insensitive and whitespace-tolerant)."""
    case = get_test_case_by_id("mcs_01")
    assert case is not None
    assert case.id.lower() == "mcs_01"

    # Case insensitivity
    case_upper = get_test_case_by_id("MCS_01")
    assert case_upper is not None
    assert case_upper.id == case.id

    # Non-existent ID
    assert get_test_case_by_id("non_existent_id_999") is None


def test_all_25_test_cases_well_formedness():
    """Verify structural validity and schema adherence for all 25 test cases."""
    suite = load_cognitive_benchmark_suite()
    seen_ids = set()

    for case in suite:
        assert case.id, "Test case must have non-empty id"
        assert case.id not in seen_ids, f"Duplicate test case ID: {case.id}"
        seen_ids.add(case.id)

        assert case.title, f"Case {case.id} must have a non-empty title"
        assert case.prompt, f"Case {case.id} must have a non-empty prompt"
        assert case.ground_truth, f"Case {case.id} must have a non-empty ground truth"
        assert isinstance(case.expected_constraints, list), f"Case {case.id} constraints must be list"
        assert isinstance(case.metadata, dict), f"Case {case.id} metadata must be dict"

        # Roundtrip dict serialization
        d = case.to_dict()
        assert d["id"] == case.id
        reconstructed = CognitiveTestCase.from_dict(d)
        assert reconstructed.id == case.id


def test_evaluation_result_tuple_unpacking():
    """Verify EvaluationResult supports both attribute access and tuple unpacking."""
    res = EvaluationResult(passed=True, score=1.0, feedback="Success", details={"k": "v"})
    passed, score, details = res
    assert passed is True
    assert score == 1.0
    assert details == {"k": "v"}
    assert res.feedback == "Success"

    d = res.to_dict()
    assert d["passed"] is True
    assert d["score"] == 1.0


# ============================================================================
# Domain 1: Multi-Constraint Satisfaction (MCS)
# ============================================================================

def test_rubrics_mcs_01_spacecraft_payload():
    """Test spacecraft payload optimization rubric (Beta, Gamma)."""
    case = get_test_case_by_id("mcs_01")
    assert case is not None

    # Passing response (Optimal: Beta + Gamma = 40 Mbps, 33kg <= 40kg, 95W <= 110W, Zones A and B)
    res_pass = verify_test_case_result(case, "Selected instruments: Beta, Gamma")
    assert res_pass.passed is True
    assert res_pass.score == 1.0

    # Suboptimal but valid combinations (e.g. Alpha + Beta = 35 Mbps)
    res_subopt = verify_test_case_result(case, "Alpha, Beta")
    assert res_subopt.passed is False
    assert res_subopt.score == 0.5

    # Failing response: Mass violation (Beta + Delta = 40kg mass, 140W > 110W power)
    res_fail_power = verify_test_case_result(case, "Beta, Delta")
    assert res_fail_power.passed is False
    assert res_fail_power.score == 0.0

    # Failing response: Zone violation (Alpha + Gamma: both Zone A, no Zone B)
    res_fail_zone = verify_test_case_result(case, "Alpha, Gamma")
    assert res_fail_zone.passed is False


def test_rubrics_mcs_02_pangrammatic_sentence():
    """Test 7-word constrained sentence without 'o' (mcs_02)."""
    case = get_test_case_by_id("mcs_02")
    assert case is not None

    # Passing response (7 words, starts with adverb -ly, ends in plural noun, k/x/z present, no 'o')
    valid_sentence = "Quickly six black wizards fix tiny puzzles"
    res_pass = verify_test_case_result(case, valid_sentence)
    assert res_pass.passed is True
    assert res_pass.score == 1.0

    # Fail: Contains forbidden letter 'o'
    res_fail_o = verify_test_case_result(case, "Slowly six black wizards fix tiny puzzles")
    assert res_fail_o.passed is False

    # Fail: Wrong word count (6 words)
    res_fail_len = verify_test_case_result(case, "Quickly six wizards fix tiny puzzles")
    assert res_fail_len.passed is False

    # Fail: First word not -ly adverb
    res_fail_first = verify_test_case_result(case, "Quick six black wizards fix tiny puzzles")
    assert res_fail_first.passed is False

    # Fail: Missing letter 'k'
    res_fail_char = verify_test_case_result(case, "Briefly six brave wizards fix tiny puzzles")
    assert res_fail_char.passed is False


def test_rubrics_mcs_03_budget_itinerary():
    """Test 4-day conference itinerary (mcs_03)."""
    case = get_test_case_by_id("mcs_03")
    assert case is not None

    # Passing response (A1, B2, C1, D2: Cost $500 <= $600, CO2 35kg <= 50kg, 0 flights)
    res_pass = verify_test_case_result(case, "Optimal itinerary: A1, B2, C1, D2")
    assert res_pass.passed is True
    assert res_pass.score == 1.0

    # Fail: Budget exceeded (A1, B1, C1, D1 = $200 + $300 + $150 + $250 = $900 > $600)
    res_fail_budget = verify_test_case_result(case, "A1, B1, C1, D1")
    assert res_fail_budget.passed is False

    # Fail: Flights exceeded (B1 + C2 = 2 flights > 1)
    res_fail_flights = verify_test_case_result(case, "A2, B1, C2, D2")
    assert res_fail_flights.passed is False


def test_rubrics_mcs_04_cryptarithm():
    """Test cryptarithm modular logic W=5, X=4, Y=2, Z=6 (mcs_04)."""
    case = get_test_case_by_id("mcs_04")
    assert case is not None

    res_pass = verify_test_case_result(case, "The assignment is W=5, X=4, Y=2, Z=6")
    assert res_pass.passed is True
    assert res_pass.score == 1.0

    # Fail: W > X violated (W=4, X=5)
    res_fail_order = verify_test_case_result(case, "W=4, X=5, Y=2, Z=6")
    assert res_fail_order.passed is False

    # Fail: Non-distinct variables
    res_fail_dup = verify_test_case_result(case, "W=5, X=4, Y=2, Z=5")
    assert res_fail_dup.passed is False


def test_rubrics_mcs_05_traffic_shaper():
    """Test microservice traffic shaper P2, P3, P5 (mcs_05)."""
    case = get_test_case_by_id("mcs_05")
    assert case is not None

    res_pass = verify_test_case_result(case, "Routing traffic via: P2, P3, P5")
    assert res_pass.passed is True
    assert res_pass.score == 1.0

    # Fail: Exceeds packet loss constraint (P4 has 1.2% loss > 0.6%)
    res_fail_loss = verify_test_case_result(case, "P1, P3, P4")
    assert res_fail_loss.passed is False

    # Fail: Missing required P3
    res_fail_p3 = verify_test_case_result(case, "P1, P2, P5")
    assert res_fail_p3.passed is False


# ============================================================================
# Domain 2: Winograd Schema & Pronoun Disambiguation (WSD)
# ============================================================================

def test_rubrics_wsd_cases():
    """Test all 5 Winograd Schema test cases (wsd_01..wsd_05)."""
    # wsd_01: Trophy too large
    c1 = get_test_case_by_id("wsd_01")
    assert verify_test_case_result(c1, "The trophy").passed is True
    assert verify_test_case_result(c1, "the bronze trophy").passed is True
    assert verify_test_case_result(c1, "the suitcase").passed is False

    # wsd_02: Suitcase too small
    c2 = get_test_case_by_id("wsd_02")
    assert verify_test_case_result(c2, "the suitcase").passed is True
    assert verify_test_case_result(c2, "the travel suitcase").passed is True
    assert verify_test_case_result(c2, "the trophy").passed is False

    # wsd_03: Summit Cargo contract breach
    c3 = get_test_case_by_id("wsd_03")
    assert verify_test_case_result(c3, "Summit Cargo").passed is True
    assert verify_test_case_result(c3, "Apex Logistics").passed is False
    assert verify_test_case_result(c3, "Vertex Express").passed is False

    # wsd_04: Lisinopril ACE mechanism
    c4 = get_test_case_by_id("wsd_04")
    assert verify_test_case_result(c4, "Lisinopril").passed is True
    assert verify_test_case_result(c4, "Metoprolol").passed is False

    # wsd_05: Legal indemnity clause (The Tenant)
    c5 = get_test_case_by_id("wsd_05")
    assert verify_test_case_result(c5, "The Tenant").passed is True
    assert verify_test_case_result(c5, "The Landlord").passed is False


# ============================================================================
# Domain 3: Semantic Denoising & Noisy Intent Extraction (SDN)
# ============================================================================

def test_rubrics_sdn_cases():
    """Test all 5 Semantic Denoising test cases (sdn_01..sdn_05)."""
    # sdn_01: Angry customer refund
    c1 = get_test_case_by_id("sdn_01")
    gt1 = {
        "action": "REFUND",
        "order_id": "QX-99281",
        "product": "QuantumX Pro Headphones",
        "payment_target": "ORIGINAL_PAYMENT",
    }
    assert verify_test_case_result(c1, json.dumps(gt1)).passed is True
    # Test markdown-wrapped JSON
    assert verify_test_case_result(c1, f"```json\n{json.dumps(gt1)}\n```").passed is True
    # Test failure: wrong action
    assert verify_test_case_result(c1, json.dumps({**gt1, "action": "REPLACEMENT"})).passed is False

    # sdn_02: DevOps incident rollback
    c2 = get_test_case_by_id("sdn_02")
    gt2 = {
        "target_service": "payments-worker",
        "operation": "ROLLBACK",
        "target_version": "v3.0.9",
    }
    assert verify_test_case_result(c2, json.dumps(gt2)).passed is True
    assert verify_test_case_result(c2, json.dumps({**gt2, "target_service": "auth-service"})).passed is False

    # sdn_03: Meeting action item
    c3 = get_test_case_by_id("sdn_03")
    gt3 = {
        "assignee": "Rachel",
        "task_description": "Patch PDF export Unicode bug",
        "deadline": "Thursday 5 PM",
    }
    assert verify_test_case_result(c3, json.dumps(gt3)).passed is True
    assert verify_test_case_result(c3, json.dumps({**gt3, "assignee": "Tom"})).passed is False

    # sdn_04: Sarcastic SQL update
    c4 = get_test_case_by_id("sdn_04")
    gt4 = {
        "statement_type": "UPDATE",
        "target_table": "transactions",
        "filter_id": "TXN-884102",
        "set_status": "REFUNDED",
    }
    assert verify_test_case_result(c4, json.dumps(gt4)).passed is True
    assert verify_test_case_result(c4, json.dumps({**gt4, "statement_type": "DELETE"})).passed is False

    # sdn_05: Rambling flight search
    c5 = get_test_case_by_id("sdn_05")
    gt5 = {
        "origin_airport": "BOS",
        "destination_airport": "SFO",
        "departure_date": "2026-10-12",
        "cabin_class": "ECONOMY",
    }
    assert verify_test_case_result(c5, json.dumps(gt5)).passed is True
    assert verify_test_case_result(c5, json.dumps({**gt5, "origin_airport": "LHR"})).passed is False


# ============================================================================
# Domain 4: Cross-Context Multi-Clue Synthesis (CMS)
# ============================================================================

def test_rubrics_cms_cases():
    """Test all 5 Cross-Context Synthesis test cases (cms_01..cms_05)."""
    # cms_01: Whodunit alibi (Mrs. Peacock)
    c1 = get_test_case_by_id("cms_01")
    assert verify_test_case_result(c1, "Mrs. Peacock").passed is True
    assert verify_test_case_result(c1, "Professor Plum").passed is False
    assert verify_test_case_result(c1, "Colonel Mustard").passed is False

    # cms_02: Supply chain bottleneck (Supplier Beta)
    c2 = get_test_case_by_id("cms_02")
    assert verify_test_case_result(c2, "Supplier Beta").passed is True
    assert verify_test_case_result(c2, "Fab Alpha").passed is False

    # cms_03: Genealogy kinship (First Cousin)
    c3 = get_test_case_by_id("cms_03")
    assert verify_test_case_result(c3, "First Cousin").passed is True
    assert verify_test_case_result(c3, "Sister").passed is False

    # cms_04: Microservice distributed trace (RedisLock)
    c4 = get_test_case_by_id("cms_04")
    assert verify_test_case_result(c4, "RedisLock").passed is True
    assert verify_test_case_result(c4, "PaymentService").passed is False

    # cms_05: Biochemical pathway inhibition (Decreases)
    c5 = get_test_case_by_id("cms_05")
    assert verify_test_case_result(c5, "Decreases").passed is True
    assert verify_test_case_result(c5, "Increases").passed is False


# ============================================================================
# Domain 5: Action & Tool Routing (ATR)
# ============================================================================

def test_rubrics_atr_cases():
    """Test all 5 Action & Tool Routing test cases (atr_01..atr_05)."""
    # atr_01: Portfolio rebalancer (T4)
    c1 = get_test_case_by_id("atr_01")
    gt1 = {
        "tool_id": "T4",
        "tool_name": "rebalance_portfolio_weights",
        "extracted_parameters": {"portfolio_id": "Fund-7", "max_slippage_bps": 15},
    }
    assert verify_test_case_result(c1, json.dumps(gt1)).passed is True
    assert verify_test_case_result(c1, json.dumps({**gt1, "tool_id": "T1"})).passed is False

    # atr_02: WAF blocklist (T4)
    c2 = get_test_case_by_id("atr_02")
    gt2 = {
        "tool_id": "T4",
        "tool_name": "update_waf_ip_blocklist",
        "target_acl": "acl-prod-us-east-1",
    }
    assert verify_test_case_result(c2, json.dumps(gt2)).passed is True
    assert verify_test_case_result(c2, json.dumps({**gt2, "target_acl": "acl-dev"})).passed is False

    # atr_03: ClinVar variant query (T1)
    c3 = get_test_case_by_id("atr_03")
    gt3 = {
        "tool_id": "T1",
        "tool_name": "query_clinvar_variant",
        "variant_identifier": "NM_000059.3:c.5946del",
    }
    assert verify_test_case_result(c3, json.dumps(gt3)).passed is True
    assert verify_test_case_result(c3, json.dumps({**gt3, "tool_id": "T3"})).passed is False

    # atr_04: Smart home HVAC (T1)
    c4 = get_test_case_by_id("atr_04")
    gt4 = {
        "tool_id": "T1",
        "tool_name": "adjust_hvac_zones",
        "target_temp": 72.0,
        "mode": "heat",
    }
    assert verify_test_case_result(c4, json.dumps(gt4)).passed is True
    assert verify_test_case_result(c4, json.dumps({**gt4, "target_temp": 65.0})).passed is False

    # atr_05: Warehouse picker (T3)
    c5 = get_test_case_by_id("atr_05")
    gt5 = {
        "tool_id": "T3",
        "tool_name": "dispatch_warehouse_picker",
        "warehouse_id": "Warehouse-West",
        "priority": "HIGH",
    }
    assert verify_test_case_result(c5, json.dumps(gt5)).passed is True
    assert verify_test_case_result(c5, json.dumps({**gt5, "priority": "LOW"})).passed is False


# ============================================================================
# Adversarial & Edge Case Verification
# ============================================================================

def test_rubrics_adversarial_and_empty_inputs():
    """Verify robust handling of empty strings, gibberish, and corrupted JSON."""
    suite = load_cognitive_benchmark_suite()
    for case in suite:
        # Empty string
        res_empty = verify_test_case_result(case, "")
        assert res_empty.passed is False
        assert res_empty.score == 0.0

        # Pure whitespace
        res_ws = verify_test_case_result(case, "   \n\t  ")
        assert res_ws.passed is False

        # Random conversational noise
        res_noise = verify_test_case_result(case, "I am an AI assistant and I am not sure about this.")
        assert res_noise.passed is False

        # Malformed JSON
        res_malformed = verify_test_case_result(case, '{"tool_id": "T1", "unclosed_string: 123')
        assert res_malformed.passed is False
