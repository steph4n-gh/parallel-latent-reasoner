"""Comprehensive Test Suite for Multi-Domain Dataset Generation and Validation.

Covers Tier 1 (Feature Coverage & Unit Contracts) and Tier 2 (Boundary & Adversarial Stress):
1. Procedural and curated generation across 4 core reasoning domains:
   - Domain 1: Multi-Step Reasoning & Multi-Clue Synthesis
   - Domain 2: Constraint Satisfaction & Combinatorial Optimization
   - Domain 3: Entity Disambiguation & Semantic Binding
   - Domain 4: Arithmetic & Quantitative Reasoning
2. Schema & format validation (JSONL serialization, dataclass contracts, prlr.dataset.v1 schema).
3. 100% verifier pass rate on ground-truth solutions across all 4 domains.
4. Distractor / negative solution rejection guards.
5. 80/10/10 split generation with zero prompt hash collision and <15% 8-gram Jaccard overlap.
6. Token length boundary validation.
7. Adversarial, malformed, and boundary stress tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pytest

from parallel_latent_reasoner.cognitive_suite import (
    CognitiveTestCase,
    DomainType,
    EvaluationResult,
    VerifierType,
    load_cognitive_benchmark_suite,
    verify_test_case_result,
)


# ============================================================================
# Core Dataset Schema & Reference Generator Models
# ============================================================================

@dataclass
class DistillationSample:
    """Standardized dataset record conforming to prlr.dataset.v1 schema."""

    id: str
    domain: str
    subdomain: str
    prompt: str
    ground_truth: str
    teacher_cot: Optional[str]
    target_solution: str
    verifier_type: str
    verifier_config: Dict[str, Any]
    difficulty: int = 1
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DistillationSample:
        return cls(
            id=data["id"],
            domain=data["domain"],
            subdomain=data.get("subdomain", "general"),
            prompt=data["prompt"],
            ground_truth=data["ground_truth"],
            teacher_cot=data.get("teacher_cot") or data.get("teacher_cot_thought"),
            target_solution=data.get("target_solution") or data.get("target_solution_tokens", data["ground_truth"]),
            verifier_type=data.get("verifier_type", "exact_match"),
            verifier_config=data.get("verifier_config", {}),
            difficulty=data.get("difficulty", 1),
            seed=data.get("seed", 42),
            metadata=data.get("metadata", {}),
        )


class ProceduralMultiDomainGenerator:
    """Procedural task generator producing verifiable samples across all 4 reasoning domains."""

    @staticmethod
    def generate_multistep_reasoning(seed: int = 101, split: str = "train") -> DistillationSample:
        """Domain 1: Multi-Step Reasoning & Multi-Clue Synthesis."""
        if split == "test":
            # Holdout test domain scenarios
            prompt = (
                f"Aerospace Diagnostic #{seed}: Telemetry indicates Hydraulic Actuator failed, which caused Flap Lock. "
                "Flap Lock forced Manual Override, which triggered Safe Mode. "
                "What was the initial mechanical failure? Options: [Hydraulic Actuator, Flap Lock, Safe Mode]. "
                "Respond with only the root cause."
            )
            gt = "Hydraulic Actuator"
            cot = f"Root cause trace: Safe Mode <- Manual Override <- Flap Lock <- Hydraulic Actuator. Initial root cause: {gt}."
            v_type = "exact_match"
            v_cfg = {"expected_entity": gt, "case_sensitive": False}
            subdomain = "aerospace_causal_holdout"
        else:
            subdomain_type = seed % 4
            if subdomain_type == 0:
                suspects_pool = [
                    ("Alice", "Bob", "Charlie", "David"),
                    ("Elena", "Felix", "Grace", "Hector"),
                    ("Iris", "Jack", "Kira", "Liam"),
                    ("Maya", "Noah", "Olivia", "Peter"),
                ]
                s1, s2, culprit, s4 = suspects_pool[seed % len(suspects_pool)]
                hour = 7 + (seed % 4)
                minute = 10 + (seed % 40)
                prompt = (
                    f"Incident Report #{seed}: Security breach in Sector {seed % 10 + 1} at {hour}:{minute:02d} PM.\n"
                    f"- {s1} attended the {hour}:00 PM keynote with 50 witnesses in Auditorium A.\n"
                    f"- {s2} was logged on a video call from {hour-1}:45 PM to {hour+1}:00 PM.\n"
                    f"- {culprit} claims they walked alone in the courtyard at {hour}:{minute:02d} PM.\n"
                    f"- {s4} was in the server room with timestamped badge access logs.\n"
                    f"Identify the only suspect without a verified alibi. Respond with only the name."
                )
                cot = f"Checking alibis: {s1}, {s2}, {s4} have verified logs. {culprit} has no alibi. Conclusion: {culprit}."
                gt = culprit
                v_type = "exact_match"
                v_cfg = {"expected_entity": culprit, "case_sensitive": False}
                subdomain = "alibi_deduction"
            elif subdomain_type == 1:
                root_cause = "Power Surge"
                prompt = (
                    f"Diagnostic Telemetry #{seed}: Upstream analysis indicates Power Surge caused Sensor Failure. "
                    "Sensor Failure triggered Cooling Leak, which led to Emergency Shutdown and Turbine Trip. "
                    "What was the initial root cause component failure? Respond with only the root cause."
                )
                cot = f"Root cause trace: Turbine Trip <- Emergency Shutdown <- Cooling Leak <- Sensor Failure <- Power Surge. Conclusion: {root_cause}."
                gt = root_cause
                v_type = "exact_match"
                v_cfg = {"expected_entity": root_cause, "case_sensitive": False}
                subdomain = "causal_dependency"
            elif subdomain_type == 2:
                prompt = (
                    f"Kinship Query #{seed}: Arthur is the father of Beatrice. Beatrice is the mother of Charles. "
                    "Charles is the brother of Daisy. What is the kinship relation of Arthur to Daisy? "
                    "Options: [father, grandfather, uncle, brother]. Respond with only the relation."
                )
                cot = "Arthur is father of mother (Beatrice) of Daisy -> Arthur is grandfather to Daisy. Conclusion: grandfather."
                gt = "grandfather"
                v_type = "exact_match"
                v_cfg = {"expected_entity": "grandfather", "case_sensitive": False}
                subdomain = "genealogical_kinship"
            else:
                prompt = (
                    f"Metabolic Flux #{seed}: Compound A is converted to Compound B by Enzyme E1. "
                    "Compound B is converted to Product C by Enzyme E2. If a competitive inhibitor blocks Enzyme E1, "
                    "what happens to the concentration of Product C? Options: [increases, decreases, remains constant]."
                )
                cot = "Inhibiting E1 halts production of B -> lack of B halts production of C -> Product C decreases. Conclusion: decreases."
                gt = "decreases"
                v_type = "exact_match"
                v_cfg = {"expected_entity": "decreases", "case_sensitive": False}
                subdomain = "biochemical_flux"

        return DistillationSample(
            id=f"msr_{subdomain}_{seed}",
            domain="multi_step_reasoning",
            subdomain=subdomain,
            prompt=prompt,
            ground_truth=gt,
            teacher_cot=cot,
            target_solution=gt,
            verifier_type=v_type,
            verifier_config=v_cfg,
            difficulty=2,
            seed=seed,
            metadata={"split": split, **v_cfg},
        )

    @staticmethod
    def generate_constraint_satisfaction(seed: int = 202, split: str = "train") -> DistillationSample:
        """Domain 2: Constraint Satisfaction & Combinatorial Optimization."""
        if split == "test":
            # Holdout server provisioning scenario
            optimal_servers = ["Node_X", "Node_Z"]
            prompt = (
                f"Cloud Infrastructure Allocation #{seed}: Select 2 compute instances providing total RAM >= 64 GB and cost <= $120/mo.\n"
                "Instances: Node_X (32 GB, $50/mo), Node_Y (16 GB, $30/mo), Node_Z (48 GB, $65/mo).\n"
                "Output JSON with key 'selected_nodes'."
            )
            gt = json.dumps({"selected_nodes": optimal_servers})
            cot = f"Selecting Node_X + Node_Z yields 80 GB RAM (> 64) for $115/mo (<= $120). Solution: {gt}."
            v_type = "json_schema"
            v_cfg = {"expected_fields": {"selected_nodes": optimal_servers}, "required_keys": ["selected_nodes"]}
            subdomain = "cloud_provisioning_holdout"
        else:
            subdomain_type = seed % 4
            if subdomain_type == 0:
                optimal_payload = ["Spectrometer", "Magnetometer", "Camera"]
                prompt = (
                    f"Payload Optimization #{seed}: Select instrument set maximizing data rate with mass <= 30kg, power <= 85W, "
                    "including Zone A and Zone B instruments.\n"
                    "Options: Spectrometer (12kg, 35W, 25Mbps, Zone A), Magnetometer (8kg, 20W, 15Mbps, Zone B), "
                    "Radar (18kg, 55W, 30Mbps, Zone A), Camera (10kg, 30W, 20Mbps, Zone B).\n"
                    "Output JSON with key 'selected_instruments'."
                )
                gt = json.dumps({"selected_instruments": optimal_payload})
                cot = f"Optimal spacecraft payload selection evaluated across constraints. Solution: {gt}."
                v_type = "json_schema"
                v_cfg = {"expected_fields": {"selected_instruments": optimal_payload}, "required_keys": ["selected_instruments"]}
                subdomain = "payload_knapsack"
            elif subdomain_type == 1:
                selected_paths = ["Path_Alpha", "Path_Gamma"]
                prompt = (
                    f"Network Traffic Shaper #{seed}: Choose 2 routing paths achieving total bandwidth >= 300 Mbps and latency <= 25 ms.\n"
                    "Paths: Path_Alpha (200 Mbps, 15 ms), Path_Beta (100 Mbps, 35 ms), Path_Gamma (150 Mbps, 10 ms).\n"
                    "Output JSON with key 'active_paths'."
                )
                gt = json.dumps({"active_paths": selected_paths})
                cot = f"Network path selection optimization satisfying latency and bandwidth constraints. Solution: {gt}."
                v_type = "json_schema"
                v_cfg = {"expected_fields": {"active_paths": selected_paths}, "required_keys": ["active_paths"]}
                subdomain = "qos_routing"
            elif subdomain_type == 2:
                prompt = (
                    f"Diophantine Cryptarithm #{seed}: Find unique single-digit positive integers W, X, Y, Z such that "
                    "W + X = 10, W * X = 21, and W > X. What is the value of W? Respond with only the digit."
                )
                gt = "7"
                cot = "Solving quadratic system: W + X = 10 and W * X = 21. Solutions are 7 and 3. Since W > X, W = 7."
                v_type = "exact_match"
                v_cfg = {"expected_entity": "7", "case_sensitive": False}
                subdomain = "cryptarithm_diophantine"
            else:
                prompt = (
                    f"Lexical Constraint #{seed}: What is the 5-letter word meaning 'a fast-running bird' starting with 'z' and ending with 'a'? "
                    "Options: [zebra, zonda, zorba]. Respond with only the word."
                )
                gt = "zebra"
                cot = "Searching lexicon for 5-letter animal word starting with 'z' and ending with 'a'. Answer is zebra."
                v_type = "exact_match"
                v_cfg = {"expected_entity": "zebra", "case_sensitive": False}
                subdomain = "lexical_pangram"

        return DistillationSample(
            id=f"csp_{subdomain}_{seed}",
            domain="constraint_satisfaction",
            subdomain=subdomain,
            prompt=prompt,
            ground_truth=gt,
            teacher_cot=cot,
            target_solution=gt,
            verifier_type=v_type,
            verifier_config=v_cfg,
            difficulty=3,
            seed=seed,
            metadata={"split": split, **v_cfg},
        )

    @staticmethod
    def generate_entity_disambiguation(seed: int = 303, split: str = "train") -> DistillationSample:
        """Domain 3: Entity Disambiguation & Semantic Binding."""
        if split == "test":
            # Holdout robotics manipulation scenario
            prompt = (
                f"Robotics Gripper Telemetry #{seed}: The robotic arm dropped the delicate glass vial onto the steel workbench because it was too slippery.\n"
                "What does 'it' refer to? Options: [the delicate glass vial, the steel workbench]. Respond with only the referent."
            )
            gt = "the delicate glass vial"
            cot = "Physical affordance analysis: An object slips from a gripper because the object itself is too slippery. 'it' = the delicate glass vial."
            v_type = "exact_match"
            v_cfg = {"expected_entity": gt, "case_sensitive": False}
            subdomain = "robotics_affordance_holdout"
        else:
            subdomain_type = seed % 4
            if subdomain_type == 0:
                prompt = (
                    f"Affordance Query #{seed}: The grand piano could not fit through the narrow doorway because it was too wide.\n"
                    "What does 'it' refer to? Options: [the grand piano, the narrow doorway]. Respond with only the referent."
                )
                gt = "the grand piano"
                cot = "Physical geometry: An object cannot pass through an opening because the object is too wide. Referent is the grand piano."
                v_type = "exact_match"
                v_cfg = {"expected_entity": "the grand piano", "case_sensitive": False}
                subdomain = "winograd_affordance"
            elif subdomain_type == 1:
                order_id = f"ORD-{1000 + (seed % 9000)}"
                prompt = (
                    f"Support Ticket #{seed}: 'Oh wonderful, my package {order_id} arrived in pieces like a puzzle! "
                    "I definitely wanted broken glass for dinner. Give me a full REFUND right now!'\n"
                    "Extract structured JSON with keys 'action' and 'order_id'."
                )
                gt = json.dumps({"action": "REFUND", "order_id": order_id})
                cot = f"Semantic denoising sarcasm detection: Action is REFUND and Order ID is {order_id}."
                v_type = "json_schema"
                v_cfg = {"expected_fields": {"action": "REFUND", "order_id": order_id}, "required_keys": ["action", "order_id"]}
                subdomain = "sarcastic_intent"
            elif subdomain_type == 2:
                prompt = (
                    f"Legal Clause #{seed}: 'The Licensor shall indemnify the Licensee against third-party patent claims, "
                    "provided the Licensee gives prompt written notice.' Who is obligated to provide indemnity?\n"
                    "Options: [the Licensor, the Licensee]. Respond with only the party."
                )
                gt = "the Licensor"
                cot = "Contract analysis: The Licensor agrees to indemnify the Licensee. Obligated party: the Licensor."
                v_type = "exact_match"
                v_cfg = {"expected_entity": "the Licensor", "case_sensitive": False}
                subdomain = "legal_indemnity"
            else:
                prompt = (
                    f"Pharmacology Disambiguation #{seed}: Lisinopril and Metoprolol were administered to the patient. "
                    "Which medication acts as an ACE inhibitor? Options: [Lisinopril, Metoprolol]. Respond with only the drug name."
                )
                gt = "Lisinopril"
                cot = "Biomedical taxonomy: Lisinopril is an ACE inhibitor, while Metoprolol is a beta-blocker. Answer: Lisinopril."
                v_type = "exact_match"
                v_cfg = {"expected_entity": "Lisinopril", "case_sensitive": False}
                subdomain = "biomedical_binding"

        return DistillationSample(
            id=f"wsd_{subdomain}_{seed}",
            domain="entity_disambiguation",
            subdomain=subdomain,
            prompt=prompt,
            ground_truth=gt,
            teacher_cot=cot,
            target_solution=gt,
            verifier_type=v_type,
            verifier_config=v_cfg,
            difficulty=2,
            seed=seed,
            metadata={"split": split, **v_cfg},
        )

    @staticmethod
    def generate_arithmetic(seed: int = 404, split: str = "train") -> DistillationSample:
        """Domain 4: Arithmetic & Quantitative Reasoning."""
        if split == "test":
            # Holdout compound chemistry concentration calculation
            sol_vol = 500
            solute_pct = 12
            mass = int(sol_vol * solute_pct / 100)  # 60 grams
            prompt = (
                f"Chemistry Laboratory Measurement #{seed}: A solution has a volume of {sol_vol} mL with a solute concentration of {solute_pct}%. "
                "How many grams of solute are dissolved in the solution?"
            )
            gt = str(mass)
            cot = f"Calculation: {sol_vol} mL * {solute_pct}% = {mass} grams solute dissolved."
            v_type = "mathematical_constraint"
            v_cfg = {"target_value": float(mass), "tolerance": 1e-4}
            subdomain = "chemistry_solution_holdout"
        else:
            subdomain_type = seed % 4
            if subdomain_type == 0:
                init_apples = 24
                given_bob = 6
                sold_alice = (init_apples - given_bob) // 3
                left = (init_apples - given_bob) - sold_alice
                prompt = (
                    f"Word Problem #{seed}: Janet starts with {init_apples} apples. She gives {given_bob} to Bob, "
                    "and sells 1/3 of the remainder to Alice. How many apples does Janet have left?"
                )
                gt = str(left)
                cot = f"1. Initial: {init_apples}. 2. Minus Bob: {init_apples - given_bob}. 3. Minus 1/3: {left} apples."
                v_type = "mathematical_constraint"
                v_cfg = {"target_value": float(left), "tolerance": 1e-4}
                subdomain = "multi_step_word_problem"
            elif subdomain_type == 1:
                p_shirt = 25
                p_pants = 40
                discount = 15
                total = (p_shirt * 2 + p_pants) - discount  # 50 + 40 - 15 = 75
                prompt = (
                    f"Commerce Calculation #{seed}: A customer buys 2 shirts at ${p_shirt} each and 1 pair of pants for ${p_pants}. "
                    f"They apply a coupon for ${discount} off the total. What is the final total in dollars?"
                )
                gt = str(total)
                cot = f"Calculation: 2 * ${p_shirt} + ${p_pants} - ${discount} = ${total}."
                v_type = "mathematical_constraint"
                v_cfg = {"target_value": float(total), "tolerance": 1e-4}
                subdomain = "commerce_pricing"
            elif subdomain_type == 2:
                speed = 60
                hours = 2.5
                dist = int(speed * hours)  # 150
                prompt = (
                    f"Velocity Calculation #{seed}: A vehicle travels at a constant speed of {speed} miles per hour for {hours} hours. "
                    "How many miles does the vehicle travel?"
                )
                gt = str(dist)
                cot = f"Distance equation: {speed} mph * {hours} hours = {dist} miles."
                v_type = "mathematical_constraint"
                v_cfg = {"target_value": float(dist), "tolerance": 1e-4}
                subdomain = "rate_speed_distance"
            else:
                base = 80
                pct = 15
                result = int(base * (1 + pct / 100))  # 92
                prompt = (
                    f"Percentage Problem #{seed}: A stock price of ${base} increases by {pct}%. "
                    "What is the new stock price in dollars?"
                )
                gt = str(result)
                cot = f"Percentage calculation: ${base} * (1 + {pct}/100) = ${result}."
                v_type = "mathematical_constraint"
                v_cfg = {"target_value": float(result), "tolerance": 1e-4}
                subdomain = "percentage_ratio"

        return DistillationSample(
            id=f"arith_{subdomain}_{seed}",
            domain="arithmetic_reasoning",
            subdomain=subdomain,
            prompt=prompt,
            ground_truth=gt,
            teacher_cot=cot,
            target_solution=gt,
            verifier_type=v_type,
            verifier_config=v_cfg,
            difficulty=2,
            seed=seed,
            metadata={"split": split, **v_cfg},
        )


# ============================================================================
# Tier 1 Tests: Unit Contracts, Schema Integrity & Domain Coverage
# ============================================================================

def test_distillation_sample_dataclass_contract():
    """Verify DistillationSample serialization and roundtrip dictionary fidelity."""
    sample = ProceduralMultiDomainGenerator.generate_arithmetic(seed=999)
    d = sample.to_dict()
    assert isinstance(d, dict)
    assert d["id"] == sample.id
    assert d["domain"] == "arithmetic_reasoning"
    assert d["ground_truth"] == sample.ground_truth

    reconstructed = DistillationSample.from_dict(d)
    assert reconstructed.id == sample.id
    assert reconstructed.domain == sample.domain
    assert reconstructed.ground_truth == sample.ground_truth
    assert reconstructed.difficulty == sample.difficulty
    assert reconstructed.seed == sample.seed


def test_four_core_domains_generation():
    """Verify procedural task generation yields valid samples across all 4 core domains."""
    domains_tested = set()

    # 1. Multi-Step Reasoning
    s1 = ProceduralMultiDomainGenerator.generate_multistep_reasoning(1)
    assert s1.domain == "multi_step_reasoning"
    assert len(s1.ground_truth) > 0
    assert len(s1.prompt) > 20
    domains_tested.add(s1.domain)

    # 2. Constraint Satisfaction
    s2 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(0)
    assert s2.domain == "constraint_satisfaction"
    parsed_csp = json.loads(s2.ground_truth)
    assert "selected_instruments" in parsed_csp
    assert len(parsed_csp["selected_instruments"]) == 3
    domains_tested.add(s2.domain)

    # 3. Entity Disambiguation
    s3 = ProceduralMultiDomainGenerator.generate_entity_disambiguation(0)
    assert s3.domain == "entity_disambiguation"
    assert s3.ground_truth == "the grand piano"
    domains_tested.add(s3.domain)

    # 4. Arithmetic
    s4 = ProceduralMultiDomainGenerator.generate_arithmetic(0)
    assert s4.domain == "arithmetic_reasoning"
    assert s4.ground_truth == "12"
    domains_tested.add(s4.domain)

    assert len(domains_tested) == 4, f"Expected 4 distinct domains, got {len(domains_tested)}"


def test_verifier_ground_truth_pass_rates():
    """Verify 100% verifier pass rate on ground truth across all 4 domain generators."""
    samples = [
        ProceduralMultiDomainGenerator.generate_multistep_reasoning(10),
        ProceduralMultiDomainGenerator.generate_constraint_satisfaction(20),
        ProceduralMultiDomainGenerator.generate_entity_disambiguation(30),
        ProceduralMultiDomainGenerator.generate_arithmetic(40),
    ]

    for s in samples:
        # Construct temporary CognitiveTestCase to evaluate via deterministic verifier engine
        v_type = (
            VerifierType.JSON_SCHEMA if s.verifier_type == "json_schema"
            else (VerifierType.MATHEMATICAL_CONSTRAINT if s.verifier_type == "mathematical_constraint"
                  else VerifierType.EXACT_MATCH)
        )
        case = CognitiveTestCase(
            id=s.id,
            domain=s.domain,
            title=f"Test {s.id}",
            prompt=s.prompt,
            ground_truth=s.ground_truth,
            expected_constraints=[],
            verifier_type=v_type,
            metadata=s.verifier_config,
        )

        res = verify_test_case_result(case, s.ground_truth)
        assert res.passed is True, f"Ground truth for {s.id} failed verification! Details: {res.feedback}"
        assert res.score == 1.0, f"Expected perfect score 1.0 for {s.id}, got {res.score}"


def test_verifier_negative_distractor_rejection():
    """Verify programmatic verifiers correctly reject invalid / distractor answers."""
    # 1. Multi-Step: Distractor suspect Alice should fail
    s1 = ProceduralMultiDomainGenerator.generate_multistep_reasoning(1)
    case1 = CognitiveTestCase(
        id=s1.id, domain=s1.domain, title="Alibi", prompt=s1.prompt,
        ground_truth=s1.ground_truth, expected_constraints=[],
        verifier_type=VerifierType.EXACT_MATCH,
    )
    res_wrong = verify_test_case_result(case1, "Alice")
    assert res_wrong.passed is False
    assert res_wrong.score == 0.0

    # 2. Constraint: Over-budget payload should fail
    s2 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(0)
    case2 = CognitiveTestCase(
        id=s2.id, domain=s2.domain, title="Payload", prompt=s2.prompt,
        ground_truth=s2.ground_truth, expected_constraints=[],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata=s2.verifier_config,
    )
    res_over = verify_test_case_result(case2, json.dumps({"selected_instruments": ["Spectrometer", "Radar"]}))
    assert res_over.passed is False

    # 3. Arithmetic: Wrong numerical answer should fail
    s4 = ProceduralMultiDomainGenerator.generate_arithmetic(0)
    case4 = CognitiveTestCase(
        id=s4.id, domain=s4.domain, title="Apples", prompt=s4.prompt,
        ground_truth=s4.ground_truth, expected_constraints=[],
        verifier_type=VerifierType.MATHEMATICAL_CONSTRAINT,
        metadata=s4.verifier_config,
    )
    res_num_wrong = verify_test_case_result(case4, "18")
    assert res_num_wrong.passed is False
    assert res_num_wrong.score == 0.0


# ============================================================================
# Tier 2 Tests: Split Disjointness, Bounds, and Adversarial Resilience
# ============================================================================

def _compute_8gram_set(text: str) -> set[str]:
    """Compute normalized word 8-grams."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 8:
        return {" ".join(words)}
    return {" ".join(words[i:i + 8]) for i in range(len(words) - 7)}


def test_split_generation_and_zero_contamination():
    """Verify 80/10/10 split generation with zero prompt hash collision and <15% n-gram leakage."""
    # 80 Train samples (seeds 1000..1019)
    train_samples: list[DistillationSample] = []
    for i in range(20):
        train_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=1000 + i, split="train"))
        train_samples.append(ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=2000 + i, split="train"))
        train_samples.append(ProceduralMultiDomainGenerator.generate_entity_disambiguation(seed=3000 + i, split="train"))
        train_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=4000 + i, split="train"))

    # 10 Validation samples (seeds 1020..1022)
    val_samples: list[DistillationSample] = []
    for i in range(2):
        val_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=1020 + i, split="train"))
        val_samples.append(ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=2020 + i, split="train"))
        val_samples.append(ProceduralMultiDomainGenerator.generate_entity_disambiguation(seed=3020 + i, split="train"))
        val_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=4020 + i, split="train"))
    val_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=1023, split="train"))
    val_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=4023, split="train"))

    # 10 Holdout Test samples (distinct holdout domain tasks, split="test")
    test_samples: list[DistillationSample] = []
    for i in range(2):
        test_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=5000 + i, split="test"))
        test_samples.append(ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=6000 + i, split="test"))
        test_samples.append(ProceduralMultiDomainGenerator.generate_entity_disambiguation(seed=7000 + i, split="test"))
        test_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=8000 + i, split="test"))
    test_samples.append(ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=5003, split="test"))
    test_samples.append(ProceduralMultiDomainGenerator.generate_arithmetic(seed=8003, split="test"))

    assert len(train_samples) == 80
    assert len(val_samples) == 10
    assert len(test_samples) == 10

    # 1. Exact Prompt Hash Uniqueness across splits
    train_hashes = {hashlib.sha256(s.prompt.encode("utf-8")).hexdigest() for s in train_samples}
    val_hashes = {hashlib.sha256(s.prompt.encode("utf-8")).hexdigest() for s in val_samples}
    test_hashes = {hashlib.sha256(s.prompt.encode("utf-8")).hexdigest() for s in test_samples}

    assert len(train_hashes.intersection(val_hashes)) == 0, "Train and Val splits share identical prompts!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Train and Test splits share identical prompts!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Val and Test splits share identical prompts!"

    # 2. 8-Gram Jaccard Overlap < 15%
    train_grams: set[str] = set()
    for s in train_samples:
        train_grams.update(_compute_8gram_set(s.prompt))

    test_grams: set[str] = set()
    for s in test_samples:
        test_grams.update(_compute_8gram_set(s.prompt))

    intersection = len(train_grams.intersection(test_grams))
    union = len(train_grams.union(test_grams))
    jaccard = (intersection / union) if union > 0 else 0.0

    assert jaccard < 0.15, f"8-gram Jaccard similarity ({jaccard:.3f}) exceeds 15% maximum allowed leakage!"


def test_token_and_text_length_bounds():
    """Verify prompt, solution, and CoT length bounds across generated records."""
    generators = [
        ProceduralMultiDomainGenerator.generate_multistep_reasoning,
        ProceduralMultiDomainGenerator.generate_constraint_satisfaction,
        ProceduralMultiDomainGenerator.generate_entity_disambiguation,
        ProceduralMultiDomainGenerator.generate_arithmetic,
    ]

    for gen in generators:
        sample = gen(seed=42)
        words_prompt = len(sample.prompt.split())
        words_solution = len(sample.target_solution.split())
        words_cot = len(sample.teacher_cot.split()) if sample.teacher_cot else 0

        assert 5 <= words_prompt <= 512, f"Prompt length out of bounds: {words_prompt} words in {sample.id}"
        assert 1 <= words_solution <= 64, f"Solution length out of bounds: {words_solution} words in {sample.id}"
        assert 5 <= words_cot <= 384, f"CoT length out of bounds: {words_cot} words in {sample.id}"


def test_jsonl_batch_io_roundtrip(tmp_path):
    """Verify serialization of dataset records to JSONL file and roundtrip deserialization."""
    samples = [
        ProceduralMultiDomainGenerator.generate_multistep_reasoning(seed=501),
        ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=502),
        ProceduralMultiDomainGenerator.generate_entity_disambiguation(seed=503),
        ProceduralMultiDomainGenerator.generate_arithmetic(seed=504),
    ]

    jsonl_path = tmp_path / "dataset_train.jsonl"

    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")

    # Read JSONL
    loaded_samples: list[DistillationSample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                loaded_samples.append(DistillationSample.from_dict(json.loads(line)))

    assert len(loaded_samples) == len(samples)
    for orig, loaded in zip(samples, loaded_samples):
        assert orig.id == loaded.id
        assert orig.domain == loaded.domain
        assert orig.ground_truth == loaded.ground_truth
        assert orig.verifier_type == loaded.verifier_type


def test_adversarial_malformed_records():
    """Verify robust error handling when encountering corrupt or missing dataset fields."""
    # 1. Missing required field 'prompt'
    bad_dict = {"id": "bad_01", "domain": "math", "ground_truth": "42"}
    with pytest.raises(KeyError):
        DistillationSample.from_dict(bad_dict)

    # 2. Missing 'id'
    bad_dict_2 = {"prompt": "What is 1+1?", "domain": "math", "ground_truth": "2"}
    with pytest.raises(KeyError):
        DistillationSample.from_dict(bad_dict_2)


def test_deterministic_seed_reproducibility():
    """Verify identical random seeds produce byte-for-byte identical dataset samples."""
    s1 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=777)
    s2 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=777)
    s3 = ProceduralMultiDomainGenerator.generate_constraint_satisfaction(seed=888)

    assert s1.to_dict() == s2.to_dict(), "Identical seeds must produce identical data samples."
    assert s1.id != s3.id or s1.prompt != s3.prompt, "Different seeds must produce distinct data samples."


def test_all_sixteen_subdomains_generation_and_verification():
    """Verify all 16 distinct subdomains across 4 domains generate valid verifiable samples."""
    subdomains_found = set()
    for domain_fn in [
        ProceduralMultiDomainGenerator.generate_multistep_reasoning,
        ProceduralMultiDomainGenerator.generate_constraint_satisfaction,
        ProceduralMultiDomainGenerator.generate_entity_disambiguation,
        ProceduralMultiDomainGenerator.generate_arithmetic,
    ]:
        for sub_seed in range(4):
            sample = domain_fn(seed=sub_seed, split="train")
            subdomains_found.add(sample.subdomain)

            # Check verification
            v_type = (
                VerifierType.JSON_SCHEMA if sample.verifier_type == "json_schema"
                else (VerifierType.MATHEMATICAL_CONSTRAINT if sample.verifier_type == "mathematical_constraint"
                      else VerifierType.EXACT_MATCH)
            )
            case = CognitiveTestCase(
                id=sample.id,
                domain=sample.domain,
                title=f"Subdomain {sample.subdomain}",
                prompt=sample.prompt,
                ground_truth=sample.ground_truth,
                expected_constraints=[],
                verifier_type=v_type,
                metadata=sample.verifier_config,
            )
            res = verify_test_case_result(case, sample.ground_truth)
            assert res.passed is True, f"Subdomain {sample.subdomain} failed verification: {res.feedback}"

    assert len(subdomains_found) == 16, f"Expected 16 distinct subdomains, got {len(subdomains_found)}: {subdomains_found}"


def test_mathematical_tolerance_boundaries():
    """Verify mathematical verifier handles exact matches, distinct numbers, and negative numbers."""
    case = CognitiveTestCase(
        id="math_tol_01",
        domain=DomainType.MULTI_CONSTRAINT,
        title="Tolerance Test",
        prompt="Calculate value",
        ground_truth="100.0",
        expected_constraints=[],
        verifier_type=VerifierType.MATHEMATICAL_CONSTRAINT,
        metadata={"target_value": 100.0, "tolerance": 1e-3},
    )

    # Exact match
    assert verify_test_case_result(case, "100.0").passed is True
    # Different numbers should fail
    assert verify_test_case_result(case, "250.0").passed is False
    assert verify_test_case_result(case, "99.0").passed is False
    # Negative number
    assert verify_test_case_result(case, "-50.0").passed is False


def test_empty_and_whitespace_prompt_handling():
    """Verify behavior on empty or whitespace-only inputs."""
    case = CognitiveTestCase(
        id="ws_01",
        domain=DomainType.MULTI_CONSTRAINT,
        title="Whitespace Test",
        prompt="   \n\t  ",
        ground_truth="answer",
        expected_constraints=[],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={"expected_entity": "answer"},
    )
    # Empty response should fail
    assert verify_test_case_result(case, "").passed is False
    assert verify_test_case_result(case, "   ").passed is False
    # Correct response should pass even if prompt was whitespace
    assert verify_test_case_result(case, "answer").passed is True


def test_unicode_and_latex_math_in_prompts():
    """Verify prompts containing Unicode, LaTeX math expressions, and special characters."""
    latex_prompt = r"Calculate $\Delta E = \sum_{i=1}^3 m_i c^2$ where $m_1=2, m_2=3, m_3=5$ and $c=3 \times 10^8$ m/s."
    sample = DistillationSample(
        id="latex_001",
        domain="arithmetic_reasoning",
        subdomain="physics_math",
        prompt=latex_prompt,
        ground_truth="9.0e17",
        teacher_cot=r"1. Sum masses: 2 + 3 + 5 = 10 kg. 2. c^2 = 9e16. 3. 10 * 9e16 = 9e17.",
        target_solution="9.0e17",
        verifier_type="mathematical_constraint",
        verifier_config={"target_value": 9.0e17, "tolerance": 1e12},
    )
    d = sample.to_dict()
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert r"\Delta" in deserialized["prompt"]
    assert r"\sum" in deserialized["prompt"]
