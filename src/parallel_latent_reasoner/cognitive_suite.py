"""Native Cognitive Domain Benchmark Suite for Parallel Latent Reasoner (PRLR).

Curated empirical benchmark dataset and deterministic programmatic scoring rubrics
across 5 core cognitive domains where continuous latent deliberation excels:
1. Multi-Constraint Satisfaction (MCS)
2. Winograd Schema & Pronoun Disambiguation (WSD)
3. Semantic Denoising & Noisy Intent Extraction (SDN)
4. Cross-Context Multi-Clue Synthesis (CMS)
5. Action & Tool Routing (ATR)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class DomainType(str, Enum):
    """Supported cognitive domains for large-scale Gemma 4 evaluation."""

    MULTI_CONSTRAINT = "multi_constraint"
    WINOGRAD_SCHEMA = "winograd_schema"
    SEMANTIC_DENOISING = "semantic_denoising"
    MULTI_CLUE_SYNTHESIS = "multi_clue_synthesis"
    ACTION_TOOL_ROUTING = "action_tool_routing"

    @classmethod
    def from_str(cls, value: str) -> "DomainType":
        val = value.strip().lower()
        for member in cls:
            if member.value == val:
                return member
        raise ValueError(f"Unknown DomainType: {value}")


class VerifierType(str, Enum):
    """Deterministic verifier categories."""

    JSON_SCHEMA = "json_schema"
    EXACT_MATCH = "exact_match"
    REGEX_CONSTRAINT = "regex_constraint"
    MATHEMATICAL_CONSTRAINT = "mathematical_constraint"
    SEMANTIC_CONTAINS = "semantic_contains"


@dataclass
class EvaluationResult:
    """Outcome of programmatic test case verification.

    Supports tuple unpacking: `passed, score, details = result`
    for compatibility with multiple evaluation interfaces.
    """

    passed: bool
    score: float
    feedback: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.passed
        yield self.score
        yield self.details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "feedback": self.feedback,
            "details": self.details,
        }


@dataclass
class CognitiveTestCase:
    """Benchmark test case definition with deterministic scoring rules."""

    id: str
    domain: Union[str, DomainType]
    title: str
    prompt: str
    ground_truth: str
    expected_constraints: List[str]
    verifier_type: Union[str, VerifierType]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": str(self.domain.value if isinstance(self.domain, DomainType) else self.domain),
            "title": self.title,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "expected_constraints": list(self.expected_constraints),
            "verifier_type": str(self.verifier_type.value if isinstance(self.verifier_type, VerifierType) else self.verifier_type),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveTestCase":
        return cls(
            id=data["id"],
            domain=data["domain"],
            title=data["title"],
            prompt=data["prompt"],
            ground_truth=data["ground_truth"],
            expected_constraints=data.get("expected_constraints", []),
            verifier_type=data.get("verifier_type", VerifierType.EXACT_MATCH),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Deterministic Programmatic Verifiers
# ============================================================================

def _extract_json_payload(text: str) -> Optional[Any]:
    """Robustly extract and parse a JSON object or list from arbitrary text."""
    clean = text.strip()
    # 1. Direct parse attempt
    try:
        return json.loads(clean)
    except Exception:
        pass

    # 2. Markdown code block extraction
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, re.IGNORECASE)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except Exception:
            pass

    # 3. Outermost bracket extraction
    brace_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean)
    if brace_match:
        try:
            return json.loads(brace_match.group(0).strip())
        except Exception:
            pass

    return None


def _verify_json_schema(
    expected_fields: Dict[str, Any],
    required_keys: List[str],
    response_text: str,
) -> EvaluationResult:
    """Verify JSON structure and key-value correctness."""
    parsed = _extract_json_payload(response_text)
    if parsed is None or not isinstance(parsed, dict):
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback="Response does not contain a valid JSON object.",
            details={"raw_text": response_text},
        )

    # Normalize parsed dictionary keys to lower-case for lookup tolerance
    parsed_norm = {k.strip().lower(): (k, v) for k, v in parsed.items()}

    matched_keys = []
    missing_keys = []
    mismatched_values = []

    for req_key in required_keys:
        req_norm = req_key.strip().lower()
        if req_norm not in parsed_norm:
            missing_keys.append(req_key)
            continue

        orig_k, val = parsed_norm[req_norm]
        matched_keys.append(orig_k)

        # Value comparison if key exists in expected_fields
        if req_key in expected_fields:
            exp_val = expected_fields[req_key]
            if isinstance(exp_val, str) and isinstance(val, str):
                if exp_val.strip().lower() != val.strip().lower():
                    mismatched_values.append(
                        {"key": req_key, "expected": exp_val, "actual": val}
                    )
            elif isinstance(exp_val, (int, float)) and isinstance(val, (int, float)):
                if abs(float(exp_val) - float(val)) > 1e-4:
                    mismatched_values.append(
                        {"key": req_key, "expected": exp_val, "actual": val}
                    )
            elif isinstance(exp_val, dict) and isinstance(val, dict):
                # Check nested dict values
                for sub_k, sub_exp in exp_val.items():
                    sub_val = val.get(sub_k)
                    if isinstance(sub_exp, str) and isinstance(sub_val, str):
                        if sub_exp.strip().lower() != sub_val.strip().lower():
                            mismatched_values.append(
                                {"key": f"{req_key}.{sub_k}", "expected": sub_exp, "actual": sub_val}
                            )
                    elif sub_exp != sub_val:
                        mismatched_values.append(
                            {"key": f"{req_key}.{sub_k}", "expected": sub_exp, "actual": sub_val}
                        )
            elif exp_val != val:
                mismatched_values.append(
                    {"key": req_key, "expected": exp_val, "actual": val}
                )

    if missing_keys or mismatched_values:
        feedback = f"Missing keys: {missing_keys}; Mismatches: {mismatched_values}"
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback=feedback,
            details={
                "parsed": parsed,
                "missing_keys": missing_keys,
                "mismatched_values": mismatched_values,
            },
        )

    return EvaluationResult(
        passed=True,
        score=1.0,
        feedback="All required JSON fields and values matched successfully.",
        details={"parsed": parsed},
    )


def _verify_exact_entity(
    valid_patterns: List[str],
    invalid_patterns: List[str],
    response_text: str,
) -> EvaluationResult:
    """Verify entity referents, preventing false positives from mentioning other candidates."""
    norm = response_text.strip().lower()

    # Clean leading answer prefixes
    norm_clean = re.sub(r"^(answer|referent|culprit|entity|result)\s*[:=-]\s*", "", norm, flags=re.IGNORECASE)

    has_valid = any(
        re.search(r"\b" + re.escape(v.lower()) + r"\b", norm_clean)
        for v in valid_patterns
    )
    has_invalid = any(
        re.search(r"\b" + re.escape(inv.lower()) + r"\b", norm_clean)
        for inv in invalid_patterns
    )

    if has_valid and not has_invalid:
        return EvaluationResult(
            passed=True,
            score=1.0,
            feedback=f"Correct entity identified without conflicting entities.",
            details={"text": response_text, "matched": valid_patterns},
        )
    elif has_valid and has_invalid:
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback=f"Conflicting entities detected in output.",
            details={"text": response_text, "valid": valid_patterns, "invalid_found": invalid_patterns},
        )
    else:
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback=f"Target entity not found in response.",
            details={"text": response_text, "expected_any": valid_patterns},
        )


def _verify_mcs_01_spacecraft(response_text: str) -> EvaluationResult:
    """Verify spacecraft payload scheduling constraints (Alpha..Epsilon)."""
    instruments = {
        "alpha": {"mass": 12, "power": 45, "data": 10, "zone": "A"},
        "beta": {"mass": 18, "power": 60, "data": 25, "zone": "B"},
        "gamma": {"mass": 15, "power": 35, "data": 15, "zone": "A"},
        "delta": {"mass": 22, "power": 80, "data": 30, "zone": "B"},
        "epsilon": {"mass": 8, "power": 20, "data": 5, "zone": "A"},
    }
    norm = response_text.strip().lower()
    selected = [k for k in instruments if re.search(r"\b" + k + r"\b", norm)]

    if not selected:
        return EvaluationResult(passed=False, score=0.0, feedback="No valid instruments identified in output.")

    total_mass = sum(instruments[k]["mass"] for k in selected)
    total_power = sum(instruments[k]["power"] for k in selected)
    total_data = sum(instruments[k]["data"] for k in selected)
    zones = {instruments[k]["zone"] for k in selected}

    if total_mass > 40:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Mass constraint violated: {total_mass}kg > 40kg")
    if total_power > 110:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Power constraint violated: {total_power}W > 110W")
    if total_data < 30:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Data constraint violated: {total_data}Mbps < 30Mbps")
    if "A" not in zones or "B" not in zones:
        return EvaluationResult(passed=False, score=0.0, feedback="Thermal zone constraint violated: Must include Zone A and Zone B")

    # Optimal solution maximizes data rate (Beta + Gamma = 40 Mbps)
    if total_data < 40:
        return EvaluationResult(
            passed=False,
            score=0.5,
            feedback=f"Valid but suboptimal data rate ({total_data} Mbps vs optimal 40 Mbps).",
            details={"selected": selected, "total_data": total_data},
        )

    return EvaluationResult(
        passed=True,
        score=1.0,
        feedback="Optimal payload selected satisfying all 5 constraints (Beta, Gamma).",
        details={"selected": selected, "mass": total_mass, "power": total_power, "data": total_data},
    )


def _verify_mcs_02_pangrammatic(response_text: str) -> EvaluationResult:
    """Verify 7-word constrained sentence without letter 'o'."""
    text = response_text.strip()
    # Strip quotes if present
    text = re.sub(r"^[\"']|[\"']$", "", text).strip()
    words = [w for w in re.split(r"\s+", text) if w]

    if len(words) != 7:
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback=f"Word count constraint violated: {len(words)} words (expected exactly 7).",
            details={"words": words},
        )

    first_word = re.sub(r"[^\w]", "", words[0]).lower()
    if not first_word.endswith("ly"):
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback=f"First word constraint violated: '{words[0]}' does not end in '-ly'.",
        )

    last_word = re.sub(r"[^\w]", "", words[-1]).lower()
    # Check plural noun pattern (ends in 's', 'es', or 'men', 'feet', 'children', 'puzzles', etc.)
    if not (last_word.endswith("s") or last_word in ["children", "people", "men", "women", "feet", "teeth"]):
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback=f"Last word constraint violated: '{words[-1]}' is not a plural noun.",
        )

    clean_all = text.lower()
    for char in ["k", "x", "z"]:
        if char not in clean_all:
            return EvaluationResult(
                passed=False,
                score=0.0,
                feedback=f"Letter inclusion constraint violated: '{char}' missing.",
            )

    if "o" in clean_all:
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback="Forbidden letter constraint violated: Contains letter 'o'.",
        )

    return EvaluationResult(
        passed=True,
        score=1.0,
        feedback="Sentence successfully satisfies all 5 lexical and structural constraints.",
        details={"sentence": text, "words": words},
    )


def _verify_mcs_03_itinerary(response_text: str) -> EvaluationResult:
    """Verify conference budget itinerary (A1..D2)."""
    activities = {
        "A1": {"cost": 200, "co2": 10, "flight": 0, "tech": 1},
        "A2": {"cost": 100, "co2": 5, "flight": 0, "tech": 0},
        "B1": {"cost": 300, "co2": 40, "flight": 1, "tech": 1},
        "B2": {"cost": 50, "co2": 2, "flight": 0, "tech": 0},
        "C1": {"cost": 150, "co2": 15, "flight": 0, "tech": 1},
        "C2": {"cost": 250, "co2": 35, "flight": 1, "tech": 1},
        "D1": {"cost": 250, "co2": 20, "flight": 0, "tech": 1},
        "D2": {"cost": 100, "co2": 8, "flight": 0, "tech": 0},
    }
    found = re.findall(r"\b([A-D][1-2])\b", response_text.upper())
    if len(found) < 4:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Found {len(found)} activity codes, expected 4.")

    chosen = found[:4]
    cost = sum(activities[c]["cost"] for c in chosen)
    co2 = sum(activities[c]["co2"] for c in chosen)
    flights = sum(activities[c]["flight"] for c in chosen)
    tech = sum(activities[c]["tech"] for c in chosen)

    if cost > 600:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Budget violated: ${cost} > $600.")
    if co2 > 50:
        return EvaluationResult(passed=False, score=0.0, feedback=f"CO2 violated: {co2}kg > 50kg.")
    if flights > 1:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Flight limit violated: {flights} > 1.")
    if not ("A1" in chosen or "C1" in chosen):
        return EvaluationResult(passed=False, score=0.0, feedback="Must include A1 or C1.")

    if chosen != ["A1", "B2", "C1", "D2"]:
        return EvaluationResult(
            passed=False,
            score=0.5,
            feedback=f"Valid but suboptimal itinerary ({chosen} with {tech} tech events).",
            details={"chosen": chosen, "cost": cost, "co2": co2},
        )

    return EvaluationResult(
        passed=True,
        score=1.0,
        feedback="Optimal itinerary selected (A1, B2, C1, D2).",
        details={"chosen": chosen, "cost": cost, "co2": co2, "tech_events": tech},
    )


def _verify_mcs_04_cryptarithm(response_text: str) -> EvaluationResult:
    """Verify cryptarithm integer assignment W, X, Y, Z."""
    matches = re.findall(r"([WXYZ])\s*[:=]\s*(\d+)", response_text.upper())
    assignments = {var: int(val) for var, val in matches}

    for req in ["W", "X", "Y", "Z"]:
        if req not in assignments:
            return EvaluationResult(
                passed=False,
                score=0.0,
                feedback=f"Variable {req} missing in assignment.",
                details={"assignments": assignments},
            )

    w, x, y, z = assignments["W"], assignments["X"], assignments["Y"], assignments["Z"]
    digits = [w, x, y, z]

    if any(d < 1 or d > 9 for d in digits):
        return EvaluationResult(passed=False, score=0.0, feedback="Digits must be between 1 and 9.")
    if len(set(digits)) != 4:
        return EvaluationResult(passed=False, score=0.0, feedback="All 4 digits must be distinct.")
    if w + x != y + z + 1:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Equation 1 violated: {w}+{x} != {y}+{z}+1.")
    if w * x != y * z + 8:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Equation 2 violated: {w}*{x} != {y}*{z}+8.")
    if not (w > x):
        return EvaluationResult(passed=False, score=0.0, feedback=f"Inequality W > X violated: {w} not > {x}.")
    if not (y < z):
        return EvaluationResult(passed=False, score=0.0, feedback=f"Inequality Y < Z violated: {y} not < {z}.")

    return EvaluationResult(
        passed=True,
        score=1.0,
        feedback="Cryptarithm assignment W=5, X=4, Y=2, Z=6 verified exactly.",
        details=assignments,
    )


def _verify_mcs_05_traffic_shaper(response_text: str) -> EvaluationResult:
    """Verify microservice QoS path routing {P1..P5}."""
    paths = {
        "P1": {"lat": 15, "bw": 100, "loss": 0.1, "cost": 10},
        "P2": {"lat": 25, "bw": 250, "loss": 0.5, "cost": 20},
        "P3": {"lat": 10, "bw": 80, "loss": 0.05, "cost": 15},
        "P4": {"lat": 40, "bw": 400, "loss": 1.2, "cost": 25},
        "P5": {"lat": 20, "bw": 150, "loss": 0.2, "cost": 10},
    }
    found = sorted(list(set(re.findall(r"\b(P[1-5])\b", response_text.upper()))))
    if len(found) != 3:
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback=f"Selected {len(found)} paths ({found}), expected exactly 3 distinct paths.",
        )

    if "P3" not in found:
        return EvaluationResult(passed=False, score=0.0, feedback="Path P3 is mandatory for telemetry.")

    tot_bw = sum(paths[p]["bw"] for p in found)
    avg_lat = sum(paths[p]["lat"] for p in found) / 3.0
    max_loss = max(paths[p]["loss"] for p in found)
    tot_cost = sum(paths[p]["cost"] for p in found)

    if tot_bw < 450:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Bandwidth violated: {tot_bw} < 450 Mbps.")
    if avg_lat > 20.0:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Latency violated: {avg_lat:.2f}ms > 20ms.")
    if max_loss > 0.6:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Packet loss violated: {max_loss}% > 0.6%.")
    if tot_cost > 45:
        return EvaluationResult(passed=False, score=0.0, feedback=f"Cost violated: ${tot_cost} > $45.")

    if found != ["P2", "P3", "P5"]:
        return EvaluationResult(passed=False, score=0.5, feedback="Valid but suboptimal path selection.")

    return EvaluationResult(
        passed=True,
        score=1.0,
        feedback="Optimal QoS network paths selected (P2, P3, P5).",
        details={"paths": found, "bw": tot_bw, "avg_lat": avg_lat, "cost": tot_cost},
    )


def verify_test_case_result(
    test_case: CognitiveTestCase,
    response_text: str,
) -> EvaluationResult:
    """Evaluate an LLM response against a test case using deterministic rules."""
    case_id = test_case.id.lower()

    # Specialized custom verifiers for MCS domain
    if case_id == "mcs_01":
        return _verify_mcs_01_spacecraft(response_text)
    elif case_id == "mcs_02":
        return _verify_mcs_02_pangrammatic(response_text)
    elif case_id == "mcs_03":
        return _verify_mcs_03_itinerary(response_text)
    elif case_id == "mcs_04":
        return _verify_mcs_04_cryptarithm(response_text)
    elif case_id == "mcs_05":
        return _verify_mcs_05_traffic_shaper(response_text)

    verifier = test_case.verifier_type
    if isinstance(verifier, str):
        verifier = VerifierType(verifier)

    if verifier == VerifierType.JSON_SCHEMA:
        expected_fields = test_case.metadata.get("expected_fields", {})
        required_keys = test_case.metadata.get("required_keys", list(expected_fields.keys()))
        return _verify_json_schema(expected_fields, required_keys, response_text)

    elif verifier in (VerifierType.EXACT_MATCH, VerifierType.SEMANTIC_CONTAINS):
        valid_patterns = test_case.metadata.get("valid_patterns", [test_case.ground_truth])
        invalid_patterns = test_case.metadata.get("invalid_patterns", [])
        return _verify_exact_entity(valid_patterns, invalid_patterns, response_text)

    elif verifier == VerifierType.REGEX_CONSTRAINT:
        required_regex = test_case.metadata.get("required_regex", [])
        forbidden_regex = test_case.metadata.get("forbidden_regex", [])
        for r in required_regex:
            if not re.search(r, response_text, re.IGNORECASE):
                return EvaluationResult(
                    passed=False,
                    score=0.0,
                    feedback=f"Required regex pattern '{r}' not matched.",
                )
        for fr in forbidden_regex:
            if re.search(fr, response_text, re.IGNORECASE):
                return EvaluationResult(
                    passed=False,
                    score=0.0,
                    feedback=f"Forbidden regex pattern '{fr}' was matched.",
                )
        return EvaluationResult(passed=True, score=1.0, feedback="All regex constraints satisfied.")

    # Fallback to simple containment
    clean_gt = test_case.ground_truth.strip().lower()
    clean_resp = response_text.strip().lower()
    if clean_gt in clean_resp:
        return EvaluationResult(passed=True, score=1.0, feedback="Ground truth substring matched.")
    return EvaluationResult(passed=False, score=0.0, feedback="Ground truth not found in response.")


# ============================================================================
# Benchmark Suite Test Cases Definition (25 Curated Benchmark Cases)
# ============================================================================

_BENCHMARK_CASES: List[CognitiveTestCase] = [
    # ------------------------------------------------------------------------
    # Domain 1: Multi-Constraint Satisfaction (5 Cases)
    # ------------------------------------------------------------------------
    CognitiveTestCase(
        id="mcs_01",
        domain=DomainType.MULTI_CONSTRAINT,
        title="Orbital Spacecraft Payload Optimization",
        prompt=(
            "You are an orbital payload scheduler. Select a subset of instruments to activate from:\n"
            "- Alpha (Mass: 12kg, Power: 45W, Data: 10Mbps, Thermal Zone: A)\n"
            "- Beta (Mass: 18kg, Power: 60W, Data: 25Mbps, Thermal Zone: B)\n"
            "- Gamma (Mass: 15kg, Power: 35W, Data: 15Mbps, Thermal Zone: A)\n"
            "- Delta (Mass: 22kg, Power: 80W, Data: 30Mbps, Thermal Zone: B)\n"
            "- Epsilon (Mass: 8kg, Power: 20W, Data: 5Mbps, Thermal Zone: A)\n\n"
            "Constraints:\n"
            "1. Total Mass must NOT exceed 40 kg.\n"
            "2. Total Power must NOT exceed 110 W.\n"
            "3. Total Data Rate must be at least 30 Mbps.\n"
            "4. At least one instrument from Thermal Zone A and at least one from Thermal Zone B must be active.\n"
            "5. Maximize the total Data Rate.\n\n"
            "Output ONLY the exact list of chosen instrument names separated by commas (e.g. \"Alpha, Beta\")."
        ),
        ground_truth="Beta, Gamma",
        expected_constraints=[
            "Total mass <= 40 kg",
            "Total power <= 110 W",
            "Total data >= 30 Mbps",
            "At least one Zone A and one Zone B",
            "Maximal data rate (40 Mbps)",
        ],
        verifier_type=VerifierType.MATHEMATICAL_CONSTRAINT,
        metadata={"valid_subset": ["Beta", "Gamma"], "optimal_data": 40},
    ),
    CognitiveTestCase(
        id="mcs_02",
        domain=DomainType.MULTI_CONSTRAINT,
        title="Constrained Pangrammatic Sentence Generation",
        prompt=(
            "Construct a single meaningful English sentence satisfying ALL of the following 5 constraints simultaneously:\n"
            "1. Contains exactly 7 words.\n"
            "2. Starts with an adverb ending in '-ly'.\n"
            "3. Ends with a plural noun.\n"
            "4. Contains the letters 'k', 'x', and 'z' somewhere in the sentence.\n"
            "5. Does NOT contain the letter 'o' anywhere in any word.\n\n"
            "Output ONLY the 7-word sentence without explanation."
        ),
        ground_truth="Quickly six black wizards fix tiny puzzles",
        expected_constraints=[
            "Exactly 7 words",
            "Starts with adverb in -ly",
            "Ends with plural noun",
            "Contains 'k', 'x', 'z'",
            "No letter 'o'",
        ],
        verifier_type=VerifierType.MATHEMATICAL_CONSTRAINT,
        metadata={"target_word_count": 7, "forbidden_letters": ["o"], "required_letters": ["k", "x", "z"]},
    ),
    CognitiveTestCase(
        id="mcs_03",
        domain=DomainType.MULTI_CONSTRAINT,
        title="Conference Budget & Carbon Itinerary Optimizer",
        prompt=(
            "Plan a 4-day conference itinerary choosing 1 activity per day from:\n"
            "- Day 1: [A1: Workshop ($200, 10kg CO2), A2: Keynote ($100, 5kg CO2)]\n"
            "- Day 2: [B1: Lab Tour ($300, 40kg CO2, includes flight), B2: Virtual Expo ($50, 2kg CO2)]\n"
            "- Day 3: [C1: Hackathon ($150, 15kg CO2), C2: Site Visit ($250, 35kg CO2, includes flight)]\n"
            "- Day 4: [D1: Gala Dinner ($250, 20kg CO2), D2: Networking Lunch ($100, 8kg CO2)]\n\n"
            "Constraints:\n"
            "1. Total cost must be <= $600.\n"
            "2. Total CO2 must be <= 50 kg.\n"
            "3. Total flights must be <= 1 (activities B1 and C2 include flights).\n"
            "4. Must select at least one of [A1, C1].\n"
            "5. Maximize total number of in-person technical events (A1, B1, C1, C2, D1).\n\n"
            "Output ONLY the chosen 4 activity codes in order separated by commas (e.g. \"A1, B2, C1, D2\")."
        ),
        ground_truth="A1, B2, C1, D2",
        expected_constraints=[
            "Cost <= $600",
            "CO2 <= 50 kg",
            "Flights <= 1",
            "Includes A1 or C1",
            "Maximized technical events",
        ],
        verifier_type=VerifierType.MATHEMATICAL_CONSTRAINT,
        metadata={"expected_codes": ["A1", "B2", "C1", "D2"]},
    ),
    CognitiveTestCase(
        id="mcs_04",
        domain=DomainType.MULTI_CONSTRAINT,
        title="Cryptarithm Modular Diophantine Logic",
        prompt=(
            "Find single distinct non-zero decimal digits (1-9) for variables W, X, Y, Z such that:\n"
            "1. W + X = Y + Z + 1\n"
            "2. W * X = Y * Z + 8\n"
            "3. W > X\n"
            "4. Y < Z\n"
            "5. All four variables W, X, Y, Z are distinct digits from {1, 2, 3, 4, 5, 6, 7, 8, 9}.\n\n"
            "Output ONLY the assignment in the format: \"W=?, X=?, Y=?, Z=?\" (e.g. \"W=5, X=4, Y=2, Z=6\")."
        ),
        ground_truth="W=5, X=4, Y=2, Z=6",
        expected_constraints=[
            "W + X = Y + Z + 1",
            "W * X = Y * Z + 8",
            "W > X and Y < Z",
            "Distinct non-zero single digits",
        ],
        verifier_type=VerifierType.MATHEMATICAL_CONSTRAINT,
        metadata={"expected_assignment": {"W": 5, "X": 4, "Y": 2, "Z": 6}},
    ),
    CognitiveTestCase(
        id="mcs_05",
        domain=DomainType.MULTI_CONSTRAINT,
        title="Microservice QoS Traffic Shaper",
        prompt=(
            "Route video traffic across 3 network paths chosen from {P1, P2, P3, P4, P5} with attributes:\n"
            "- P1: Latency 15ms, Bandwidth 100Mbps, Packet Loss 0.1%, Cost $10/hr\n"
            "- P2: Latency 25ms, Bandwidth 250Mbps, Packet Loss 0.5%, Cost $20/hr\n"
            "- P3: Latency 10ms, Bandwidth 80Mbps, Packet Loss 0.05%, Cost $15/hr\n"
            "- P4: Latency 40ms, Bandwidth 400Mbps, Packet Loss 1.2%, Cost $25/hr\n"
            "- P5: Latency 20ms, Bandwidth 150Mbps, Packet Loss 0.2%, Cost $10/hr\n\n"
            "Select exactly 3 distinct paths satisfying:\n"
            "1. Total aggregated bandwidth must be >= 450 Mbps.\n"
            "2. Average latency across the 3 paths must be <= 20 ms.\n"
            "3. Maximum packet loss among chosen paths must not exceed 0.6%.\n"
            "4. Total hourly cost must be <= $45/hr.\n"
            "5. Must include P3 for critical telemetry.\n\n"
            "Output ONLY the 3 path IDs separated by commas in alphabetical order (e.g. \"P2, P3, P5\")."
        ),
        ground_truth="P2, P3, P5",
        expected_constraints=[
            "Total bandwidth >= 450 Mbps",
            "Average latency <= 20 ms",
            "Packet loss <= 0.6%",
            "Cost <= $45/hr",
            "Must include P3",
        ],
        verifier_type=VerifierType.MATHEMATICAL_CONSTRAINT,
        metadata={"expected_paths": ["P2", "P3", "P5"]},
    ),

    # ------------------------------------------------------------------------
    # Domain 2: Winograd Schema & Pronoun Disambiguation (5 Cases)
    # ------------------------------------------------------------------------
    CognitiveTestCase(
        id="wsd_01",
        domain=DomainType.WINOGRAD_SCHEMA,
        title="Physical Affordance & Containment Binding",
        prompt=(
            "Context: \"The heavy bronze trophy could not fit into the leather travel suitcase because it was too large.\"\n"
            "Question: What was too large?\n"
            "Answer with ONLY the exact referent noun phrase (either \"the trophy\" or \"the suitcase\")."
        ),
        ground_truth="the trophy",
        expected_constraints=["Correct antecedent binding based on physical size affordance"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={"valid_patterns": ["the trophy", "trophy"], "invalid_patterns": ["the suitcase", "suitcase"]},
    ),
    CognitiveTestCase(
        id="wsd_02",
        domain=DomainType.WINOGRAD_SCHEMA,
        title="Semantic Polarity Reversal Disambiguation",
        prompt=(
            "Context: \"The heavy bronze trophy could not fit into the leather travel suitcase because it was too small.\"\n"
            "Question: What was too small?\n"
            "Answer with ONLY the exact referent noun phrase (either \"the trophy\" or \"the suitcase\")."
        ),
        ground_truth="the suitcase",
        expected_constraints=["Correct polarity inversion binding based on container capacity"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={"valid_patterns": ["the suitcase", "suitcase"], "invalid_patterns": ["the trophy", "trophy"]},
    ),
    CognitiveTestCase(
        id="wsd_03",
        domain=DomainType.WINOGRAD_SCHEMA,
        title="Corporate Contract Breach Fiduciary Binding",
        prompt=(
            "Context: \"Apex Logistics sued Summit Cargo rather than Vertex Express because they breached the exclusive regional distribution contract.\"\n"
            "Question: In this sentence, who breached the exclusive regional distribution contract?\n"
            "Answer with ONLY the exact company name (\"Apex Logistics\", \"Summit Cargo\", or \"Vertex Express\")."
        ),
        ground_truth="Summit Cargo",
        expected_constraints=["Disambiguate defendant agent from plaintiff and third-party entity"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={"valid_patterns": ["Summit Cargo"], "invalid_patterns": ["Apex Logistics", "Vertex Express"]},
    ),
    CognitiveTestCase(
        id="wsd_04",
        domain=DomainType.WINOGRAD_SCHEMA,
        title="Pharmacotherapy Mechanism of Action Disambiguation",
        prompt=(
            "Context: \"Dr. Chen prescribed Lisinopril to Marcus instead of Metoprolol because it effectively lowers angiotensin-converting enzyme activity.\"\n"
            "Question: What effectively lowers angiotensin-converting enzyme activity?\n"
            "Answer with ONLY the exact medication name (\"Lisinopril\" or \"Metoprolol\")."
        ),
        ground_truth="Lisinopril",
        expected_constraints=["Select correct medication matching the biochemical mechanism of action"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={"valid_patterns": ["Lisinopril"], "invalid_patterns": ["Metoprolol"]},
    ),
    CognitiveTestCase(
        id="wsd_05",
        domain=DomainType.WINOGRAD_SCHEMA,
        title="Legal Indemnity Clause Reciprocal Disambiguation",
        prompt=(
            "Context: \"The Landlord shall defend and hold harmless the Tenant against any third-party property damage claims arising from common areas, provided that they did not cause the structural defect through gross negligence.\"\n"
            "Question: Who must not have caused the structural defect through gross negligence to qualify for protection?\n"
            "Answer with ONLY the exact party (\"The Landlord\" or \"The Tenant\")."
        ),
        ground_truth="The Tenant",
        expected_constraints=["Resolve conditional qualification pronoun in indemnity clause"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={"valid_patterns": ["The Tenant", "Tenant"], "invalid_patterns": ["The Landlord", "Landlord"]},
    ),

    # ------------------------------------------------------------------------
    # Domain 3: Semantic Denoising & Noisy Intent Extraction (5 Cases)
    # ------------------------------------------------------------------------
    CognitiveTestCase(
        id="sdn_01",
        domain=DomainType.SEMANTIC_DENOISING,
        title="Angry Customer Return & Sarcasm Denoising",
        prompt=(
            "Extract the structured support action from the user message below.\n"
            "User Message:\n"
            "\"Oh wow, absolutely fantastic job guys! My brand new QuantumX Pro Headphones (Order #QX-99281) arrived today, and shocker—the left earcup is completely dead! What a technological marvel! I don't want your replacement junk or a store credit coupon for 5% off; just give me a full refund to my original Visa immediately before I lose my mind!\"\n\n"
            "Output ONLY a JSON object with keys:\n"
            "- \"action\": (\"REFUND\", \"REPLACEMENT\", \"REPAIR\", or \"INQUIRY\")\n"
            "- \"order_id\": string\n"
            "- \"product\": string\n"
            "- \"payment_target\": (\"ORIGINAL_PAYMENT\", \"STORE_CREDIT\", or \"NONE\")"
        ),
        ground_truth='{"action": "REFUND", "order_id": "QX-99281", "product": "QuantumX Pro Headphones", "payment_target": "ORIGINAL_PAYMENT"}',
        expected_constraints=[
            "Identify REFUND intent despite sarcastic praise",
            "Extract order ID QX-99281",
            "Extract product name QuantumX Pro Headphones",
            "Extract payment target ORIGINAL_PAYMENT",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "action": "REFUND",
                "order_id": "QX-99281",
                "product": "QuantumX Pro Headphones",
                "payment_target": "ORIGINAL_PAYMENT",
            },
            "required_keys": ["action", "order_id", "product", "payment_target"],
        },
    ),
    CognitiveTestCase(
        id="sdn_02",
        domain=DomainType.SEMANTIC_DENOISING,
        title="DevOps Multi-Speaker Incident Log Isolation",
        prompt=(
            "Analyze this chaotic incident channel log and extract the agreed incident action:\n"
            "Dave (14:02): \"Database is burning down! CPU 100%! Is it the new auth-service v2.4.1 release??\"\n"
            "Sarah (14:03): \"No wait, I checked auth-service, queries are fine. Look at payments-worker v3.1.0 that Alex pushed 10 mins ago!\"\n"
            "Alex (14:04): \"My bad, payments-worker is stuck in an unindexed retry loop hammering PostgreSQL.\"\n"
            "Dave (14:05): \"Should we scale up the RDS instance to db.m5.4xlarge?\"\n"
            "Sarah (14:05): \"No, scaling won't fix the loop. Roll back payments-worker to v3.0.9 immediately.\"\n"
            "Alex (14:06): \"Agreed, rolling back now.\"\n\n"
            "Output ONLY a JSON object with:\n"
            "- \"target_service\": string\n"
            "- \"operation\": (\"ROLLBACK\", \"SCALE_UP\", \"RESTART\", or \"PATCH\")\n"
            "- \"target_version\": string"
        ),
        ground_truth='{"target_service": "payments-worker", "operation": "ROLLBACK", "target_version": "v3.0.9"}',
        expected_constraints=[
            "Isolate true consensus target service payments-worker",
            "Extract agreed operation ROLLBACK (rejecting proposed SCALE_UP)",
            "Extract target version v3.0.9",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "target_service": "payments-worker",
                "operation": "ROLLBACK",
                "target_version": "v3.0.9",
            },
            "required_keys": ["target_service", "operation", "target_version"],
        },
    ),
    CognitiveTestCase(
        id="sdn_03",
        domain=DomainType.SEMANTIC_DENOISING,
        title="Meeting Transcript Action Item & Banter Filtering",
        prompt=(
            "Extract the single verified assigned action item from this meeting transcript snippet:\n"
            "\"Tom: So anyway, did anyone see the game last night? Hilarious fourth quarter.\n"
            "Elena: Yeah, but we need to focus. We had 400 customer complaints about the PDF export bug.\n"
            "Tom: Maybe we should just redesign the entire dashboard in Vue 3?\n"
            "Elena: No way Tom, that's a 6-month project. Let's stay on topic.\n"
            "Rachel: I found the bug—it's an unescaped Unicode character in the PDF header. I can patch the export service by Thursday 5 PM.\n"
            "Elena: Perfect Rachel, let's lock that in. Tom, don't touch the frontend.\"\n\n"
            "Output ONLY a JSON object with:\n"
            "- \"assignee\": string\n"
            "- \"task_description\": string\n"
            "- \"deadline\": string"
        ),
        ground_truth='{"assignee": "Rachel", "task_description": "Patch the PDF export service Unicode bug", "deadline": "Thursday 5 PM"}',
        expected_constraints=[
            "Extract true assignee Rachel",
            "Extract deadline Thursday 5 PM",
            "Filter out conversational tangents",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "assignee": "Rachel",
                "deadline": "Thursday 5 PM",
            },
            "required_keys": ["assignee", "task_description", "deadline"],
        },
    ),
    CognitiveTestCase(
        id="sdn_04",
        domain=DomainType.SEMANTIC_DENOISING,
        title="Sarcastic Hypothetical SQL Database Update Extraction",
        prompt=(
            "Extract the true database query requirement from this user prompt:\n"
            "\"Oh sure, why don't we just DELETE all users from the database because that would solve all our problems, wouldn't it?! Or better yet, DROP TABLE transactions! But in the real world where we actually need to do our jobs, please just update the status of transaction TXN-884102 to 'REFUNDED' and set the updated_by field to 'admin_sarah'.\"\n\n"
            "Output ONLY a JSON object with:\n"
            "- \"statement_type\": (\"UPDATE\", \"DELETE\", \"DROP\", or \"SELECT\")\n"
            "- \"target_table\": string\n"
            "- \"filter_id\": string\n"
            "- \"set_status\": string"
        ),
        ground_truth='{"statement_type": "UPDATE", "target_table": "transactions", "filter_id": "TXN-884102", "set_status": "REFUNDED"}',
        expected_constraints=[
            "Extract UPDATE statement (denoising DELETE/DROP sarcasm)",
            "Extract target table transactions",
            "Extract filter ID TXN-884102",
            "Extract status REFUNDED",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "statement_type": "UPDATE",
                "target_table": "transactions",
                "filter_id": "TXN-884102",
                "set_status": "REFUNDED",
            },
            "required_keys": ["statement_type", "target_table", "filter_id", "set_status"],
        },
    ),
    CognitiveTestCase(
        id="sdn_05",
        domain=DomainType.SEMANTIC_DENOISING,
        title="Rambling Stream-of-Consciousness Flight Parameter Extraction",
        prompt=(
            "Extract flight search parameters from this rambling request:\n"
            "\"I was thinking maybe Paris, or Rome, but honestly my sister lives in London so that's out because I don't want to see her, and Tokyo is too far for a weekend trip. Actually I have a conference in San Francisco (SFO) starting on October 14th, 2026. I'll be flying out from Boston (BOS) on October 12th, 2026, traveling economy class with 1 checked bag. Don't book me on Spirit or Frontier, I'd rather walk.\"\n\n"
            "Output ONLY a JSON object with:\n"
            "- \"origin_airport\": string (3-letter IATA code)\n"
            "- \"destination_airport\": string (3-letter IATA code)\n"
            "- \"departure_date\": string (YYYY-MM-DD)\n"
            "- \"cabin_class\": (\"ECONOMY\", \"PREMIUM_ECONOMY\", \"BUSINESS\", or \"FIRST\")"
        ),
        ground_truth='{"origin_airport": "BOS", "destination_airport": "SFO", "departure_date": "2026-10-12", "cabin_class": "ECONOMY"}',
        expected_constraints=[
            "Extract origin BOS (ignoring Paris, Rome, London, Tokyo)",
            "Extract destination SFO",
            "Extract departure date 2026-10-12",
            "Extract cabin class ECONOMY",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "origin_airport": "BOS",
                "destination_airport": "SFO",
                "departure_date": "2026-10-12",
                "cabin_class": "ECONOMY",
            },
            "required_keys": ["origin_airport", "destination_airport", "departure_date", "cabin_class"],
        },
    ),

    # ------------------------------------------------------------------------
    # Domain 4: Cross-Context Multi-Clue Synthesis (5 Cases)
    # ------------------------------------------------------------------------
    CognitiveTestCase(
        id="cms_01",
        domain=DomainType.MULTI_CLUE_SYNTHESIS,
        title="Whodunit Disjoint Alibi Elimination Deduction",
        prompt=(
            "Read the scattered clues and determine who committed the theft at the Art Gallery at 9:00 PM:\n"
            "Clue 1: Professor Plum was dining at the French Bistro from 8:30 PM to 10:00 PM with Miss Scarlet.\n"
            "Clue 2: Colonel Mustard was seen at the Train Station at 9:00 PM boarding the express train to London.\n"
            "Clue 3: Mrs. White was attending the Opera with Mayor Green until 10:30 PM.\n"
            "Clue 4: The thief was at the Art Gallery at 9:00 PM and left behind a vintage fountain pen.\n"
            "Clue 5: Mr. Green, Colonel Mustard, Professor Plum, Miss Scarlet, and Mrs. Peacock are the only suspects.\n"
            "Clue 6: Mrs. Peacock owns a vintage fountain pen and had no confirmed location between 8:00 PM and 11:00 PM.\n\n"
            "Question: Who is the thief?\n"
            "Output ONLY the exact name of the culprit (e.g. \"Mrs. Peacock\")."
        ),
        ground_truth="Mrs. Peacock",
        expected_constraints=["Synthesize disjoint alibis and physical evidence to deduce Mrs. Peacock"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={
            "valid_patterns": ["Mrs. Peacock", "Mrs Peacock", "Peacock"],
            "invalid_patterns": ["Professor Plum", "Colonel Mustard", "Miss Scarlet", "Mr. Green", "Mrs. White"],
        },
    ),
    CognitiveTestCase(
        id="cms_02",
        domain=DomainType.MULTI_CLUE_SYNTHESIS,
        title="Multi-Tier Supply Chain Bottleneck Root Cause",
        prompt=(
            "Determine which manufacturing plant is causing the global assembly delay based on these reports:\n"
            "- Report A: The final assembly plant in Munich requires 500 microchips/day from Fab Alpha and 200 battery packs/day from Plant Gamma.\n"
            "- Report B: Plant Gamma is producing 220 battery packs/day and has a 3-week surplus in inventory.\n"
            "- Report C: Fab Alpha requires ultra-pure silicon wafers from Supplier Beta.\n"
            "- Report D: Supplier Beta suffered a power grid failure, reducing wafer output to 40% of normal capacity, preventing Fab Alpha from meeting its 500 microchips/day commitment.\n"
            "- Report E: Logistics transport routes between all facilities are running on schedule without customs delays.\n\n"
            "Question: What is the single primary root-cause supplier/plant responsible for the bottleneck?\n"
            "Output ONLY the exact entity name (e.g. \"Supplier Beta\")."
        ),
        ground_truth="Supplier Beta",
        expected_constraints=["Trace upstream dependency chain to Supplier Beta power failure"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={
            "valid_patterns": ["Supplier Beta"],
            "invalid_patterns": ["Fab Alpha", "Plant Gamma", "Munich"],
        },
    ),
    CognitiveTestCase(
        id="cms_03",
        domain=DomainType.MULTI_CLUE_SYNTHESIS,
        title="Multi-Generation Lineage Kinship Degree Resolution",
        prompt=(
            "Determine the exact familial relationship between Arthur and Brenda from these historical records:\n"
            "- Record 1: Charles is the father of Arthur and David.\n"
            "- Record 2: Edward is the father of Charles and Fiona.\n"
            "- Record 3: Fiona is the mother of George and Brenda.\n"
            "- Record 4: No other marital or adoption relationships exist.\n\n"
            "Question: What is the exact familial relationship of Brenda to Arthur?\n"
            "(Choose one: \"Sister\", \"First Cousin\", \"Aunt\", \"Niece\", \"Mother\")\n"
            "Output ONLY the exact relationship term."
        ),
        ground_truth="First Cousin",
        expected_constraints=["Traverse shared grandfather Edward to determine first cousin relationship"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={
            "valid_patterns": ["First Cousin", "Cousin"],
            "invalid_patterns": ["Sister", "Aunt", "Niece", "Mother"],
        },
    ),
    CognitiveTestCase(
        id="cms_04",
        domain=DomainType.MULTI_CLUE_SYNTHESIS,
        title="Distributed Microservice Trace Crash Diagnosis",
        prompt=(
            "Analyze these asynchronous system logs and identify the root cause service:\n"
            "- Log 1 [09:15:01] FrontendGateway: HTTP 504 Gateway Timeout returned to client on /checkout endpoint.\n"
            "- Log 2 [09:14:59] OrderService: Call to PaymentService timed out after 3000ms.\n"
            "- Log 3 [09:14:58] PaymentService: Acquiring distributed lock from RedisLock cluster failed due to connection timeout.\n"
            "- Log 4 [09:14:55] InventoryService: Stock reservation completed successfully in 12ms.\n"
            "- Log 5 [09:14:56] RedisLock: Cluster master node node-03 crashed due to OutOfMemory exception on key expiration queue.\n\n"
            "Question: Which service/component experienced the primary root-cause failure?\n"
            "Output ONLY the component name (e.g. \"RedisLock\")."
        ),
        ground_truth="RedisLock",
        expected_constraints=["Synthesize 5-hop distributed trace to RedisLock node-03 crash"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={
            "valid_patterns": ["RedisLock", "RedisLock node-03", "Redis"],
            "invalid_patterns": ["FrontendGateway", "OrderService", "PaymentService", "InventoryService"],
        },
    ),
    CognitiveTestCase(
        id="cms_05",
        domain=DomainType.MULTI_CLUE_SYNTHESIS,
        title="Cascading Biochemical Pathway Enzyme Inhibition",
        prompt=(
            "Synthesize the outcome on Compound E based on these biochemical findings:\n"
            "- Finding 1: Compound A is converted to Compound B by Enzyme 1.\n"
            "- Finding 2: Compound B is converted to Compound C by Enzyme 2.\n"
            "- Finding 3: Compound C activates Enzyme 3, which synthesizes Compound E from Precursor D.\n"
            "- Finding 4: Molecule X is a potent competitive inhibitor of Enzyme 2.\n"
            "- Finding 5: A cell culture is treated with high concentrations of Molecule X.\n\n"
            "Question: What happens to the concentration of Compound E in the cell culture?\n"
            "(Choose one: \"Increases\", \"Decreases\", \"Remains Unchanged\")\n"
            "Output ONLY the single word answer."
        ),
        ground_truth="Decreases",
        expected_constraints=["Deduce downstream decrease in Compound E from Enzyme 2 inhibition"],
        verifier_type=VerifierType.EXACT_MATCH,
        metadata={
            "valid_patterns": ["Decreases", "Decrease"],
            "invalid_patterns": ["Increases", "Increase", "Remains Unchanged"],
        },
    ),

    # ------------------------------------------------------------------------
    # Domain 5: Action & Tool Routing (5 Cases)
    # ------------------------------------------------------------------------
    CognitiveTestCase(
        id="atr_01",
        domain=DomainType.ACTION_TOOL_ROUTING,
        title="Financial Portfolio Rebalancer Tool Routing",
        prompt=(
            "You have access to the following 5 financial tools:\n"
            "- T1: `fetch_realtime_quote(symbol: str)` — Retrieves latest bid/ask and volume for a ticker.\n"
            "- T2: `calculate_portfolio_var(portfolio_id: str, confidence_level: float)` — Computes Value-at-Risk using historical simulation.\n"
            "- T3: `execute_twap_order(symbol: str, quantity: int, duration_minutes: int)` — Submits a Time-Weighted Average Price algorithmic trade.\n"
            "- T4: `rebalance_portfolio_weights(portfolio_id: str, target_allocations: dict, max_slippage_bps: int)` — Rebalances asset allocation to target percentages while bounding transaction costs.\n"
            "- T5: `get_tax_loss_harvesting_candidates(portfolio_id: str, min_loss_usd: float)` — Scans portfolio for unrealized capital losses suitable for tax offset.\n\n"
            "User Request:\n"
            "\"Our risk committee approved the new asset mix for Fund-7: shift to 60% equities and 40% fixed income. Execute the trades across the portfolio to match these target allocations, keeping slippage under 15 basis points.\"\n\n"
            "Output ONLY a JSON object with:\n"
            "- \"tool_id\": string (e.g. \"T4\")\n"
            "- \"tool_name\": string\n"
            "- \"extracted_parameters\": object"
        ),
        ground_truth='{"tool_id": "T4", "tool_name": "rebalance_portfolio_weights", "extracted_parameters": {"portfolio_id": "Fund-7", "max_slippage_bps": 15}}',
        expected_constraints=[
            "Route to tool T4 rebalance_portfolio_weights",
            "Extract portfolio_id Fund-7",
            "Extract max_slippage_bps 15",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "tool_id": "T4",
                "tool_name": "rebalance_portfolio_weights",
                "extracted_parameters": {
                    "portfolio_id": "Fund-7",
                    "max_slippage_bps": 15,
                },
            },
            "required_keys": ["tool_id", "tool_name", "extracted_parameters"],
        },
    ),
    CognitiveTestCase(
        id="atr_02",
        domain=DomainType.ACTION_TOOL_ROUTING,
        title="Cloud WAF IP Blocklist Infrastructure Routing",
        prompt=(
            "Select the correct cloud automation tool from:\n"
            "- T1: `scale_k8s_nodegroup(cluster_name: str, nodegroup: str, desired_count: int)` — Adjusts Kubernetes worker node pool capacity.\n"
            "- T2: `restart_pod(namespace: str, pod_name: str, force: bool)` — Triggers rolling pod restart in a namespace.\n"
            "- T3: `provision_rds_aurora(cluster_id: str, engine: str, instance_class: str, replicas: int)` — Deploys a new managed Amazon Aurora database cluster.\n"
            "- T4: `update_waf_ip_blocklist(waf_acl_id: str, ip_cidr_list: list, action: str)` — Updates IP blocking rules in Web Application Firewall.\n"
            "- T5: `rotate_iam_access_keys(username: str, expire_old_after_hours: int)` — Rotates programmatic API keys for an IAM user.\n\n"
            "User Request:\n"
            "\"We are seeing a distributed brute-force attack on our payment gateway coming from subnet 198.51.100.0/24 and 203.0.113.0/24. Block both CIDRs in WAF ACL 'acl-prod-us-east-1' immediately.\"\n\n"
            "Output ONLY a JSON object with:\n"
            "- \"tool_id\": (\"T1\", \"T2\", \"T3\", \"T4\", or \"T5\")\n"
            "- \"tool_name\": string\n"
            "- \"target_acl\": string"
        ),
        ground_truth='{"tool_id": "T4", "tool_name": "update_waf_ip_blocklist", "target_acl": "acl-prod-us-east-1"}',
        expected_constraints=[
            "Route to T4 update_waf_ip_blocklist",
            "Extract target ACL acl-prod-us-east-1",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "tool_id": "T4",
                "tool_name": "update_waf_ip_blocklist",
                "target_acl": "acl-prod-us-east-1",
            },
            "required_keys": ["tool_id", "tool_name", "target_acl"],
        },
    ),
    CognitiveTestCase(
        id="atr_03",
        domain=DomainType.ACTION_TOOL_ROUTING,
        title="Biomedical ClinVar Genomic Variant Lookup Routing",
        prompt=(
            "Select the optimal bioinformatics API tool from:\n"
            "- T1: `query_clinvar_variant(hgvs_notation: str, genome_assembly: str)` — Retrieves clinical significance and pathogenicity for human genomic variants.\n"
            "- T2: `fetch_uniprot_structure(uniprot_id: str, format: str)` — Downloads 3D coordinates and AlphaFold pLDDT scores for a protein.\n"
            "- T3: `run_blast_alignment(sequence: str, database: str, evalue_cutoff: float)` — Performs local sequence alignment search against NCBI nucleotide/protein DB.\n"
            "- T4: `query_gtex_expression(gene_symbol: str, tissue_site: str)` — Fetches quantitative RNA tissue expression and eQTL data.\n"
            "- T5: `fetch_chembl_bioactivity(target_chembl_id: str, activity_type: str)` — Queries IC50/Ki bioactivity values for chemical compounds against a target.\n\n"
            "User Request:\n"
            "\"We identified a missense mutation NM_000059.3:c.5946del in BRCA2. Look up its clinical pathogenicity classification and review status on GRCh38.\"\n\n"
            "Output ONLY a JSON object with:\n"
            "- \"tool_id\": (\"T1\", \"T2\", \"T3\", \"T4\", or \"T5\")\n"
            "- \"tool_name\": string\n"
            "- \"variant_identifier\": string"
        ),
        ground_truth='{"tool_id": "T1", "tool_name": "query_clinvar_variant", "variant_identifier": "NM_000059.3:c.5946del"}',
        expected_constraints=[
            "Route to T1 query_clinvar_variant",
            "Extract variant NM_000059.3:c.5946del",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "tool_id": "T1",
                "tool_name": "query_clinvar_variant",
                "variant_identifier": "NM_000059.3:c.5946del",
            },
            "required_keys": ["tool_id", "tool_name", "variant_identifier"],
        },
    ),
    CognitiveTestCase(
        id="atr_04",
        domain=DomainType.ACTION_TOOL_ROUTING,
        title="Smart Home Multimodal HVAC Controller Dispatch",
        prompt=(
            "Select the correct smart home device controller from:\n"
            "- T1: `adjust_hvac_zones(zone_names: list, target_temp_f: float, mode: str)` — Sets thermostat temperature and HVAC mode for home zones.\n"
            "- T2: `set_security_alarm_armed(mode: str, bypass_sensors: list)` — Arms home alarm in 'AWAY', 'STAY', or 'NIGHT' mode.\n"
            "- T3: `dim_lighting_scene(room_name: str, scene_name: str, brightness_pct: int)` — Activates pre-configured lighting scene in a room.\n"
            "- T4: `schedule_irrigation(zone_id: int, duration_minutes: int, skip_if_rain: bool)` — Schedules lawn sprinkler cycle.\n"
            "- T5: `query_energy_consumption(time_range: str, device_filter: str)` — Returns kilowatt-hour telemetry for smart meter circuits.\n\n"
            "User Request:\n"
            "\"It's getting chilly in the nursery and the master bedroom; please turn the heat on and warm both rooms up to 72 degrees.\"\n\n"
            "Output ONLY a JSON object with:\n"
            "- \"tool_id\": string (\"T1\"..\"T5\")\n"
            "- \"tool_name\": string\n"
            "- \"target_temp\": float\n"
            "- \"mode\": string"
        ),
        ground_truth='{"tool_id": "T1", "tool_name": "adjust_hvac_zones", "target_temp": 72.0, "mode": "heat"}',
        expected_constraints=[
            "Route to T1 adjust_hvac_zones",
            "Extract target temperature 72.0",
            "Extract mode heat",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "tool_id": "T1",
                "tool_name": "adjust_hvac_zones",
                "target_temp": 72.0,
                "mode": "heat",
            },
            "required_keys": ["tool_id", "tool_name", "target_temp", "mode"],
        },
    ),
    CognitiveTestCase(
        id="atr_05",
        domain=DomainType.ACTION_TOOL_ROUTING,
        title="E-Commerce Warehouse Robotics Picker Dispatch",
        prompt=(
            "Route this fulfillment request to the appropriate logistics API:\n"
            "- T1: `calculate_shipping_rates(origin_zip: str, dest_zip: str, weight_lbs: float, carrier: str)` — Fetches real-time freight and parcel rate quotes.\n"
            "- T2: `generate_return_label(order_id: str, return_reason: str, carrier: str)` — Creates printable PDF return shipping label.\n"
            "- T3: `dispatch_warehouse_picker(warehouse_id: str, sku_list: list, priority: str)` — Queues SKU pick list for automated warehouse robots.\n"
            "- T4: `cancel_unfulfilled_order(order_id: str, refund_customer: bool)` — Cancels backordered items before shipping label generation.\n"
            "- T5: `schedule_freight_pickup(dock_id: str, pickup_datetime: str, pallet_count: int)` — Books LTL freight carrier dock appointment.\n\n"
            "User Request:\n"
            "\"Order #ORD-77192 has 4 units of SKU-A99 and 2 units of SKU-B12 sitting in Warehouse-West. The customer paid for next-day air, so send the pick robot immediately with HIGH priority.\"\n\n"
            "Output ONLY a JSON object with:\n"
            "- \"tool_id\": string\n"
            "- \"tool_name\": string\n"
            "- \"warehouse_id\": string\n"
            "- \"priority\": string"
        ),
        ground_truth='{"tool_id": "T3", "tool_name": "dispatch_warehouse_picker", "warehouse_id": "Warehouse-West", "priority": "HIGH"}',
        expected_constraints=[
            "Route to T3 dispatch_warehouse_picker",
            "Extract warehouse_id Warehouse-West",
            "Extract priority HIGH",
        ],
        verifier_type=VerifierType.JSON_SCHEMA,
        metadata={
            "expected_fields": {
                "tool_id": "T3",
                "tool_name": "dispatch_warehouse_picker",
                "warehouse_id": "Warehouse-West",
                "priority": "HIGH",
            },
            "required_keys": ["tool_id", "tool_name", "warehouse_id", "priority"],
        },
    ),
]


# ============================================================================
# Public API and Benchmark Suite Loader Functions
# ============================================================================

def load_cognitive_benchmark_suite(
    domain: Optional[Union[str, DomainType]] = None,
) -> List[CognitiveTestCase]:
    """Load the complete benchmark suite or filter by a specific cognitive domain.

    Args:
        domain: Optional domain filter (str or DomainType enum).

    Returns:
        List of CognitiveTestCase objects.
    """
    if domain is None:
        return list(_BENCHMARK_CASES)

    if isinstance(domain, str):
        try:
            target_domain = DomainType.from_str(domain)
        except ValueError:
            target_domain = domain.strip().lower()
    else:
        target_domain = domain

    filtered = []
    for case in _BENCHMARK_CASES:
        case_dom = case.domain if isinstance(case.domain, DomainType) else DomainType.from_str(str(case.domain))
        if case_dom == target_domain or str(case.domain).lower() == str(domain).lower():
            filtered.append(case)

    return filtered


def get_test_case_by_id(case_id: str) -> Optional[CognitiveTestCase]:
    """Retrieve a single test case by its ID (e.g. 'mcs_01', 'sdn_03')."""
    cid = case_id.strip().lower()
    for case in _BENCHMARK_CASES:
        if case.id.lower() == cid:
            return case
    return None


def get_domain_summary() -> Dict[str, Any]:
    """Provide a structured summary of available domains and test cases."""
    summary: Dict[str, Any] = {}
    for d in DomainType:
        cases = load_cognitive_benchmark_suite(d)
        summary[d.value] = {
            "name": d.name,
            "count": len(cases),
            "test_case_ids": [c.id for c in cases],
            "titles": [c.title for c in cases],
        }
    summary["_total_test_cases"] = len(_BENCHMARK_CASES)
    return summary
