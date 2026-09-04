"""Procedural Reasoning Domain Lane for PRLR: Multi-Step Tool Routing DAG (MTR-DAG).

Implements:
- 4 Domain Families: api_workflow, data_pipeline, security_ops, robotics_control (held-out)
- Deterministic BFS Oracle Solver with canonical lexicographical tie-breaking and execution tracing
- ProceduralLaneGenerator with guaranteed multi-step deliberation (K >= 2)
- ProceduralVerifier with independent ground-truth evaluation
- Gemma 2B instruction prompt and target formatting (<start_of_turn>user / <start_of_turn>model)
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import hashlib
import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ToolDefinition:
    """Immutable definition of an executable tool transition."""
    name: str
    required_inputs: Tuple[str, ...]
    produced_outputs: Tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ExecutionTraceStep:
    """Record of a single execution step along a reasoning trajectory."""
    step: int
    tool: str
    consumed: Tuple[str, ...]
    produced: Tuple[str, ...]
    cumulative_state: Tuple[str, ...]


@dataclass(frozen=True)
class ProceduralProblemInstance:
    """Fully specified procedural problem instance conforming to prlr.domain.v1."""
    sample_id: str
    domain: str
    prompt: str
    ground_truth_json: str
    expected_route: Tuple[str, ...]
    terminal_tool: str
    depth_K: int
    initial_state: Tuple[str, ...]
    target_goal: str
    trace: Tuple[ExecutionTraceStep, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Domain Catalogues ---

DOMAIN_CATALOGUES: Dict[str, Dict[str, List[ToolDefinition]]] = {
    "api_workflow": {
        "core": [
            ToolDefinition("schema_parser", ("raw_payload",), ("parsed_request",), "Parses raw inbound HTTP payload"),
            ToolDefinition("auth_validator", ("parsed_request",), ("user_session",), "Validates auth credentials and creates session"),
            ToolDefinition("permission_checker", ("user_session",), ("perm_token",), "Checks role permissions and creates token"),
            ToolDefinition("db_reader", ("perm_token",), ("user_profile",), "Reads user database records"),
            ToolDefinition("risk_evaluator", ("user_profile",), ("risk_score",), "Computes transaction risk score"),
            ToolDefinition("policy_gate", ("risk_score",), ("approval_decision",), "Evaluates access policy gate"),
            ToolDefinition("audit_logger", ("approval_decision",), ("audit_receipt",), "Records audit trail and creates receipt"),
        ],
        "distractors": [
            ToolDefinition("cache_reader", ("cache_key",), ("cache_hit",), "Reads cached response"),
            ToolDefinition("billing_meter", ("billing_id",), ("invoice_record",), "Meters API usage"),
            ToolDefinition("webhook_emitter", ("audit_receipt",), ("callback_event",), "Emits outgoing webhook"),
            ToolDefinition("rate_limiter", ("client_ip",), ("throttle_state",), "Evaluates rate limit quota"),
        ],
    },
    "data_pipeline": {
        "core": [
            ToolDefinition("stream_ingester", ("raw_records",), ("ingested_batch",), "Ingests raw event stream"),
            ToolDefinition("null_cleaner", ("ingested_batch",), ("clean_records",), "Filters null and malformed records"),
            ToolDefinition("schema_normalizer", ("clean_records",), ("normalized_frame",), "Normalizes tabular schema"),
            ToolDefinition("feature_extractor", ("normalized_frame",), ("feature_matrix",), "Extracts analytical features"),
            ToolDefinition("model_evaluator", ("feature_matrix",), ("prediction_metrics",), "Evaluates ML model inference"),
            ToolDefinition("alert_dispatcher", ("prediction_metrics",), ("incident_ticket",), "Dispatches alert on anomaly"),
            ToolDefinition("cold_archiver", ("incident_ticket",), ("archive_receipt",), "Archives clean records to cold store"),
        ],
        "distractors": [
            ToolDefinition("drift_monitor", ("baseline_spec",), ("drift_report",), "Monitors data drift"),
            ToolDefinition("sample_exporter", ("export_id",), ("csv_dump",), "Exports raw samples to CSV"),
            ToolDefinition("latency_tracker", ("telemetry_port",), ("perf_log",), "Tracks pipeline latency"),
            ToolDefinition("lineage_tracer", ("lineage_id",), ("lineage_graph",), "Traces data provenance"),
        ],
    },
    "security_ops": {
        "core": [
            ToolDefinition("telemetry_parser", ("pcap_stream",), ("network_flows",), "Parses raw packet capture"),
            ToolDefinition("ioc_correlator", ("network_flows",), ("threat_indicators",), "Correlates threat indicators"),
            ToolDefinition("endpoint_inspector", ("threat_indicators",), ("host_forensics",), "Inspects endpoint artifacts"),
            ToolDefinition("quarantine_enforcer", ("host_forensics",), ("isolated_host",), "Isolates compromised host"),
            ToolDefinition("firewall_reconfig", ("isolated_host",), ("blocked_subnet",), "Blocks malicious subnet"),
            ToolDefinition("incident_publisher", ("blocked_subnet",), ("security_advisory",), "Publishes incident report"),
        ],
        "distractors": [
            ToolDefinition("patch_scheduler", ("cve_catalog",), ("remediation_plan",), "Schedules system patch"),
            ToolDefinition("dns_sinkhole", ("sinkhole_ip",), ("filtered_dns",), "Redirects malicious DNS"),
            ToolDefinition("hash_lookup", ("artifact_hash",), ("known_bad_hashes",), "Looks up file hash database"),
            ToolDefinition("sandbox_detonator", ("vm_image",), ("sandbox_log",), "Executes artifact in sandbox"),
        ],
    },
    "robotics_control": {
        "core": [
            ToolDefinition("lidar_processor", ("point_cloud_raw",), ("filtered_points",), "Filters raw lidar scan"),
            ToolDefinition("point_cloud_filter", ("filtered_points",), ("voxel_grid",), "Downsamples into 3D voxel grid"),
            ToolDefinition("obstacle_detector", ("voxel_grid",), ("obstacle_map",), "Detects geometric obstacles"),
            ToolDefinition("kinematics_solver", ("obstacle_map",), ("joint_angles",), "Solves inverse kinematics"),
            ToolDefinition("collision_checker", ("joint_angles",), ("safe_configuration",), "Verifies trajectory collision"),
            ToolDefinition("trajectory_planner", ("safe_configuration",), ("spline_path",), "Plans smooth polynomial spline"),
            ToolDefinition("actuator_controller", ("spline_path",), ("motor_commands",), "Outputs actuator torque commands"),
        ],
        "distractors": [
            ToolDefinition("battery_monitor", ("sensor_bus",), ("power_consumption",), "Monitors power draw"),
            ToolDefinition("gps_sync", ("satellite_fix",), ("geo_fix",), "Synchronizes GPS coordinates"),
            ToolDefinition("imu_filter", ("raw_gyro",), ("gravity_vector",), "Filters IMU acceleration"),
            ToolDefinition("thermal_guard", ("sensor_temp",), ("motor_temp_est",), "Estimates actuator heat"),
        ],
    },
}


class DeterministicToolRoutingOracle:
    """Deterministic BFS oracle solver with canonical lexicographical tie-breaking."""

    def solve(
        self,
        tools: Sequence[ToolDefinition],
        initial_state: Set[str],
        goal: str,
        max_depth: int = 12,
    ) -> Optional[Dict[str, Any]]:
        """Find the minimal tool route from initial_state to goal.

        Args:
            tools: Available tool catalog.
            initial_state: Set of available state variables.
            goal: Target state variable.
            max_depth: Maximum search depth.

        Returns:
            Dict containing route, terminal_tool, depth_K, trace, or None if unreachable.
        """
        if goal in initial_state:
            return None  # Trivial / 0-step problem prohibited

        tools_map = {t.name: t for t in tools}
        queue = collections.deque([([], set(initial_state))])
        found_solutions: List[List[str]] = []
        best_depth: Optional[int] = None

        while queue:
            route, current_state = queue.popleft()
            depth = len(route)

            if best_depth is not None and depth > best_depth:
                break

            if goal in current_state:
                if best_depth is None:
                    best_depth = depth
                if depth == best_depth:
                    found_solutions.append(route)
                continue

            if depth >= max_depth:
                continue

            for t in tools:
                if t.name not in route and all(req in current_state for req in t.required_inputs):
                    next_state = set(current_state) | set(t.produced_outputs)
                    queue.append((route + [t.name], next_state))

        if not found_solutions:
            return None

        # Canonical tie-breaking: sort candidate routes lexicographically
        found_solutions.sort(key=lambda r: tuple(r))
        canonical_route = found_solutions[0]

        # Generate execution trace
        trace_steps: List[ExecutionTraceStep] = []
        c_state = set(initial_state)
        for step_idx, tool_name in enumerate(canonical_route, start=1):
            t = tools_map[tool_name]
            c_state = c_state | set(t.produced_outputs)
            trace_steps.append(
                ExecutionTraceStep(
                    step=step_idx,
                    tool=tool_name,
                    consumed=tuple(sorted(t.required_inputs)),
                    produced=tuple(sorted(t.produced_outputs)),
                    cumulative_state=tuple(sorted(c_state)),
                )
            )

        return {
            "route": tuple(canonical_route),
            "terminal_tool": canonical_route[-1],
            "depth_K": len(canonical_route),
            "trace": tuple(trace_steps),
            "initial_state": tuple(sorted(initial_state)),
            "goal": goal,
        }


class ProceduralLaneGenerator:
    """Generator constructing verifiable procedural routing problems with guaranteed K >= 2."""

    def __init__(self, oracle: Optional[DeterministicToolRoutingOracle] = None):
        self.oracle = oracle if oracle is not None else DeterministicToolRoutingOracle()

    def generate_instance(
        self,
        domain: str,
        seed: int,
        target_depth_K: int = 2,
        num_distractors: int = 2,
    ) -> ProceduralProblemInstance:
        """Generate a procedural problem instance enforcing minimal route depth K.

        Args:
            domain: Domain family ('api_workflow', 'data_pipeline', 'security_ops', 'robotics_control').
            seed: Deterministic random seed.
            target_depth_K: Desired reasoning depth (strictly >= 2).
            num_distractors: Number of distractor tools to append to prompt.

        Returns:
            ProceduralProblemInstance with verified K >= 2 and Gemma formatting.
        """
        if target_depth_K < 2:
            raise ValueError(f"target_depth_K must be >= 2 for multi-step deliberation, got {target_depth_K}")

        if domain not in DOMAIN_CATALOGUES:
            raise KeyError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CATALOGUES.keys())}")

        core_tools = list(DOMAIN_CATALOGUES[domain]["core"])
        distractor_pool = list(DOMAIN_CATALOGUES[domain]["distractors"])

        # If target_depth_K exceeds the core tools length, extend with deterministic sequential tools
        if target_depth_K > len(core_tools):
            extended_tools = list(core_tools)
            last_out = extended_tools[-1].produced_outputs[0]
            extra_needed = target_depth_K - len(core_tools)
            for ext in range(extra_needed):
                next_out = f"derived_stage_{ext + 1}"
                t_ext = ToolDefinition(
                    name=f"stage_processor_{ext + 1}",
                    required_inputs=(last_out,),
                    produced_outputs=(next_out,),
                    description=f"Sequential reasoning extension stage {ext + 1}",
                )
                extended_tools.append(t_ext)
                last_out = next_out
            available_chain_pool = extended_tools
        else:
            available_chain_pool = core_tools

        max_attempts = 100
        for attempt in range(max_attempts):
            attempt_rng = random.Random(seed + attempt * 10007)
            max_start = len(available_chain_pool) - target_depth_K
            start_idx = attempt_rng.randint(0, max_start)
            chain_tools = available_chain_pool[start_idx : start_idx + target_depth_K]

            # Collect unfulfilled inputs along the chain
            init_state: Set[str] = set()
            produced_so_far: Set[str] = set()
            for t in chain_tools:
                for req in t.required_inputs:
                    if req not in produced_so_far:
                        init_state.add(req)
                produced_so_far.update(t.produced_outputs)

            # Add deterministic instance context token to guarantee 0% cross-split collisions
            init_state.add(f"context_tag_{seed}")

            goal = chain_tools[-1].produced_outputs[0]

            # Pick distractors
            selected_distractors = attempt_rng.sample(
                distractor_pool, min(num_distractors, len(distractor_pool))
            )

            # Available tool list in problem
            problem_tools = list(chain_tools) + list(selected_distractors)
            attempt_rng.shuffle(problem_tools)

            # Solve via oracle
            solution = self.oracle.solve(problem_tools, init_state, goal, max_depth=target_depth_K + 2)
            if solution is not None and solution["depth_K"] == target_depth_K:
                expected_route = solution["route"]
                terminal_tool = solution["terminal_tool"]
                trace = solution["trace"]
                depth_K = solution["depth_K"]

                sample_id = f"mtr_{domain}_s{seed}_k{depth_K}"
                target_solution_dict = {
                    "route": list(expected_route),
                    "terminal": terminal_tool,
                }
                ground_truth_json = json.dumps(target_solution_dict, separators=(", ", ": "))

                # Gemma 2B instruction prompt
                prompt = self._format_gemma_prompt(
                    tools=problem_tools,
                    initial_state=init_state,
                    target_goal=goal,
                )

                fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "domain": domain,
                            "initial_state": sorted(init_state),
                            "goal": goal,
                            "K": target_depth_K,
                            "route": list(expected_route),
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()

                instance_text = f"Initial State: {sorted(init_state)} Target Goal: {goal}"

                metadata = {
                    "domain": domain,
                    "seed": seed,
                    "fingerprint": fingerprint,
                    "instance_text": instance_text,
                    "num_tools": len(problem_tools),
                    "distractors": [t.name for t in selected_distractors],
                    "all_tool_names": [t.name for t in problem_tools],
                }

                return ProceduralProblemInstance(
                    sample_id=sample_id,
                    domain=domain,
                    prompt=prompt,
                    ground_truth_json=ground_truth_json,
                    expected_route=expected_route,
                    terminal_tool=terminal_tool,
                    depth_K=depth_K,
                    initial_state=tuple(sorted(init_state)),
                    target_goal=goal,
                    trace=trace,
                    metadata=metadata,
                )

        raise RuntimeError(
            f"Failed to generate valid instance for domain {domain} with K={target_depth_K} after {max_attempts} attempts"
        )

    def _format_gemma_prompt(
        self,
        tools: Sequence[ToolDefinition],
        initial_state: Set[str],
        target_goal: str,
    ) -> str:
        """Format problem prompt matching Google Gemma 2B instruction template."""
        tools_str = "\n".join(
            f"- {t.name}: requires [{', '.join(t.required_inputs)}], produces [{', '.join(t.produced_outputs)}]"
            for t in tools
        )
        init_str = ", ".join(sorted(initial_state))

        # System role instructions embedded inside user turn per Gemma 2B requirement
        return (
            "<start_of_turn>user\n"
            "You are an autonomous execution planner. Given the available tool registry, "
            "determine the minimal valid sequence of tools to achieve the target goal from the initial state.\n\n"
            f"Available Tools:\n{tools_str}\n\n"
            f"Initial State: [{init_str}]\n"
            f"Target Goal: {target_goal}\n\n"
            "Respond with a JSON object containing keys 'route' (list of tool names in execution order) "
            "and 'terminal' (the final tool producing the goal).<end_of_turn>\n"
            "<start_of_turn>model\n"
        )


class ProceduralVerifier:
    """Independent ground-truth verifier enforcing Non-Negotiable Evidence Rule 1 and Rule 2."""

    @staticmethod
    def verify(
        prediction_str: str,
        expected_route: Tuple[str, ...],
        tools: Optional[Sequence[ToolDefinition]] = None,
        initial_state: Optional[Sequence[str]] = None,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify model prediction string against expected route and tool semantics.

        Args:
            prediction_str: Raw text generated by model.
            expected_route: Canonical ground truth route.
            tools: Optional tool catalog for operational semantics verification.
            initial_state: Optional initial state.
            goal: Optional target goal.

        Returns:
            Dict containing validation results and diagnostic error messages.
        """
        errors: List[str] = []

        cleaned_text = prediction_str.strip()
        cleaned_text = re.sub(r"<end_of_turn>.*$", "", cleaned_text)
        cleaned_text = re.sub(r"<turn|>.*$", "", cleaned_text)
        cleaned_text = re.sub(r"<eos>.*$", "", cleaned_text)
        if "<channel|>" in cleaned_text:
            cleaned_text = cleaned_text.split("<channel|>")[-1].strip()

        json_match = re.search(r"\{.*?\}", cleaned_text, re.DOTALL)
        if not json_match:
            return {
                "is_valid": False,
                "exact_match": False,
                "predicted_route": [],
                "expected_route": list(expected_route),
                "terminal_tool": None,
                "errors": ["Failed to extract JSON object from output string"],
            }

        try:
            parsed = json.loads(json_match.group(0))
        except Exception as e:
            return {
                "is_valid": False,
                "exact_match": False,
                "predicted_route": [],
                "expected_route": list(expected_route),
                "terminal_tool": None,
                "errors": [f"JSON decode error: {e}"],
            }

        pred_route = parsed.get("route")
        if not isinstance(pred_route, list) or not all(isinstance(x, str) for x in pred_route):
            return {
                "is_valid": False,
                "exact_match": False,
                "predicted_route": [],
                "expected_route": list(expected_route),
                "terminal_tool": None,
                "errors": ["Key 'route' must be a list of tool name strings"],
            }

        pred_terminal = parsed.get("terminal")
        exact_match = (tuple(pred_route) == tuple(expected_route)) and (
            pred_terminal == expected_route[-1] if expected_route else True
        )

        is_operationally_valid = True
        if tools is not None and initial_state is not None and goal is not None:
            tools_map = {t.name: t for t in tools}
            current_state = set(initial_state)
            for step_tool in pred_route:
                if step_tool not in tools_map:
                    is_operationally_valid = False
                    errors.append(f"Tool '{step_tool}' not in tool registry")
                    break
                t = tools_map[step_tool]
                if not all(req in current_state for req in t.required_inputs):
                    is_operationally_valid = False
                    missing = [req for req in t.required_inputs if req not in current_state]
                    errors.append(f"Tool '{step_tool}' missing required inputs: {missing}")
                    break
                current_state.update(t.produced_outputs)

            if is_operationally_valid and goal not in current_state:
                is_operationally_valid = False
                errors.append(f"Route executed but target goal '{goal}' not produced")

        is_valid = exact_match or (is_operationally_valid and len(errors) == 0)

        return {
            "is_valid": is_valid,
            "exact_match": exact_match,
            "predicted_route": pred_route,
            "expected_route": list(expected_route),
            "terminal_tool": pred_terminal,
            "errors": errors,
        }


__all__ = [
    "ToolDefinition",
    "ExecutionTraceStep",
    "ProceduralProblemInstance",
    "DOMAIN_CATALOGUES",
    "DeterministicToolRoutingOracle",
    "ProceduralLaneGenerator",
    "ProceduralVerifier",
]
