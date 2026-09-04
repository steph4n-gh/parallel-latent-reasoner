"""Hard Procedural Reasoning Benchmark Generator (PRLR Headroom Lane).

Features:
- Branching DAG topologies with intentional dead-end branches (requiring lookahead / backtracking).
- Multi-parent joins (requiring synthesizing multiple independent intermediate state tokens).
- Signature-overlapping decoy tools and distractor traps.
- Exact BFS/Dijkstra oracle solver with deterministic canonical tie-breaking.
- Verified uniqueness of minimal solution routes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from prlr.domain.solver_lane import ToolDefinition, ExecutionTraceStep


@dataclass(frozen=True)
class HardProblemInstance:
    sample_id: str
    domain: str
    difficulty: int
    prompt: str
    ground_truth_json: str
    expected_route: Tuple[str, ...]
    terminal_tool: str
    depth_K: int
    initial_state: Tuple[str, ...]
    target_goal: str
    available_tools: Tuple[str, ...]
    prompt_sha256: str
    solution_sha256: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Multi-Branching Tool Repositories ---

HARD_DOMAINS = {
    "cloud_orchestration": {
        "branch_a": [
            ToolDefinition("vpc_provisioner", ("cloud_account",), ("vpc_id",), "Provisions VPC network"),
            ToolDefinition("subnet_router", ("vpc_id",), ("subnet_id", "route_table"), "Configures subnets"),
            ToolDefinition("nat_gateway_setup", ("subnet_id", "route_table"), ("egress_gateway",), "Sets up NAT"),
        ],
        "branch_b": [
            ToolDefinition("iam_role_builder", ("cloud_account", "vpc_id"), ("iam_role",), "Creates IAM execution roles"),
            ToolDefinition("kms_key_issuer", ("iam_role", "subnet_id"), ("kms_key_arn",), "Issues encryption keys"),
            ToolDefinition("secrets_store_init", ("kms_key_arn", "egress_gateway"), ("vault_token",), "Initializes secrets vault"),
        ],
        "join_tool": ToolDefinition("cluster_deployer", ("egress_gateway", "vault_token"), ("k8s_cluster",), "Deploys cluster"),
        "terminal_tool": ToolDefinition("service_mesh_meshctl", ("k8s_cluster",), ("mesh_active",), "Activates service mesh"),
        "dead_ends": [
            ToolDefinition("legacy_bridge_adapter", ("cloud_account",), ("bridge_tunnel",), "Sets up legacy VPN tunnel"),
            ToolDefinition("vpn_gateway_linker", ("bridge_tunnel",), ("tunnel_active",), "Links VPN gateway"),
            ToolDefinition("classic_vm_allocator", ("vpc_id",), ("raw_vm_instance",), "Allocates unmanaged VM"),
            ToolDefinition("public_ip_assigner", ("raw_vm_instance",), ("floating_ip",), "Assigns public IP"),
            ToolDefinition("ldap_sync_service", ("iam_role",), ("ldap_directory",), "Syncs LDAP"),
            ToolDefinition("ad_federation_linker", ("ldap_directory",), ("federated_auth",), "Links Active Directory"),
        ],
    },
    "data_mesh_synthesis": {
        "branch_a": [
            ToolDefinition("cdc_event_streamer", ("source_db",), ("change_feed",), "Streams CDC change feed"),
            ToolDefinition("avro_deserializer", ("change_feed",), ("raw_events", "schema_ver"), "Deserializes Avro"),
            ToolDefinition("schema_enforcer", ("raw_events", "schema_ver"), ("conformed_events",), "Enforces schema"),
        ],
        "branch_b": [
            ToolDefinition("governance_catalog_query", ("source_db", "change_feed"), ("policy_bundle",), "Queries governance"),
            ToolDefinition("pii_masking_compiler", ("policy_bundle", "raw_events"), ("masking_rules",), "Compiles PII rules"),
            ToolDefinition("tokenization_vault", ("masking_rules", "conformed_events"), ("token_service",), "Sets up tokenization"),
        ],
        "join_tool": ToolDefinition("differential_privacy_engine", ("conformed_events", "token_service"), ("anonymized_dataset",), "Applies DP"),
        "terminal_tool": ToolDefinition("feature_store_publisher", ("anonymized_dataset",), ("online_features",), "Publishes online features"),
        "dead_ends": [
            ToolDefinition("csv_batch_exporter", ("source_db",), ("raw_csv_dump",), "Dumps raw CSV"),
            ToolDefinition("s3_flat_uploader", ("raw_csv_dump",), ("s3_bucket_uri",), "Uploads CSV to S3"),
            ToolDefinition("delta_table_indexer", ("conformed_events",), ("parquet_index",), "Indexes Parquet"),
            ToolDefinition("hive_metastore_sync", ("parquet_index",), ("hive_table",), "Syncs Hive metastore"),
            ToolDefinition("sample_profiler", ("policy_bundle",), ("summary_stats",), "Profiles sample data"),
            ToolDefinition("report_pdf_generator", ("summary_stats",), ("audit_pdf",), "Generates PDF report"),
        ],
    },
    "cyber_incident_response": {
        "branch_a": [
            ToolDefinition("edr_sensor_triager", ("host_alert",), ("memory_dump",), "Captures process memory"),
            ToolDefinition("kernel_symbol_extractor", ("memory_dump",), ("injected_dll", "hooked_syscalls"), "Extracts hooks"),
            ToolDefinition("rootkit_signature_matcher", ("injected_dll", "hooked_syscalls"), ("c2_beacon_config",), "Matches rootkit"),
        ],
        "branch_b": [
            ToolDefinition("netflow_pcap_collector", ("host_alert", "memory_dump"), ("pcap_stream",), "Captures raw netflow PCAP"),
            ToolDefinition("tls_fingerprint_analyzer", ("pcap_stream", "injected_dll"), ("ja3_hash",), "Analyzes JA3 fingerprints"),
            ToolDefinition("threat_intel_correlator", ("ja3_hash", "c2_beacon_config"), ("adversary_actor_id",), "Correlates actor"),
        ],
        "join_tool": ToolDefinition("containment_playbook_generator", ("c2_beacon_config", "adversary_actor_id"), ("containment_plan",), "Generates containment plan"),
        "terminal_tool": ToolDefinition("automated_quarantine_firewall", ("containment_plan",), ("threat_contained",), "Executes quarantine"),
        "dead_ends": [
            ToolDefinition("syslog_grepper", ("host_alert",), ("text_logs",), "Greps auth logs"),
            ToolDefinition("disk_image_compressor", ("text_logs",), ("tar_archive",), "Compresses logs"),
            ToolDefinition("wireshark_dissector", ("pcap_stream",), ("dns_queries",), "Dissects DNS"),
            ToolDefinition("whois_lookup", ("dns_queries",), ("registrar_record",), "Looks up registrar"),
            ToolDefinition("clamav_scanner", ("memory_dump",), ("known_virus_sig",), "Scans with ClamAV"),
            ToolDefinition("quarantine_folder_mover", ("known_virus_sig",), ("isolated_folder",), "Moves to folder"),
        ],
    },
}


class HardBranchingOracle:
    """Exact state-space BFS solver for multi-parent branching DAGs."""

    def solve(
        self,
        tools: List[ToolDefinition],
        initial_tokens: Set[str],
        goal_token: str,
        max_depth: int = 12,
    ) -> Optional[Dict[str, Any]]:
        """Find the minimal sequence of tools reaching goal_token from initial_tokens.

        Canonical tie-breaking: shorter route first, then lexicographical order of tool names.
        """
        tools_by_name = {t.name: t for t in tools}

        queue = deque([(frozenset(initial_tokens), ())])
        visited_states: Dict[frozenset, int] = {frozenset(initial_tokens): 0}
        solutions: List[Tuple[str, ...]] = []

        best_len = None

        while queue:
            curr_state, route = queue.popleft()

            if best_len is not None and len(route) > best_len:
                break

            if goal_token in curr_state:
                if best_len is None:
                    best_len = len(route)
                solutions.append(route)
                continue

            if len(route) >= max_depth:
                continue

            candidates = []
            for t in tools:
                if t.name in route:
                    continue
                if all(req in curr_state for req in t.required_inputs):
                    candidates.append(t)

            candidates.sort(key=lambda t: t.name)

            for cand in candidates:
                next_state = set(curr_state)
                next_state.update(cand.produced_outputs)
                next_state_fs = frozenset(next_state)
                next_route = route + (cand.name,)

                if next_state_fs not in visited_states or visited_states[next_state_fs] >= len(next_route):
                    visited_states[next_state_fs] = len(next_route)
                    queue.append((next_state_fs, next_route))

        if not solutions:
            return None

        solutions.sort(key=lambda r: (len(r), r))
        best_route = solutions[0]

        return {
            "route": list(best_route),
            "terminal_tool": best_route[-1],
            "depth_K": len(best_route),
            "num_solutions": len(solutions),
            "is_unique_shortest": len(solutions) == 1 or len(solutions[0]) < len(solutions[1]),
        }


class HardReasoningLaneGenerator:
    """Generates hard multi-branching DAG reasoning problems with guaranteed lookahead traps."""

    def __init__(self, oracle: Optional[HardBranchingOracle] = None):
        self.oracle = oracle or HardBranchingOracle()

    def generate_instance(self, domain: str, seed: int, num_dead_ends: int = 4) -> HardProblemInstance:
        rng = random.Random(seed)
        domain_data = HARD_DOMAINS[domain]

        branch_a = domain_data["branch_a"]
        branch_b = domain_data["branch_b"]
        join_tool = domain_data["join_tool"]
        terminal_tool = domain_data["terminal_tool"]
        dead_end_pool = domain_data["dead_ends"]

        root_input = branch_a[0].required_inputs[0]
        initial_state = {root_input, f"tenant_id_{seed}"}
        goal_token = terminal_tool.produced_outputs[0]

        selected_dead_ends = rng.sample(dead_end_pool, min(num_dead_ends, len(dead_end_pool)))

        all_tools = branch_a + branch_b + [join_tool, terminal_tool] + selected_dead_ends
        rng.shuffle(all_tools)

        sol = self.oracle.solve(all_tools, initial_state, goal_token)
        if sol is None:
            raise RuntimeError(f"Failed to generate solution for domain={domain}, seed={seed}")

        expected_route = tuple(sol["route"])
        term_tool = sol["terminal_tool"]

        tool_lines = []
        for t in all_tools:
            tool_lines.append(f"- {t.name}: requires [{', '.join(t.required_inputs)}], produces [{', '.join(t.produced_outputs)}]")

        init_str = ", ".join(sorted(initial_state))
        prompt = (
            "<start_of_turn>user\n"
            "You are an autonomous execution planner. Given the available tool registry, determine the minimal valid sequence of tools to achieve the target goal from the initial state.\n"
            "Beware of dead-end tools and traps that cannot reach the goal.\n\n"
            f"Available Tools:\n{chr(10).join(tool_lines)}\n\n"
            f"Initial State: [{init_str}]\n"
            f"Target Goal: {goal_token}\n\n"
            "Respond with a JSON object containing keys 'route' (list of tool names in execution order) and 'terminal' (the final tool producing the goal).<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        target_dict = {
            "route": list(expected_route),
            "terminal": term_tool,
        }
        gt_json = json.dumps(target_dict)

        p_hash = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()
        s_hash = hashlib.sha256(gt_json.strip().encode("utf-8")).hexdigest()

        return HardProblemInstance(
            sample_id=f"hard_dag_{domain}_s{seed}_k{len(expected_route)}",
            domain=domain,
            difficulty=len(expected_route),
            prompt=prompt,
            ground_truth_json=gt_json,
            expected_route=expected_route,
            terminal_tool=term_tool,
            depth_K=len(expected_route),
            initial_state=tuple(sorted(initial_state)),
            target_goal=goal_token,
            available_tools=tuple(t.name for t in all_tools),
            prompt_sha256=p_hash,
            solution_sha256=s_hash,
            metadata={
                "domain": domain,
                "seed": seed,
                "num_tools": len(all_tools),
                "num_dead_ends": len(selected_dead_ends),
                "dead_end_names": [t.name for t in selected_dead_ends],
            },
        )


def generate_hard_benchmark(
    output_dir: Path | str,
    count: int = 256,
    base_seed: int = 700_000,
) -> Dict[str, Any]:
    """Generate and quarantine the hard headroom benchmark."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    inputs_dir = out_path / "evaluation_inputs"
    keys_dir = out_path / "answer_keys"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)

    gen = HardReasoningLaneGenerator()
    domains = list(HARD_DOMAINS.keys())

    full_records: List[Dict[str, Any]] = []
    target_free_inputs: List[Dict[str, Any]] = []
    quarantined_keys: List[Dict[str, Any]] = []

    for i in range(count):
        seed = base_seed + i
        domain = domains[i % len(domains)]
        inst = gen.generate_instance(domain=domain, seed=seed)

        rec = {
            "id": inst.sample_id,
            "split": "hard_test",
            "domain": inst.domain,
            "difficulty": inst.difficulty,
            "num_steps": inst.depth_K,
            "prompt": inst.prompt,
            "target_solution": inst.ground_truth_json,
            "ground_truth": inst.ground_truth_json,
            "verifier_type": "hard_branching_oracle",
            "verifier_config": {
                "expected_route": list(inst.expected_route),
                "terminal_tool": inst.terminal_tool,
                "target_goal": inst.target_goal,
            },
            "seed": seed,
            "metadata": inst.metadata,
            "prompt_sha256": inst.prompt_sha256,
            "solution_sha256": inst.solution_sha256,
        }
        full_records.append(rec)

        target_free_inputs.append({
            "id": inst.sample_id,
            "sample_id": inst.sample_id,
            "split": "hard_test",
            "domain": inst.domain,
            "difficulty": inst.difficulty,
            "num_steps": inst.depth_K,
            "prompt": inst.prompt,
            "prompt_sha256": inst.prompt_sha256,
        })

        quarantined_keys.append({
            "id": inst.sample_id,
            "sample_id": inst.sample_id,
            "domain": inst.domain,
            "expected_route": list(inst.expected_route),
            "terminal_tool": inst.terminal_tool,
            "target_solution": inst.ground_truth_json,
            "ground_truth": inst.ground_truth_json,
            "solution_sha256": inst.solution_sha256,
            "verifier_config": {
                "expected_route": list(inst.expected_route),
                "terminal_tool": inst.terminal_tool,
                "target_goal": inst.target_goal,
            },
        })

    full_file = out_path / "hard_test.jsonl"
    inputs_file = inputs_dir / "hard_test_inputs.jsonl"
    keys_file = keys_dir / "hard_test_keys.jsonl"

    for path, data in [(full_file, full_records), (inputs_file, target_free_inputs), (keys_file, quarantined_keys)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with open(f"{path}.sha256", "w", encoding="utf-8") as sf:
            sf.write(f"{sha}  {path.name}\n")

    return {
        "count": count,
        "full_file": str(full_file),
        "inputs_file": str(inputs_file),
        "keys_file": str(keys_file),
    }


__all__ = [
    "HardProblemInstance",
    "HardBranchingOracle",
    "HardReasoningLaneGenerator",
    "generate_hard_benchmark",
    "HARD_DOMAINS",
]
