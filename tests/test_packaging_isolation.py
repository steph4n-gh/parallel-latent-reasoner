"""Test packaging isolation and zero-monolith import constraints for PRLR."""

import ast
from pathlib import Path

import pytest


def test_zero_monolith_imports():
    """Verify that no Python file in projects/parallel_latent_reasoner imports qan_transformers."""
    pkg_root = Path(__file__).resolve().parents[1]
    python_files = list(pkg_root.rglob("*.py"))
    assert len(python_files) >= 5, "Expected at least 5 python files in the standalone package."

    forbidden_module = "qan_transformers"
    violations: list[str] = []

    for p in python_files:
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content, filename=str(p))
        except SyntaxError as e:
            violations.append(f"Syntax error in {p}: {e}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == forbidden_module or alias.name.startswith(f"{forbidden_module}."):
                        violations.append(f"{p}:{node.lineno} imports '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module == forbidden_module or (node.module and node.module.startswith(f"{forbidden_module}.")):
                    violations.append(f"{p}:{node.lineno} imports from '{node.module}'")

    assert not violations, f"Forbidden imports from '{forbidden_module}' found:\n" + "\n".join(violations)


def test_package_exports():
    """Verify that all core components are cleanly importable from parallel_latent_reasoner."""
    import parallel_latent_reasoner as prlr

    expected_symbols = [
        "GemmaLatentConfig",
        "sinusoidal_step_embedding",
        "MLXRMSNorm",
        "MLXAdaRMSNorm",
        "MLXGemmaAttention",
        "MLXGemmaMLP",
        "MLXRecurrentGemmaBlock",
        "MLXPreludeProjection",
        "MLXCodaLMHead",
        "MLXCompactGemmaModel",
        "DeliberationResult",
        "MLXParallelLatentEngine",
        "compute_effective_rank",
        "compute_slot_cosine_similarity",
        "compute_slot_velocity",
        "analyze_deliberation_trajectory",
        "detect_limit_cycle",
        "DynamicDeliberationGate",
        "DynamicConsensusEGate",
        "GateTelemetry",
        "GateDecision",
        "GemmaDeliberationPipeline",
        "DeliberationPipelineOutput",
    ]

    for sym in expected_symbols:
        assert hasattr(prlr, sym), f"Expected symbol '{sym}' missing from top-level package exports."
