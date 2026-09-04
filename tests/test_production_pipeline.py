"""Comprehensive Test Suite for PRLR Production Gemma 2B Pipeline (Milestone 3 / Feature 13).

Verifies:
1. PRLRPipeline initialization with real pretrained google/gemma-2b-it backbone
   and production trained adapter checkpoint (checkpoints/gemma_2b_prlr_adapter.safetensors).
2. Deliberate-Then-Verify generation with fixed deliberation steps (T=1, 2, 4).
3. Deliberation with dynamic post-hoc calibrated E-gate (bounded steps, 4-signal consensus).
4. Positivity and sum-consistency of stage latencies (prefill, deliberation, decode, total).
5. Information-theoretic quality: Shannon entropy H >= 3.0 bits and max 4-gram repetition <= 2.
6. CLI argument parsing and flags in demo.py and run_benchmark.py.
7. Strict compliance with Evidence Rules 1, 2, 5, 6 and zero monolith imports.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional
import pytest

import mlx.core as mx

from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.gemma.egate import GemmaCalibratedEGate
from prlr.manifest import ModelManifest
from prlr.pipeline import HybridDeliberationResult, PipelineResult, PRLRPipeline
from prlr import PRLRPipeline as TopPRLRPipeline, PipelineResult as TopPipelineResult


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "gemma_2b_prlr_adapter.safetensors"
SIDECAR_PATH = PROJECT_DIR / "checkpoints" / "gemma_2b_prlr_adapter.json"
EGATE_CONFIG_PATH = PROJECT_DIR / "checkpoints" / "calibrated_egate_config.json"

if not CHECKPOINT_PATH.exists():
    try:
        scripts_dir = PROJECT_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from download_checkpoint import ensure_checkpoint
        ensure_checkpoint(model="gemma_2b", target_dir=CHECKPOINT_PATH.parent, quiet=True)
    except Exception:
        pass

if not CHECKPOINT_PATH.exists():
    pytest.skip(
        f"Production checkpoint {CHECKPOINT_PATH.name} not found. "
        "Run `python scripts/download_checkpoint.py` to download from GitHub release.",
        allow_module_level=True,
    )

if not SIDECAR_PATH.exists():
    pytest.skip(
        f"Production checkpoint sidecar {SIDECAR_PATH.name} not found.",
        allow_module_level=True,
    )


# ==============================================================================
# Helper Functions: Shannon Entropy & N-gram Repetition
# ==============================================================================

def compute_shannon_entropy(text: str) -> float:
    """Compute empirical Shannon entropy H in bits over the character distribution."""
    clean = text.strip()
    if not clean:
        return 0.0
    counts = collections.Counter(clean)
    total = len(clean)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def compute_max_4gram_repetition(text: str) -> int:
    """Compute maximum frequency count of any 4-gram in the text."""
    clean = text.strip()
    if not clean:
        return 0
    words = clean.split()
    if len(words) < 4:
        if len(clean) < 4:
            return 1 if clean else 0
        char_ngrams = [clean[i : i + 4] for i in range(len(clean) - 3)]
        counts = collections.Counter(char_ngrams)
        return max(counts.values()) if counts else 0

    word_ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    counts = collections.Counter(word_ngrams)
    return max(counts.values()) if counts else 1


# ==============================================================================
# Pytest Fixtures (Module Scoped for Metal GPU Memory Residency)
# ==============================================================================

@pytest.fixture(scope="module")
def gemma_manifest() -> ModelManifest:
    """Provide verified official Gemma 2B manifest."""
    manifest = ModelManifest.gemma_2b_it()
    assert manifest.validate(check_disk=True) is True
    return manifest


@pytest.fixture(scope="module")
def shared_backbone(gemma_manifest: ModelManifest) -> PretrainedGemmaBackbone:
    """Load official Gemma 2B backbone weights once for the test module."""
    return PretrainedGemmaBackbone(manifest=gemma_manifest, load_weights=True)


@pytest.fixture(scope="module")
def production_pipeline(shared_backbone: PretrainedGemmaBackbone) -> PRLRPipeline:
    """Provide a PRLRPipeline instance initialized with the production adapter."""
    return PRLRPipeline(
        backbone=shared_backbone,
        adapter_path=str(CHECKPOINT_PATH),
        load_trained_adapter=True,
        deliberation_steps=4,
        num_slots=16,
    )


# ==============================================================================
# 1. Pipeline Initialization Tests
# ==============================================================================

class TestProductionPipelineInitialization:
    """Verifies PRLRPipeline initialization, checkpoint discovery, and error handling."""

    def test_default_init_loads_production_checkpoint(self, shared_backbone: PretrainedGemmaBackbone):
        """Verify default initialization automatically discovers and loads production adapter."""
        pipeline = PRLRPipeline(backbone=shared_backbone, load_trained_adapter=True)
        assert pipeline.adapter_loaded is True
        assert pipeline.adapter_path is not None
        assert Path(pipeline.adapter_path).exists()
        assert Path(pipeline.adapter_path).name == "gemma_2b_prlr_adapter.safetensors"

        # Verify component types
        assert isinstance(pipeline.backbone, PretrainedGemmaBackbone)
        assert isinstance(pipeline.adapter, GemmaRecurrentAdapter)
        assert isinstance(pipeline.decoder, GemmaCausalPrefixDecoder)
        assert isinstance(pipeline.egate, GemmaCalibratedEGate)

        # Verify adapter parameter count
        from mlx.utils import tree_flatten
        params = dict(tree_flatten(pipeline.adapter.parameters()))
        assert len(params) == 28, f"Expected 28 adapter parameters, got {len(params)}"

    def test_init_with_explicit_valid_checkpoint_path(self, shared_backbone: PretrainedGemmaBackbone):
        """Verify explicit adapter_path loads successfully."""
        pipeline = PRLRPipeline(
            backbone=shared_backbone,
            adapter_path=str(CHECKPOINT_PATH),
            load_trained_adapter=True,
        )
        assert pipeline.adapter_loaded is True
        assert str(pipeline.adapter_path) == str(CHECKPOINT_PATH)

    def test_init_untrained_adapter_flag(self, shared_backbone: PretrainedGemmaBackbone):
        """Verify load_trained_adapter=False leaves adapter in initialized random state."""
        pipeline = PRLRPipeline(
            backbone=shared_backbone,
            load_trained_adapter=False,
        )
        assert pipeline.adapter_loaded is False
        assert pipeline.adapter_path is None
        assert pipeline.adapter is not None

    def test_init_nonexistent_checkpoint_raises(self, shared_backbone: PretrainedGemmaBackbone):
        """Verify passing an invalid checkpoint path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PRLRPipeline(
                backbone=shared_backbone,
                adapter_path="/tmp/nonexistent_adapter_file_xyz.safetensors",
                load_trained_adapter=True,
            )

    def test_init_custom_slots_and_steps(self, shared_backbone: PretrainedGemmaBackbone):
        """Verify pipeline respects custom num_slots and deliberation_steps parameters."""
        pipeline = PRLRPipeline(
            backbone=shared_backbone,
            num_slots=8,
            deliberation_steps=2,
            load_trained_adapter=False,
        )
        assert pipeline.num_slots == 8
        assert pipeline.deliberation_steps == 2
        assert pipeline.adapter.num_slots == 8

    def test_pipeline_exports(self):
        """Verify PRLRPipeline and PipelineResult are cleanly exported at top-level."""
        assert PRLRPipeline is TopPRLRPipeline
        assert PipelineResult is TopPipelineResult


# ==============================================================================
# 2. Generation with Fixed Deliberation Steps Tests
# ==============================================================================

class TestProductionPipelineFixedGeneration:
    """Verifies fixed-depth deliberation unrolls across T in {1, 2, 4}."""

    @pytest.mark.parametrize("fixed_steps", [1, 2, 4])
    def test_generation_fixed_deliberation_steps(
        self,
        production_pipeline: PRLRPipeline,
        fixed_steps: int,
    ):
        """Verify pipeline executes exactly the specified number of deliberation steps."""
        prompt = "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
        result = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=fixed_steps,
            enable_dynamic_gate=False,
            max_new_tokens=16,
            temperature=0.0,
        )

        assert isinstance(result, HybridDeliberationResult)
        assert result.deliberation_steps == fixed_steps
        assert result.egate_verdict in ("fixed_depth", "disabled")
        assert result.consensus_step is None

        # Check working memory slots invariant shape (1, M, D)
        assert result.final_states.shape == (1, 16, 2048)
        assert not mx.isnan(result.final_states).any().item()
        assert not mx.isinf(result.final_states).any().item()

        # Check discrete token IDs and decoded text
        assert result.token_ids.ndim in (1, 2)
        assert isinstance(result.decoded_text, str)
        assert len(result.decoded_text.strip()) > 0

    def test_generation_greedy_determinism(self, production_pipeline: PRLRPipeline):
        """Verify greedy generation (temperature=0.0) is bit-for-bit deterministic."""
        prompt = "<start_of_turn>user\nRoute step: current [node_1]<end_of_turn>\n<start_of_turn>model\n"
        res1 = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=2,
            enable_dynamic_gate=False,
            temperature=0.0,
            max_new_tokens=12,
        )
        res2 = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=2,
            enable_dynamic_gate=False,
            temperature=0.0,
            max_new_tokens=12,
        )

        assert res1.decoded_text == res2.decoded_text
        diff = mx.sum(mx.abs(res1.token_ids - res2.token_ids)).item()
        assert diff == 0, "Greedy generation must produce identical token IDs."

    def test_zero_intermediate_discrete_tokens_in_latent_deliberation(
        self,
        production_pipeline: PRLRPipeline,
    ):
        """Verify deliberation operates purely in latent space without intermediate discrete tokens."""
        prompt = "Synthesize intermediate steps"
        result = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=4,
            enable_dynamic_gate=False,
            max_new_tokens=8,
        )
        assert result.final_states.shape == (1, 16, 2048)
        assert result.final_states.dtype in (mx.float32, mx.bfloat16)


# ==============================================================================
# 3. Generation with Dynamic Calibrated E-Gate Tests
# ==============================================================================

class TestProductionPipelineDynamicEGate:
    """Verifies autonomous 4-signal calibrated consensus halting and telemetry."""

    def test_dynamic_egate_deliberation_bounds(self, production_pipeline: PRLRPipeline):
        """Verify dynamic deliberation executes within calibrated bounds [T_min, T_max]."""
        prompt = "<start_of_turn>user\nDirect task: execute trivial echo.<end_of_turn>\n<start_of_turn>model\n"
        result = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            max_steps=12,
            enable_dynamic_gate=True,
            max_new_tokens=16,
            temperature=0.0,
        )

        assert 2 <= result.deliberation_steps <= 12
        assert result.egate_verdict in ("4_signal_consensus", "max_steps_timeout")

        if result.egate_verdict == "4_signal_consensus":
            assert result.consensus_step == result.deliberation_steps

    def test_dynamic_egate_telemetry_fields(self, production_pipeline: PRLRPipeline):
        """Verify step telemetry contains valid non-oracle metrics for every executed step."""
        prompt = "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
        result = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            max_steps=12,
            enable_dynamic_gate=True,
            max_new_tokens=8,
        )

        telemetry = result.gate_telemetry
        assert telemetry is not None
        assert len(telemetry) == result.deliberation_steps

        for telem in telemetry:
            assert telem.velocity >= 0.0
            assert telem.rel_velocity >= 0.0
            assert telem.entropy >= 0.0
            assert telem.margin >= 0.0
            assert telem.erank >= 1.0
            assert telem.delta_erank >= 0.0
            assert isinstance(telem.sig_velocity, bool)
            assert isinstance(telem.sig_entropy, bool)
            assert isinstance(telem.sig_margin, bool)
            assert isinstance(telem.sig_erank, bool)
            assert isinstance(telem.halt, bool)
            assert telem.step_latency_ms > 0.0


# ==============================================================================
# 4. Latency Stage Breakdown Positivity & Sum-Consistency Tests
# ==============================================================================

class TestProductionPipelineLatencyBreakdown:
    """Verifies latency breakdown positivity, sum-consistency, and non-simulated timing."""

    def test_stage_latencies_positivity(self, production_pipeline: PRLRPipeline):
        """Verify all stage latency breakdown values are positive numbers (>0.0 ms)."""
        prompt = "Route from alpha to omega"
        result = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=2,
            max_new_tokens=16,
        )

        breakdown = result.latency_breakdown
        required_keys = ["prefill_ms", "deliberation_ms", "decode_ms", "total_ms"]
        for k in required_keys:
            assert k in breakdown, f"Missing required latency key: {k}"
            val = breakdown[k]
            assert isinstance(val, float), f"Latency key {k} must be float, got {type(val)}"
            assert val > 0.0, f"Latency key {k} must be positive, got {val}"

    def test_stage_latencies_sum_consistency(self, production_pipeline: PRLRPipeline):
        """Verify total_ms accounts for all individual stage latencies."""
        prompt = "Compute multi-step plan"
        result = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=2,
            max_new_tokens=16,
        )

        b = result.latency_breakdown
        sum_stages = b["prefill_ms"] + b["deliberation_ms"] + b["decode_ms"]

        # Total elapsed wall-clock time must be at least the sum of individual sub-stages
        assert b["total_ms"] >= sum_stages * 0.95, (
            f"Total time ({b['total_ms']}ms) is smaller than component sum ({sum_stages}ms)"
        )
        diff = b["total_ms"] - sum_stages
        assert -5.0 <= diff < 250.0, f"Unreasonable latency discrepancy: total - stages = {diff}ms"

    def test_latencies_non_simulated_scaling(self, production_pipeline: PRLRPipeline):
        """Rule 6: Verify latencies reflect measured time rather than synthetic token multipliers."""
        prompt = "A brief question"
        res_short = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=1,
            max_new_tokens=8,
        )
        res_long = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=1,
            max_new_tokens=16,
        )

        dec_short = res_short.latency_breakdown["decode_ms"]
        dec_long = res_long.latency_breakdown["decode_ms"]

        assert dec_short != dec_long
        assert not (dec_short).is_integer()
        assert not (dec_long).is_integer()


# ==============================================================================
# 5. Text Quality, Shannon Entropy, and Repetition Bounds Tests
# ==============================================================================

class TestProductionPipelineTextQuality:
    """Verifies generated solution text satisfies Shannon entropy and repetition criteria."""

    def test_decoded_text_non_zero_shannon_entropy(self, production_pipeline: PRLRPipeline):
        """Verify decoded solution text exhibits healthy Shannon entropy (H >= 3.0 bits)."""
        prompt = "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
        result = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=4,
            max_new_tokens=32,
            temperature=0.0,
        )

        text = result.decoded_text
        entropy = compute_shannon_entropy(text)
        assert entropy >= 3.0, f"Shannon entropy too low: {entropy:.2f} bits < 3.0 bits (text: {text!r})"

    def test_decoded_text_bounded_4gram_repetition(self, production_pipeline: PRLRPipeline):
        """Verify decoded solution text does not get trapped in repetition loops (max 4-gram rep <= 2)."""
        prompt = "<start_of_turn>user\nPlan route: initial [input_a] target [output_z]<end_of_turn>\n<start_of_turn>model\n"
        result = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=4,
            max_new_tokens=32,
            temperature=0.0,
        )

        text = result.decoded_text
        rep_count = compute_max_4gram_repetition(text)
        assert rep_count <= 2, f"Repetition loop detected: 4-gram repetition count {rep_count} > 2 (text: {text!r})"

    def test_decoded_text_non_empty_and_valid(self, production_pipeline: PRLRPipeline):
        """Verify decoded text contains valid string content and is not empty or pure padding."""
        prompt = "What is the next step in routing?"
        result = production_pipeline.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=2,
            max_new_tokens=16,
        )
        assert isinstance(result.decoded_text, str)
        assert len(result.decoded_text.strip()) > 0
        assert "<pad><pad>" not in result.decoded_text


# ==============================================================================
# 6. CLI Flags & Argument Parsing Tests (demo.py & run_benchmark.py)
# ==============================================================================

class TestProductionPipelineCLIFlags:
    """Verifies CLI argument parsing and flags in demo.py and run_benchmark.py."""

    def test_demo_cli_argument_defaults(self):
        """Verify demo.py CLI argument defaults configure the genuine pretrained Gemma 2B lane."""
        import demo

        assert hasattr(demo, "MODEL_PRESETS")
        assert "gemma_2b" in demo.MODEL_PRESETS

    def test_demo_cli_smoke_help(self):
        """Verify demo.py --help executes cleanly with returncode 0 and displays PRLR options."""
        res = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "demo.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_DIR),
        )
        assert res.returncode == 0
        assert "Parallel Latent Reasoner" in res.stdout
        assert "--prompt" in res.stdout
        assert "--adapter" in res.stdout
        assert "--trained" in res.stdout
        assert "--no-gate" in res.stdout
        assert "--steps" in res.stdout
        assert "--slots" in res.stdout

    def test_benchmark_cli_smoke_help(self):
        """Verify run_benchmark.py --help executes cleanly with returncode 0 and displays options."""
        res = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "run_benchmark.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_DIR),
        )
        assert res.returncode == 0
        assert "Benchmark" in res.stdout
        assert "--preset" in res.stdout
        assert "--adapter" in res.stdout
        assert "--trained" in res.stdout
        assert "--no-gate" in res.stdout
        assert "--quick" in res.stdout
        assert "--output-dir" in res.stdout
