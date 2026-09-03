"""Authoritative Top-Level Production Pipeline for Parallel Latent Reasoner (PRLR).

Milestone 3 Requirement R3:
- Integrates PretrainedGemmaBackbone (frozen official google/gemma-2b-it weights).
- Integrates GemmaRecurrentAdapter with loaded checkpoints/gemma_2b_prlr_adapter.safetensors.
- Integrates GemmaCausalPrefixDecoder with native MLX KVCache and EOS halting.
- Integrates GemmaCalibratedEGate with calibrated thresholds from checkpoints/calibrated_egate_config.json.
- Synchronizes all MLX arrays with mx.eval() at all stage boundaries (prefill, prelude, deliberation, decode).
- Returns structured PipelineResult dataclass with stage-level latencies and Shannon entropy.
- Provides generate_baseline() for genuine autoregressive Gemma baseline comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.gemma.egate import CalibratedGateThresholds, EGateStepTelemetry, GemmaCalibratedEGate
from prlr.manifest import ModelManifest


def compute_shannon_entropy(text: str) -> float:
    """Compute Shannon entropy H in bits of the character distribution of text.

    H = - sum_i p(x_i) * log2(p(x_i))
    Healthy generated solutions have H >= 3.0 bits.
    Degenerate repetitive/empty strings have H near 0.
    """
    if not text or not text.strip():
        return 0.0

    clean = text.strip()
    length = len(clean)
    counts: Dict[str, int] = {}
    for ch in clean:
        counts[ch] = counts.get(ch, 0) + 1

    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)

    return float(entropy)


def _resolve_project_path(rel_or_abs_path: Union[str, Path]) -> Path:
    """Resolve a path relative to CWD, package root, or canonical project root."""
    p = Path(rel_or_abs_path)
    if p.is_absolute() and p.exists():
        return p

    p_cwd = Path.cwd() / rel_or_abs_path
    if p_cwd.exists():
        return p_cwd

    pkg_root = Path(__file__).resolve().parents[2]
    p_pkg = pkg_root / rel_or_abs_path
    if p_pkg.exists():
        return p_pkg

    fallback = Path("/Volumes/Storage/qan_transformers/projects/parallel_latent_reasoner") / rel_or_abs_path
    if fallback.exists():
        return fallback

    return p_pkg if not p.is_absolute() else p


class BaselineResult(tuple):
    """Container for baseline generation result, unpackable as (text, elapsed_ms)."""

    def __new__(
        cls,
        text: str,
        latency_ms: float,
        tokens: Optional[List[int]] = None,
    ) -> BaselineResult:
        obj = super().__new__(cls, (text, latency_ms))
        return obj

    def __init__(
        self,
        text: str,
        latency_ms: float,
        tokens: Optional[List[int]] = None,
    ):
        self._tokens = tokens or []

    @property
    def generated_text(self) -> str:
        return self[0]

    @property
    def text(self) -> str:
        return self[0]

    @property
    def latency_ms(self) -> float:
        return self[1]

    @property
    def elapsed_ms(self) -> float:
        return self[1]

    @property
    def tokens(self) -> List[int]:
        return self._tokens


@dataclass
class PipelineResult:
    """Structured result container for PRLRPipeline generation."""

    generated_text: str
    tokens: List[int]
    stage_latencies_ms: Dict[str, float]
    halt_step: int
    exit_reason: str
    shannon_entropy: float
    final_states: Optional[mx.array] = None
    consensus_step: Optional[int] = None
    gate_telemetry: Optional[List[EGateStepTelemetry]] = None
    adapter_loaded: bool = False
    adapter_path: Optional[str] = None
    mode: str = "hybrid_deliberate_then_verify"
    prompt: Optional[Union[str, List[int], mx.array]] = None
    memory_stats: Dict[str, float] = field(default_factory=dict)

    @property
    def decoded_text(self) -> str:
        """Contract property alias for generated_text."""
        return self.generated_text

    @property
    def token_ids(self) -> mx.array:
        """Contract property alias for tokens as MLX tensor."""
        return mx.array(self.tokens, dtype=mx.int32)

    @property
    def deliberation_steps(self) -> int:
        """Contract property alias for halt_step."""
        return self.halt_step

    @property
    def egate_verdict(self) -> str:
        """Contract property alias for exit_reason."""
        return self.exit_reason

    @property
    def latency_breakdown(self) -> Dict[str, float]:
        """Contract property alias for stage_latencies_ms."""
        return self.stage_latencies_ms


# Contract alias for backward compatibility across modules
HybridDeliberationResult = PipelineResult


class PRLRPipeline:
    """Authoritative top-level production pipeline for PRLR with genuine Gemma 2B.

    Initializes:
    1. PretrainedGemmaBackbone(manifest=ModelManifest.gemma_2b_it())
    2. Freezes backbone parameters (backbone.freeze())
    3. GemmaRecurrentAdapter(dim=2048, num_slots=16, num_layers=1)
    4. Loads checkpoints/gemma_2b_prlr_adapter.safetensors via adapter.load_weights(..., strict=True)
    5. GemmaCausalPrefixDecoder(backbone=backbone, adapter=adapter)
    6. GemmaCalibratedEGate loaded with checkpoints/calibrated_egate_config.json
    """

    def __init__(
        self,
        adapter_path: Optional[Union[str, Path]] = None,
        egate_config_path: Optional[Union[str, Path]] = None,
        manifest: Optional[ModelManifest] = None,
        backbone: Optional[PretrainedGemmaBackbone] = None,
        adapter: Optional[GemmaRecurrentAdapter] = None,
        decoder: Optional[GemmaCausalPrefixDecoder] = None,
        egate: Optional[GemmaCalibratedEGate] = None,
        dim: int = 2048,
        num_slots: int = 16,
        num_layers: int = 1,
        deliberation_steps: int = 4,
        load_weights: bool = True,
        load_trained_adapter: bool = True,
        strict_loading: bool = True,
    ):
        # 1. Initialize & Freeze Pretrained Backbone
        if backbone is not None:
            self.backbone = backbone
            self.manifest = getattr(backbone, "manifest", None)
        else:
            self.manifest = manifest if manifest is not None else ModelManifest.gemma_2b_it()
            self.backbone = PretrainedGemmaBackbone(
                manifest=self.manifest,
                load_weights=load_weights,
            )

        # Ensure backbone parameters are completely frozen
        self.backbone.freeze()

        # 2. Initialize Recurrent Adapter
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = GemmaRecurrentAdapter(
                dim=dim,
                num_slots=num_slots,
                num_layers=num_layers,
                deliberation_steps=deliberation_steps,
            )

        self.adapter_loaded: bool = False
        self.adapter_path: Optional[str] = None

        if load_weights and load_trained_adapter:
            default_adapter_name = "checkpoints/gemma_2b_prlr_adapter.safetensors"
            target_path = _resolve_project_path(adapter_path or default_adapter_name)
            if not target_path.exists() and adapter_path is None:
                # Attempt to download production checkpoint from GitHub Release
                try:
                    import importlib.util
                    download_script = target_path.parents[1] / "scripts" / "download_checkpoint.py"
                    if download_script.exists():
                        spec = importlib.util.spec_from_file_location("download_checkpoint", str(download_script))
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            target_path = mod.ensure_checkpoint(target_dir=target_path.parent)
                except Exception:
                    pass

            if target_path.exists():
                self.adapter.load_weights(str(target_path), strict=strict_loading)
                self.adapter_loaded = True
                self.adapter_path = str(target_path)
            else:
                if adapter_path is not None:
                    raise FileNotFoundError(f"Requested adapter checkpoint not found: {adapter_path}")
                else:
                    raise FileNotFoundError(
                        f"Production adapter checkpoint not found at {target_path}. "
                        "Run `python scripts/download_checkpoint.py` or download from "
                        "https://github.com/steph4n-gh/parallel-latent-reasoner/releases/tag/v0.2.0-alpha"
                    )
        else:
            self.adapter_loaded = False
            self.adapter_path = None

        # 3. Initialize Causal Prefix Decoder
        if decoder is not None:
            self.decoder = decoder
        else:
            try:
                self.decoder = GemmaCausalPrefixDecoder(
                    backbone=self.backbone,
                    adapter=self.adapter,
                )
            except TypeError:
                self.decoder = GemmaCausalPrefixDecoder(backbone=self.backbone)

        # 4. Initialize Calibrated E-Gate
        if egate is not None:
            self.egate = egate
            self.egate_config_path = None
        else:
            default_egate_name = "checkpoints/calibrated_egate_config.json"
            target_egate_path = _resolve_project_path(egate_config_path or default_egate_name)
            if target_egate_path.exists():
                with open(target_egate_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                params = cfg_data.get("parameters", cfg_data)
                thresholds = CalibratedGateThresholds.from_dict(params)
                self.egate_config_path = str(target_egate_path)
            else:
                thresholds = CalibratedGateThresholds()
                self.egate_config_path = None

            self.egate = GemmaCalibratedEGate(
                thresholds=thresholds,
                decoder=self.decoder,
            )

        self.dim = dim
        self.num_slots = num_slots
        self.deliberation_steps = deliberation_steps

    @classmethod
    def from_preset(
        cls,
        preset: str = "gemma_2b",
        **kwargs: Any,
    ) -> PRLRPipeline:
        """Factory preset constructor for backward-compatibility with demo/app."""
        return cls(**kwargs)

    def generate(
        self,
        prompt: Union[str, List[int], mx.array],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        use_egate: bool = True,
        steps: int = 4,
    ) -> PipelineResult:
        """Generate solution conditioned on deliberated working memory slots.

        Synchronizes all MLX arrays with mx.eval() at each stage boundary.

        Args:
            prompt: Text prompt string, list of token IDs, or input token array.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 for greedy argmax).
            use_egate: Whether to use dynamic calibrated deliberation E-gate.
            steps: Fixed deliberation steps when use_egate is False.

        Returns:
            PipelineResult containing generated_text, tokens, stage_latencies_ms,
            halt_step, exit_reason, and shannon_entropy.
        """
        # ----------------------------------------------------------------------
        # Stage 1: Prefill & Contextual Hidden Extraction
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        if isinstance(prompt, list):
            prompt_arr = mx.array(prompt, dtype=mx.int32)
            if prompt_arr.ndim == 1:
                prompt_arr = prompt_arr[None, :]
            prompt_ids = prompt_arr
        elif isinstance(prompt, mx.array):
            prompt_ids, _ = self.backbone.encode_prompt_context(prompt)
        else:
            prompt_ids, _ = self.backbone.encode_prompt_context(prompt)

        prompt_hiddens = self.backbone.extract_contextual_hiddens(prompt_ids)
        mx.eval(prompt_ids, prompt_hiddens)
        t_prefill = (time.perf_counter() - t0) * 1000.0

        # ----------------------------------------------------------------------
        # Stage 2: Prelude Working Memory Projection
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        s0 = self.adapter.prelude(prompt_hiddens)
        mx.eval(s0)
        t_prelude = (time.perf_counter() - t0) * 1000.0

        # ----------------------------------------------------------------------
        # Stage 3: Deliberation (Dynamic E-Gate vs Fixed Depth)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        gate_telemetry: Optional[List[EGateStepTelemetry]] = None
        consensus_step: Optional[int] = None
        if use_egate and self.egate is not None:
            final_slots, halt_step, exit_reason, gate_telemetry = self.egate.execute_dynamic_deliberation(
                prompt_hiddens=prompt_hiddens,
                prompt_ids=prompt_ids,
                adapter=self.adapter,
            )
            mx.eval(final_slots)
            if exit_reason == "4_signal_consensus":
                consensus_step = halt_step
        else:
            final_slots = self.adapter(prompt_hiddens, steps=steps)
            mx.eval(final_slots)
            halt_step = steps
            exit_reason = "fixed_depth"
        t_delib = (time.perf_counter() - t0) * 1000.0

        # ----------------------------------------------------------------------
        # Stage 4: Causal Autoregressive Token Decoding
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        gen_tokens = self.decoder.generate(
            prompt_ids=prompt_ids,
            prefix_latents=final_slots,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        mx.eval(gen_tokens)
        t_decode = (time.perf_counter() - t0) * 1000.0

        # ----------------------------------------------------------------------
        # Output Parsing & Telemetry Synthesis
        # ----------------------------------------------------------------------
        if gen_tokens.ndim > 1:
            tokens_list = [int(tok) for tok in gen_tokens[0].tolist()]
        else:
            tokens_list = [int(tok) for tok in gen_tokens.tolist()]

        if self.backbone.tokenizer is not None:
            generated_text = self.backbone.tokenizer.decode(tokens_list)
            if isinstance(generated_text, list):
                generated_text = " ".join(generated_text)
        else:
            raise ValueError(
                "Cannot decode tokens without loaded official tokenizer. "
                "Character-modulo decoding is strictly prohibited under Rule 5."
            )

        entropy = compute_shannon_entropy(generated_text)
        t_combined_prefill = t_prefill + t_prelude
        t_total = t_combined_prefill + t_delib + t_decode

        stage_latencies = {
            "prefill": round(t_prefill, 3),
            "prelude": round(t_prelude, 3),
            "deliberation": round(t_delib, 3),
            "decode": round(t_decode, 3),
            "total": round(t_total, 3),
            "prefill_ms": round(t_combined_prefill, 3),
            "prelude_ms": round(t_prelude, 3),
            "deliberation_ms": round(t_delib, 3),
            "decode_ms": round(t_decode, 3),
            "total_ms": round(t_total, 3),
        }

        return PipelineResult(
            generated_text=generated_text,
            tokens=tokens_list,
            stage_latencies_ms=stage_latencies,
            halt_step=halt_step,
            exit_reason=exit_reason,
            shannon_entropy=round(entropy, 4),
            final_states=final_slots,
            consensus_step=consensus_step,
            gate_telemetry=gate_telemetry,
            adapter_loaded=self.adapter_loaded,
            adapter_path=self.adapter_path,
            mode="hybrid_deliberate_then_verify",
            prompt=prompt,
        )

    def deliberate_and_verify(
        self,
        prompt: Union[str, List[int], mx.array],
        deliberation_steps: Optional[int] = None,
        max_steps: int = 12,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        enable_dynamic_gate: bool = True,
        **kwargs: Any,
    ) -> PipelineResult:
        """Execute hybrid deliberate-then-verify pipeline.

        Supports both dynamic E-gate consensus early exit and fixed-depth execution.
        """
        use_egate = enable_dynamic_gate
        steps = deliberation_steps if deliberation_steps is not None else max_steps
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_egate=use_egate,
            steps=steps,
        )

    def deliberate_then_verify(
        self,
        prompt: Union[str, List[int], mx.array],
        deliberation_steps: Optional[int] = None,
        max_steps: int = 12,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        enable_dynamic_gate: bool = True,
        **kwargs: Any,
    ) -> PipelineResult:
        """Alias for deliberate_and_verify."""
        return self.deliberate_and_verify(
            prompt=prompt,
            deliberation_steps=deliberation_steps,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_dynamic_gate=enable_dynamic_gate,
            **kwargs,
        )

    def generate_baseline(
        self,
        prompt: Union[str, List[int], mx.array],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> BaselineResult:
        """Genuine autoregressive baseline generation directly with Gemma backbone.

        Zero deliberation, zero working memory slots. Pure causal prefill and decode.
        """
        t0 = time.perf_counter()
        if isinstance(prompt, list):
            prompt_arr = mx.array(prompt, dtype=mx.int32)
            if prompt_arr.ndim == 1:
                prompt_arr = prompt_arr[None, :]
            prompt_ids = prompt_arr
        elif isinstance(prompt, mx.array):
            prompt_ids, _ = self.backbone.encode_prompt_context(prompt)
        else:
            prompt_ids, _ = self.backbone.encode_prompt_context(prompt)

        mx.eval(prompt_ids)

        gen_tokens = self.decoder.generate(
            prompt_ids=prompt_ids,
            prefix_latents=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        mx.eval(gen_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if gen_tokens.ndim > 1:
            tokens_list = [int(tok) for tok in gen_tokens[0].tolist()]
        else:
            tokens_list = [int(tok) for tok in gen_tokens.tolist()]

        if self.backbone.tokenizer is not None:
            generated_text = self.backbone.tokenizer.decode(tokens_list)
            if isinstance(generated_text, list):
                generated_text = " ".join(generated_text)
        else:
            raise ValueError(
                "Cannot decode tokens without loaded official tokenizer. "
                "Character-modulo decoding is strictly prohibited under Rule 5."
            )

        return BaselineResult(generated_text, round(elapsed_ms, 3), tokens=tokens_list)


__all__ = [
    "BaselineResult",
    "HybridDeliberationResult",
    "PipelineResult",
    "PRLRPipeline",
    "compute_shannon_entropy",
]
