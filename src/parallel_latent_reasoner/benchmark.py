"""Backward compatibility shim for parallel_latent_reasoner.benchmark.

Canonical implementation has moved to prlr.compact.benchmark.
"""

from __future__ import annotations

from prlr.compact.benchmark import (
    BenchmarkResult,
    DomainSampleRecord,
    MultiDomainBenchmarkSuite,
    MultiScaleBenchmarkSuite,
    _get_peak_memory_bytes,
    _get_peak_memory_mb,
    _reset_peak_memory,
    compute_max_ngram_repetition,
    compute_shannon_entropy,
    evaluate_preset,
    generate_benchmark_report_markdown,
    run_ar_cot_benchmark,
    run_prlr_benchmark,
    run_serial_recurrent_benchmark,
)

__all__ = [
    "compute_shannon_entropy",
    "compute_max_ngram_repetition",
    "BenchmarkResult",
    "DomainSampleRecord",
    "MultiScaleBenchmarkSuite",
    "MultiDomainBenchmarkSuite",
    "evaluate_preset",
    "run_ar_cot_benchmark",
    "run_serial_recurrent_benchmark",
    "run_prlr_benchmark",
    "generate_benchmark_report_markdown",
    "_reset_peak_memory",
    "_get_peak_memory_bytes",
    "_get_peak_memory_mb",
]
