"""Backward compatibility shim for parallel_latent_reasoner.dataset.

Canonical implementation has moved to prlr.compact.dataset.
"""

from __future__ import annotations

from prlr.compact.dataset import (
    DistillationSample,
    PRLRDataLoader,
    PRLRDataset,
    ProceduralMultiDomainGenerator,
    check_split_contamination,
    generate_distillation_dataset,
    split_dataset,
    train_prlr_adapter,
)

__all__ = [
    "DistillationSample",
    "ProceduralMultiDomainGenerator",
    "split_dataset",
    "check_split_contamination",
    "generate_distillation_dataset",
    "PRLRDataset",
    "PRLRDataLoader",
    "train_prlr_adapter",
]
