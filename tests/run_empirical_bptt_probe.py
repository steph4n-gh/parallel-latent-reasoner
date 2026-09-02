"""Empirical BPTT Stress and Memory Benchmark Harness for PRLR M1 Distillation."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.models import MLXCompactGemmaModel
from parallel_latent_reasoner.trainer import PRLRBPTTTrainer, TrainerConfig


def run_depth_and_batch_sweep():
    print("=== PRLR BPTT Empirical Depth & Batch Sweep ===")
    config = GemmaLatentConfig(
        dim=256,
        intermediate_dim=512,
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        num_layers=4,
        num_memory_slots=16,
        vocab_size=1000,
        deliberation_steps=4,
    )
    model = MLXCompactGemmaModel(config)
    trainer_cfg = TrainerConfig(
        learning_rate=1e-3,
        warmup_steps=5,
        total_steps=50,
        max_grad_norm=1.0,
    )
    trainer = PRLRBPTTTrainer(model, config=trainer_cfg)

    results = []

    for unroll_t in [2, 4, 8, 16, 32]:
        for batch_size in [1, 4, 16]:
            mx.metal.clear_cache() if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache") else None
            mx.reset_peak_memory()
            
            input_ids = mx.random.randint(0, config.vocab_size, (batch_size, 32))
            target_tokens = mx.random.randint(0, config.vocab_size, (batch_size,))
            teacher_latents = mx.random.normal((batch_size, config.dim))
            batch = {
                "input_ids": input_ids,
                "target_tokens": target_tokens,
                "teacher_latents": teacher_latents,
            }

            # Warmup
            trainer.train_step(batch, steps=unroll_t)
            
            # Timed iterations
            t0 = time.perf_counter()
            n_iters = 5
            for _ in range(n_iters):
                loss_val, metrics = trainer.train_step(batch, steps=unroll_t)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / n_iters
            peak_mb = mx.get_peak_memory() / (1024 * 1024)

            entry = {
                "unroll_t": unroll_t,
                "batch_size": batch_size,
                "loss": loss_val,
                "grad_norm": metrics["grad_norm"],
                "step_latency_ms": elapsed_ms,
                "peak_vram_mb": peak_mb,
            }
            results.append(entry)
            print(f"T={unroll_t:2d} | B={batch_size:2d} | Loss={loss_val:6.4f} | GradNorm={metrics['grad_norm']:6.4f} | Latency={elapsed_ms:6.2f}ms | PeakVRAM={peak_mb:6.2f}MB")

    return results


if __name__ == "__main__":
    run_depth_and_batch_sweep()
