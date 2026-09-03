#!/usr/bin/env python3
"""CLI Training Runner for Production Gemma 2B Recurrent Latent Adapter.

Requirement R1 / Milestone 1:
Executes BPTT distillation training over data/prlr_domain_v1/train.jsonl (512 samples)
with frozen google/gemma-2b-it backbone on Apple Silicon Metal GPU, converging to
target loss < 0.15, and serializing checkpoints/gemma_2b_prlr_adapter.safetensors
alongside its cryptographic SHA-256 sidecar JSON.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import List, Optional, Sequence

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mlx.core as mx

from prlr.domain.loader import PRLRDomainDataLoader, PRLRDomainDataset
from prlr.domain.schema import DomainSample
from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.gemma.decoder import GemmaCausalPrefixDecoder
from prlr.gemma.trainer import GemmaPRLRTrainer, GemmaTrainerConfig
from prlr.manifest import ModelManifest


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRLR Gemma 2B Recurrent Latent Adapter Production BPTT Training Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_DIR / "data" / "prlr_domain_v1" / "train.jsonl",
        help="Path to training dataset JSONL (512 samples).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_DIR / "checkpoints",
        help="Directory to save production adapter checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Maximum training epochs over the dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Minibatch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        default=0.15,
        help="Target loss convergence threshold for early halting.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Recurrent deliberation unroll steps T.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="MLX compute device (Metal GPU or CPU).",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="gemma_2b_prlr_adapter.safetensors",
        help="Output weight filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Optimizer linear warmup steps.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm clipping.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress every N steps.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test mode (1 epoch, 4 steps) for CI verification.",
    )
    return parser.parse_args(args)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def main() -> int:
    args = parse_args()

    # 1. Device configuration
    device = mx.gpu if args.device == "gpu" else mx.cpu
    mx.set_default_device(mx.Device(device))
    set_seeds(args.seed)

    print("=" * 80)
    print("  PARALLEL LATENT REASONER — GEMMA 2B RECURRENT ADAPTER BPTT TRAINING")
    print(f"  Backbone: google/gemma-2b-it | Device: {args.device.upper()} | Target Loss: < {args.target_loss}")
    print("=" * 80)

    # 2. Backbone loading
    print("\n[*] Loading official google/gemma-2b-it backbone and tokenizer...")
    manifest = ModelManifest.gemma_2b_it()
    backbone = PretrainedGemmaBackbone(manifest=manifest, load_weights=True)
    backbone.freeze()

    # 3. Model assembly
    print(f"[*] Initializing recurrent adapter (M=16, D=2048, T={args.steps}) and causal prefix decoder...")
    adapter = GemmaRecurrentAdapter(
        dim=2048,
        num_slots=16,
        num_layers=1,
        deliberation_steps=args.steps,
    )
    decoder = GemmaCausalPrefixDecoder(backbone=backbone)

    # 4. Dataset loading
    data_path = args.data_path if args.data_path.is_absolute() else PROJECT_DIR / args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    print(f"[*] Loading training dataset from {data_path}...")
    samples: List[DomainSample] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(DomainSample.from_dict(json.loads(line)))
    print(f"[✓] Loaded {len(samples)} training samples.")

    dataset = PRLRDomainDataset(
        samples=samples,
        tokenizer=backbone.tokenizer,
        pad_token_id=0,
        eos_token_ids=(1, 107),
        max_prompt_len=1024,
        max_target_len=128,
        pretokenize=True,
    )
    dataloader = PRLRDomainDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        mode="train",
        drop_last=False,
    )
    steps_per_epoch = len(dataloader)
    max_epochs = 1 if args.quick else args.epochs
    total_steps = max_epochs * steps_per_epoch

    # 5. Trainer configuration & instantiation
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else PROJECT_DIR / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = GemmaTrainerConfig(
        learning_rate=args.lr,
        min_learning_rate=1e-5,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        max_grad_norm=args.max_grad_norm,
        deliberation_steps=args.steps,
        checkpoint_dir=ckpt_dir,
        seed=args.seed,
    )
    trainer = GemmaPRLRTrainer(
        backbone=backbone,
        adapter=adapter,
        decoder=decoder,
        config=config,
    )

    print(f"[*] Training schedule: {max_epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps.")
    print(f"[*] Initial learning rate: {args.lr:.2e} (Warmup: {args.warmup_steps} steps).")
    print("-" * 80)

    # 6. Training loop
    t_train_start = time.perf_counter()
    recent_losses: List[float] = []
    window_size = 16
    converged = False
    best_loss = 999.0

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.perf_counter()
        epoch_loss_sum = 0.0
        steps_executed = 0

        for step_idx, batch in enumerate(dataloader, 1):
            if args.quick and step_idx > 4:
                break

            loss_val, metrics = trainer.train_step(batch)
            epoch_loss_sum += loss_val
            steps_executed += 1
            recent_losses.append(loss_val)
            if len(recent_losses) > window_size:
                recent_losses.pop(0)

            rolling_loss = sum(recent_losses) / len(recent_losses)
            if rolling_loss < best_loss:
                best_loss = rolling_loss

            if trainer.current_step % args.log_every == 0 or trainer.current_step == 1 or args.quick:
                print(
                    f"Epoch {epoch:02d}/{max_epochs:02d} | "
                    f"Step {metrics['step']:04d}/{total_steps:04d} | "
                    f"Loss: {metrics['loss']:.4f} (Rolling: {rolling_loss:.4f}) | "
                    f"GradNorm: {metrics['grad_norm']:.4f} | "
                    f"LR: {metrics['learning_rate']:.2e} | "
                    f"VRAM: {metrics['peak_memory_mb']:.1f} MB"
                )

            # Convergence check (require >= 2 epochs for stability)
            if not args.quick and epoch >= 2 and rolling_loss < args.target_loss:
                print(
                    f"\n[✓] Target loss convergence achieved! "
                    f"Rolling loss {rolling_loss:.4f} < {args.target_loss:.4f} at Step {metrics['step']}."
                )
                converged = True
                break

        avg_epoch_loss = epoch_loss_sum / max(1, steps_executed)
        epoch_time = time.perf_counter() - epoch_start_time
        print(
            f"--> Epoch {epoch:02d} complete in {epoch_time:.2f}s | "
            f"Mean Loss: {avg_epoch_loss:.4f} | "
            f"Rolling Loss: {rolling_loss:.4f} | "
            f"Peak VRAM: {mx.get_peak_memory() / (1024**2):.1f} MB"
        )

        # Reset allocator caches between epochs
        mx.eval(adapter.parameters(), trainer.optimizer.state)
        mx.clear_cache()
        mx.reset_peak_memory()
        gc.collect()

        if converged:
            break

    total_time = time.perf_counter() - t_train_start
    final_loss = trainer.training_history[-1]["loss"] if trainer.training_history else 0.0
    print("-" * 80)
    print(f"[*] Training finished in {total_time:.2f}s ({total_time / 60.0:.2f} min).")
    print(f"[*] Final Step Loss: {final_loss:.4f} | Best Rolling Loss: {best_loss:.4f}")

    # 7. Checkpoint serialization
    save_path = ckpt_dir / args.save_name
    print(f"\n[*] Serializing production checkpoint to {save_path}...")
    trainer.save_checkpoint(
        filepath=save_path,
        extra_metadata={
            "final_loss": final_loss,
            "best_rolling_loss": best_loss,
            "target_loss": args.target_loss,
            "converged": bool(converged or final_loss < args.target_loss),
            "epochs_completed": epoch,
            "total_samples": len(samples),
            "total_training_time_seconds": round(total_time, 2),
        },
    )

    # 8. Checkpoint verification
    sidecar_path = save_path.with_suffix(".json")
    if not save_path.exists() or not sidecar_path.exists():
        raise RuntimeError("Checkpoint serialization failed: files not found on disk.")

    with open(save_path, "rb") as fp:
        actual_sha = hashlib.sha256(fp.read()).hexdigest()
    with open(sidecar_path, "r", encoding="utf-8") as fp:
        sidecar_data = json.load(fp)

    if sidecar_data.get("weights_sha256") != actual_sha:
        raise ValueError(f"Sidecar SHA-256 mismatch! {sidecar_data.get('weights_sha256')} != {actual_sha}")

    # Test load weights back into fresh adapter with strict=True
    test_adapter = GemmaRecurrentAdapter(
        dim=2048,
        num_slots=16,
        num_layers=1,
        deliberation_steps=args.steps,
    )
    test_adapter.load_weights(str(save_path), strict=True)

    print(f"[✓] Successfully serialized and verified adapter checkpoint:")
    print(f"    Weights File : {save_path} ({save_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"    Sidecar JSON : {sidecar_path}")
    print(f"    SHA-256 Hash : {actual_sha}")
    print(f"    Final Loss   : {final_loss:.4f}")
    print(f"    Converged    : {converged or final_loss < args.target_loss}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
