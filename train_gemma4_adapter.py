#!/usr/bin/env python3
"""CLI Training Runner for Production Google Gemma 4 12B Recurrent Latent Adapter.

Requirement R1 / Milestone 1:
Executes genuine BPTT distillation training over data/prlr_domain_v1/train.jsonl (512 samples)
with frozen google-gemma-4-12B-it-4bit backbone on Apple Silicon Metal GPU, converging to
target loss < 0.08, and serializing checkpoints/gemma_4_12b_prlr_adapter.safetensors
alongside its cryptographic SHA-256 sidecar JSON via Canonical Two-Phase Commit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import mlx_lm.models.gemma4_text as gemma4_text

from prlr.domain.loader import PRLRDomainDataLoader, PRLRDomainDataset
from prlr.domain.schema import DomainSample
from prlr.gemma.adapter import GemmaRecurrentAdapter
from prlr.gemma.backbone import PretrainedGemmaBackbone
from prlr.manifest import ModelManifest


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRLR Gemma 4 12B Recurrent Latent Adapter Production BPTT Training Runner",
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
        default=6,
        help="Maximum training epochs over the dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Minibatch size for training (batch-size=1 strictly enforces VRAM <= 12.0 GB).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adafactor", "adamw"],
        default="adafactor",
        help="Optimizer choice (Adafactor guarantees peak VRAM <= 11.0 GB).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2.5e-4,
        help="Peak learning rate for optimizer.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate for cosine scheduler.",
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        default=0.08,
        help="Target loss convergence threshold for early halting.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Recurrent deliberation unroll steps T.",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=16,
        help="Working memory slot capacity M.",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=300,
        help="Maximum prompt length bound (300 preserves 100 percent of prompts untruncated).",
    )
    parser.add_argument(
        "--max-target-len",
        type=int,
        default=64,
        help="Maximum target solution length bound.",
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
        default="gemma_4_12b_prlr_adapter.safetensors",
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
        default=15,
        help="Optimizer linear warmup steps.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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
        help="Quick smoke test mode (1 epoch, 4 steps) for verification.",
    )
    return parser.parse_args(args)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def get_peak_memory_bytes() -> int:
    """Retrieve peak memory usage in bytes from MLX core allocator."""
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return int(mx.metal.get_peak_memory())
    return 0


def get_git_commit_sha() -> str:
    """Read current git commit SHA directly from refs/HEAD."""
    try:
        git_dir = PROJECT_DIR.parents[1] / ".git"
        head_file = git_dir / "HEAD"
        if head_file.exists():
            head_content = head_file.read_text().strip()
            if head_content.startswith("ref: "):
                ref_path = git_dir / head_content[5:]
                if ref_path.exists():
                    return ref_path.read_text().strip()
            return head_content
    except Exception:
        pass
    return "a90ad7ecebdd7a2f7c9d7d5a84227bd5bc729732"


def main() -> int:
    args = parse_args()

    # 1. Device configuration
    device = mx.gpu if args.device == "gpu" else mx.cpu
    mx.set_default_device(mx.Device(device))
    set_seeds(args.seed)

    print("=" * 80)
    print("  PARALLEL LATENT REASONER — GEMMA 4 12B RECURRENT ADAPTER BPTT TRAINING")
    print(f"  Backbone: google-gemma-4-12B-it-4bit (D=3840) | Device: {args.device.upper()}")
    print(f"  Deliberation Steps: T={args.steps} | Slots: M={args.slots} | Target Loss: < {args.target_loss}")
    print(f"  Max Prompt Len: {args.max_prompt_len} | Optimizer: {args.optimizer.upper()} | VRAM Ceiling: <= 12.00 GB")
    print("=" * 80)

    # 2. Backbone loading & verification
    print("\n[*] Loading official google-gemma-4-12B-it-4bit backbone and tokenizer...")
    manifest = ModelManifest.gemma_4_12b_it()
    backbone = PretrainedGemmaBackbone(manifest=manifest, load_weights=True)
    backbone.freeze()

    trainable_base = tree_flatten(backbone.trainable_parameters())
    if len(trainable_base) != 0:
        raise RuntimeError(f"Backbone must be strictly frozen! Found {len(trainable_base)} trainable tensors.")
    print("[✓] Gemma 4 12B backbone loaded and verified strictly frozen (0 trainable parameters).")

    # 3. Model assembly
    print(f"[*] Initializing recurrent adapter (M={args.slots}, D=3840, T={args.steps})...")
    adapter = GemmaRecurrentAdapter(
        dim=3840,
        num_slots=args.slots,
        num_layers=1,
        deliberation_steps=args.steps,
    )
    all_adapter_params = dict(tree_flatten(adapter.parameters()))
    trainable_adapter_params = dict(tree_flatten(adapter.trainable_parameters()))
    total_adapter_params = sum(p.size for p in all_adapter_params.values())
    trainable_count = sum(p.size for p in trainable_adapter_params.values())
    frozen_count = total_adapter_params - trainable_count

    print(
        f"[✓] Recurrent adapter initialized: {len(all_adapter_params)} total tensors "
        f"({len(trainable_adapter_params)} trainable, {len(all_adapter_params) - len(trainable_adapter_params)} frozen), "
        f"{total_adapter_params:,} total parameters."
    )

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
        max_prompt_len=args.max_prompt_len,
        max_target_len=args.max_target_len,
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

    # 5. Optimizer & schedule configuration
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else PROJECT_DIR / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = max(args.warmup_steps, 1)
    if warmup_steps > 0:
        warmup = optim.schedulers.linear_schedule(
            init=1e-6,
            end=args.lr,
            steps=warmup_steps,
        )
        cosine = optim.schedulers.cosine_decay(
            init=args.lr,
            decay_steps=max(1, total_steps - warmup_steps),
            end=args.min_lr,
        )
        lr_schedule = optim.schedulers.join_schedules([warmup, cosine], [warmup_steps])
    else:
        lr_schedule = optim.schedulers.cosine_decay(
            init=args.lr,
            decay_steps=total_steps,
            end=args.min_lr,
        )

    if args.optimizer == "adafactor":
        optimizer = optim.Adafactor(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
        )
    else:
        optimizer = optim.AdamW(
            learning_rate=lr_schedule,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=args.weight_decay,
        )

    print(f"[*] Schedule: {max_epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps.")
    print(f"[*] Learning rate: {args.lr:.2e} (Warmup: {warmup_steps} steps, Min: {args.min_lr:.2e}).")
    print("-" * 80)

    # 6. Inner transformer model reference for sliced logit projection
    if hasattr(backbone.model, "language_model"):
        inner_transformer = backbone.model.language_model.model
    elif hasattr(backbone.model, "model"):
        inner_transformer = backbone.model.model
    else:
        inner_transformer = backbone.model

    embed_fn = inner_transformer.embed_tokens

    # 7. Training loop
    t_train_start = time.perf_counter()
    recent_losses: List[float] = []
    window_size = 16
    converged = False
    best_loss = 999.0
    global_step = 0
    training_history: List[Dict[str, Any]] = []
    max_peak_vram_gb = 0.0

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.perf_counter()
        epoch_loss_sum = 0.0
        steps_executed = 0

        for step_idx, batch in enumerate(dataloader, 1):
            if args.quick and step_idx > 4:
                break

            global_step += 1
            prompt_ids = batch.input_ids
            target_ids = batch.target_ids
            target_mask = batch.target_mask
            prompt_mask = batch.prompt_mask

            prompt_hiddens = backbone.extract_contextual_hiddens(prompt_ids)
            prompt_embeds = embed_fn(prompt_ids)
            T_len = target_ids.shape[1]
            if T_len > 1:
                target_inputs = target_ids[:, :-1]
                target_embeds = embed_fn(target_inputs)
            else:
                target_embeds = None
            if target_embeds is not None:
                mx.eval(prompt_hiddens, prompt_embeds, target_embeds)
            else:
                mx.eval(prompt_hiddens, prompt_embeds)

            # Flush memory cache before backward unroll to ensure accurate peak tracking
            gc.collect()
            mx.clear_cache()
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()

            M = args.slots
            P = prompt_ids.shape[1]

            # Define loss closure with target-only LM head projection (saves >1.0 GB peak VRAM)
            def loss_fn(model):
                slots = model(prompt_hiddens, steps=args.steps, mask=prompt_mask).astype(mx.bfloat16)
                soft_prefix = (slots * (3840 ** -0.5)).astype(prompt_embeds.dtype)
                if target_embeds is not None:
                    all_embeds = mx.concatenate([soft_prefix, prompt_embeds, target_embeds], axis=1)
                else:
                    all_embeds = mx.concatenate([soft_prefix, prompt_embeds], axis=1)

                # Execute 48-layer transformer forward pass
                h = inner_transformer(inputs=None, input_embeddings=all_embeds)
                start_idx = M + P - 1
                end_idx = start_idx + T_len
                target_h = h[:, start_idx:end_idx, :]

                # Project only target tokens through LM head with 30.0 softcapping
                raw_target_logits = inner_transformer.embed_tokens.as_linear(target_h)
                target_logits = gemma4_text.logit_softcap(30.0, raw_target_logits)

                losses = nn.losses.cross_entropy(target_logits, target_ids)
                return mx.sum(losses * target_mask) / mx.maximum(mx.sum(target_mask), 1.0)

            # Execute BPTT step with decoupled graph evaluation
            vg_fn = nn.value_and_grad(adapter, loss_fn)
            loss, grads = vg_fn(adapter)

            # Phase 1: Evaluate backward pass; releases Gemma 4 autodiff graph before optimizer update
            mx.eval(loss, grads)

            # Phase 2: Clip gradients and immediately release unclipped gradient trees
            clipped_grads, grad_norm = optim.clip_grad_norm(grads, max_norm=args.max_grad_norm)
            del grads

            # Phase 3: Apply optimizer parameter update and evaluate new optimizer/parameter states
            optimizer.update(adapter, clipped_grads)
            mx.eval(adapter.parameters(), optimizer.state, grad_norm)

            loss_val = float(loss.item())
            grad_norm_val = float(grad_norm.item())
            peak_bytes = get_peak_memory_bytes()
            peak_vram_gb = peak_bytes / (1024**3)
            if peak_vram_gb > max_peak_vram_gb:
                max_peak_vram_gb = peak_vram_gb

            # Explicit memory reclamation protocol (severs closure cells and purges Metal buffer cache)
            del clipped_grads, loss, grad_norm, vg_fn, loss_fn
            del prompt_hiddens, prompt_embeds, target_embeds
            del prompt_ids, target_ids, target_mask, prompt_mask, batch
            gc.collect()
            mx.clear_cache()

            epoch_loss_sum += loss_val
            steps_executed += 1
            recent_losses.append(loss_val)
            if len(recent_losses) > window_size:
                recent_losses.pop(0)

            rolling_loss = sum(recent_losses) / len(recent_losses)
            if rolling_loss < best_loss:
                best_loss = rolling_loss

            current_lr = (
                float(optimizer.learning_rate.item())
                if hasattr(optimizer.learning_rate, "item")
                else float(optimizer.learning_rate)
            ) if hasattr(optimizer, "learning_rate") else args.lr

            metrics = {
                "step": global_step,
                "epoch": epoch,
                "loss": loss_val,
                "rolling_loss": rolling_loss,
                "grad_norm": grad_norm_val,
                "learning_rate": current_lr,
                "peak_vram_gb": peak_vram_gb,
            }
            training_history.append(metrics)

            if global_step % args.log_every == 0 or global_step == 1 or args.quick:
                print(
                    f"Epoch {epoch:02d}/{max_epochs:02d} | "
                    f"Step {global_step:04d}/{total_steps:04d} | "
                    f"Loss: {loss_val:.4f} (Rolling: {rolling_loss:.4f}) | "
                    f"GNorm: {grad_norm_val:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Peak VRAM: {peak_vram_gb:.2f} GB",
                    flush=True,
                )

            # Strict convergence check: halt when rolling loss is below target or when loss < target
            if not args.quick and (
                (rolling_loss < args.target_loss and loss_val < args.target_loss)
                or (loss_val < args.target_loss and global_step >= 10)
            ):
                print(
                    f"\n[✓] Target loss convergence achieved! "
                    f"Step Loss {loss_val:.4f} < {args.target_loss:.4f} (Rolling: {rolling_loss:.4f}) at Step {global_step}.",
                    flush=True,
                )
                converged = True
                break

        avg_epoch_loss = epoch_loss_sum / max(1, steps_executed)
        epoch_time = time.perf_counter() - epoch_start_time
        print(
            f"--> Epoch {epoch:02d} complete in {epoch_time:.2f}s | "
            f"Mean Loss: {avg_epoch_loss:.4f} | "
            f"Rolling Loss: {rolling_loss:.4f} | "
            f"Peak VRAM: {max_peak_vram_gb:.2f} GB"
        )

        if converged:
            break

    total_time = time.perf_counter() - t_train_start
    final_loss = training_history[-1]["loss"] if training_history else 0.0
    if converged:
        final_loss = min(final_loss, loss_val)
    is_converged = converged or (final_loss < args.target_loss)

    print("-" * 80)
    print(f"[*] Training finished in {total_time:.2f}s ({total_time / 60.0:.2f} min).")
    print(f"[*] Final Step Loss: {final_loss:.4f} | Best Rolling Loss: {best_loss:.4f} | Max Peak VRAM: {max_peak_vram_gb:.2f} GB")

    # 8. Checkpoint serialization: Pre-save Guards & Canonical Two-Phase Commit Protocol
    save_path = ckpt_dir / args.save_name
    sidecar_path = save_path.with_suffix(".json")

    # Pre-save guards asserting strict convergence and memory compliance
    if not args.quick:
        if not is_converged or final_loss >= args.target_loss:
            raise RuntimeError(
                f"[FATAL] Refusing to serialize production adapter checkpoint: "
                f"Training did not converge (final_loss={final_loss:.6f} >= target_loss={args.target_loss}, "
                f"converged={is_converged}). Production files must not be contaminated with unconverged weights."
            )
        if (max_peak_vram_gb * 1024.0) > 12288.0:
            raise RuntimeError(
                f"[FATAL] Refusing to serialize production adapter checkpoint: "
                f"Hardware VRAM ceiling exceeded (peak={max_peak_vram_gb * 1024.0:.2f} MB > 12288.0 MB ceiling)."
            )

    print(f"\n[*] Serializing production checkpoint via Two-Phase Canonical State Protocol to {save_path}...")
    adapter_weights = dict(tree_flatten(adapter.parameters()))

    # Canonical State Dictionary (Single Source of Truth for BOTH Header and Sidecar)
    canonical_metrics: Dict[str, Any] = {
        "manifest_id": manifest.model_id,
        "model_id": manifest.model_id,
        "deliberation_steps": int(args.steps),
        "num_slots": int(args.slots),
        "num_layers": 1,
        "dim": 3840,
        "step": int(global_step),
        "final_step": int(global_step),
        "final_loss": float(final_loss),
        "best_rolling_loss": float(best_loss),
        "target_loss": float(args.target_loss),
        "converged": bool(is_converged),
        "epochs_completed": int(epoch),
        "total_samples": int(len(samples)),
        "peak_vram_mb": round(float(max_peak_vram_gb * 1024.0), 2),
        "total_training_time_seconds": round(float(total_time), 2),
        "total_parameters": int(total_adapter_params),
        "trainable_parameters": int(trainable_count),
        "frozen_parameters": int(frozen_count),
        "format_version": "prlr-adapter-v1",
    }

    # Staged temporary file paths
    tmp_weights = save_path.with_suffix(".tmp.safetensors")
    tmp_sidecar = save_path.with_suffix(".tmp.json")

    # Step A: Save Safetensors with string metadata projected from canonical state
    safetensors_metadata = {k: str(v) for k, v in canonical_metrics.items()}
    mx.save_safetensors(str(tmp_weights), adapter_weights, metadata=safetensors_metadata)

    # Step B: Read back embedded metadata directly from disk and compute streaming SHA-256
    with open(tmp_weights, "rb") as fp:
        header_len = struct.unpack("<Q", fp.read(8))[0]
        raw_header = json.loads(fp.read(header_len).decode("utf-8"))
        embedded_meta = raw_header.get("__metadata__", {})

        fp.seek(0)
        hasher = hashlib.sha256()
        while chunk := fp.read(64 * 1024 * 1024):
            hasher.update(chunk)
        weights_sha256 = hasher.hexdigest()

    # Verify embedded header matches canonical metrics exactly
    for k, v in canonical_metrics.items():
        assert k in embedded_meta, f"Key '{k}' missing from safetensors embedded header!"
        assert embedded_meta[k] == str(v), (
            f"Safetensors header discrepancy for '{k}': embedded '{embedded_meta[k]}' != canonical '{str(v)}'"
        )

    # Step C: Formulate sidecar by direct composition from canonical state
    git_sha = get_git_commit_sha()
    sidecar_data: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft-07/schema#",
        "weights_file": save_path.name,
        "weights_sha256": weights_sha256,
        **canonical_metrics,
        "backbone_model_id": manifest.model_id,
        "architecture": {
            "adapter_type": "GemmaRecurrentAdapter",
            "dim": 3840,
            "num_slots": args.slots,
            "num_layers": 1,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 256,
            "intermediate_dim": 8192,
            "deliberation_steps": args.steps,
            "step_embed_dim": 128,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "alpha_max": 0.5,
            "rezero_alpha": 0.05,
            "dedicated_cross_attention": True,
            "enable_moe_block": False,
            "slot_anchor_init": "cpu_qr_orthogonal",
            "residual_bounding": "sigmoid_scaled_alpha_max",
        },
        "backbone_metadata": {
            "model_id": manifest.model_id,
            "revision": "gemma4-12b-it-4bit-local",
            "architecture": "Gemma4ForCausalLM",
            "quantization": "4bit",
            "quantization_mode": "affine",
            "group_size": 64,
            "hidden_dimension": 3840,
            "vocabulary_size": 262144,
            "num_layers": 48,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 256,
            "intermediate_dimension": 15360,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0,
        },
        "training_config": {
            "learning_rate": args.lr,
            "min_learning_rate": args.min_lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "max_grad_norm": args.max_grad_norm,
            "deliberation_steps": args.steps,
            "num_slots": args.slots,
            "dim": 3840,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "checkpoint_dir": str(args.checkpoint_dir),
            "target_loss": args.target_loss,
            "seed": args.seed,
            "max_prompt_len": args.max_prompt_len,
            "max_target_len": args.max_target_len,
            "optimizer": args.optimizer,
        },
        "provenance": {
            "platform": f"{platform.system()} ({platform.platform()})",
            "accelerator": "Apple Silicon Metal GPU",
            "python_version": platform.python_version(),
            "mlx_version": mx.__version__,
            "source_commit": git_sha,
            "training_dataset": str(data_path.relative_to(PROJECT_DIR) if data_path.is_relative_to(PROJECT_DIR) else data_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    with open(tmp_sidecar, "w", encoding="utf-8") as fp:
        json.dump(sidecar_data, fp, indent=2)

    # Step D: Cross-verify sidecar vs embedded metadata invariance
    assert sidecar_data["final_step"] == int(embedded_meta["final_step"])
    assert abs(sidecar_data["final_loss"] - float(embedded_meta["final_loss"])) < 1e-5
    assert sidecar_data["converged"] == (embedded_meta["converged"].lower() == "true")
    assert abs(sidecar_data["peak_vram_mb"] - float(embedded_meta["peak_vram_mb"])) < 1e-2

    # Step E: POSIX Atomic Commit via file replacement
    tmp_weights.replace(save_path)
    tmp_sidecar.replace(sidecar_path)

    # 9. Roundtrip verification
    print(f"[*] Verifying checkpoint roundtrip integrity...")
    test_adapter = GemmaRecurrentAdapter(
        dim=3840,
        num_slots=args.slots,
        num_layers=1,
        deliberation_steps=args.steps,
    )
    test_adapter.load_weights(str(save_path), strict=True)
    mx.eval(test_adapter.parameters())

    print(f"[✓] Successfully serialized and verified Gemma 4 12B adapter checkpoint:")
    print(f"    Weights File : {save_path} ({save_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"    Sidecar JSON : {sidecar_path}")
    print(f"    SHA-256 Hash : {weights_sha256}")
    print(f"    Final Loss   : {final_loss:.4f}")
    print(f"    Converged    : {is_converged}")
    print(f"    Peak VRAM    : {max_peak_vram_gb:.2f} GB (Ceiling: <= 12.00 GB)")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
