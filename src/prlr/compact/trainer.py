"""Native MLX BPTT Latent Thought Distillation Engine for Compact Testbed.

Implements Backpropagation Through Time (BPTT) across recurrent Jacobi unroll steps
(T=2..8) over M working memory slots on Apple Silicon Metal GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from prlr.compact.scratch_model import MLXCompactGemmaModel

if TYPE_CHECKING:
    from prlr.compact.config import GemmaLatentConfig


@dataclass
class TrainerConfig:
    """Configuration dataclass for PRLR BPTT Latent Distillation."""

    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_steps: int = 20
    total_steps: int = 200
    max_grad_norm: float = 1.0

    deliberation_steps: int = 4
    lambda_align: float = 0.5
    lambda_aux: float = 0.1
    rho_contractive: float = 0.85

    checkpoint_dir: str | Path = "checkpoints"
    save_every_epochs: int = 0
    log_every_steps: int = 5


@dataclass
class TrainMetrics:
    """Container for step-level and epoch-level training metrics."""

    total_loss: float
    ce_loss: float
    align_loss: float
    aux_loss: float
    grad_norm: float
    learning_rate: float
    step: int


def _compute_bptt_loss(
    model: MLXCompactGemmaModel,
    input_ids: mx.array,
    target_tokens: mx.array,
    teacher_latents: mx.array | None = None,
    steps: int = 4,
    lambda_align: float = 0.5,
    lambda_aux: float = 0.1,
    rho_contractive: float = 0.85,
) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array]]:
    """Compute differentiable BPTT distillation loss across T Jacobi unrolls."""
    slots, prompt_hiddens = model.prelude(input_ids)
    prompt_len = prompt_hiddens.shape[1]
    prompt_kv = model.engine.layers[0].attn.create_prompt_kv(prompt_hiddens)

    curr = slots
    trajectory: list[mx.array] = [curr]
    for t in range(1, steps + 1):
        for layer in model.engine.layers:
            curr = layer(
                curr,
                step=t,
                prompt_kv=prompt_kv,
                prompt_len=prompt_len,
            )
        trajectory.append(curr)

    final_states = trajectory[-1]
    readout = model.coda.pool_readout(final_states)

    if target_tokens.ndim == 1:
        logits = model.coda.project_logits(readout)
        l_ce = mx.mean(nn.losses.cross_entropy(logits, target_tokens))
    elif target_tokens.ndim == 2:
        seq_len = target_tokens.shape[1]
        if seq_len == 1:
            logits = model.coda.project_logits(readout)
            l_ce = mx.mean(nn.losses.cross_entropy(logits, target_tokens[:, 0]))
        else:
            seq_losses: list[mx.array] = []
            curr_h = readout
            for s in range(seq_len):
                logits_s = model.coda.project_logits(curr_h)
                ce_s = nn.losses.cross_entropy(logits_s, target_tokens[:, s])
                seq_losses.append(ce_s)
                tok_embed = model.prelude.embed_prompt(target_tokens[:, s : s + 1])[:, 0, :]
                curr_h = model.coda.final_norm(curr_h + 0.1 * tok_embed)
            l_ce = mx.mean(mx.stack(seq_losses, axis=-1))
    else:
        raise ValueError(f"Unexpected target_tokens shape: {target_tokens.shape}")

    if teacher_latents is not None:
        eps = 1e-6
        u_norm = readout / (mx.linalg.norm(readout, axis=-1, keepdims=True) + eps)
        v_norm = teacher_latents / (mx.linalg.norm(teacher_latents, axis=-1, keepdims=True) + eps)
        cos_sim = mx.sum(u_norm * v_norm, axis=-1)
        l_cos = mx.mean(1.0 - cos_sim)
        l_nmse = mx.mean(mx.mean((u_norm - v_norm) ** 2, axis=-1))
        l_align = 0.5 * l_cos + 0.5 * l_nmse
    else:
        l_align = mx.array(0.0)

    target_first = target_tokens if target_tokens.ndim == 1 else target_tokens[:, 0]
    intermediate_ce: list[mx.array] = []
    for t in range(1, steps):
        step_readout = model.coda.pool_readout(trajectory[t])
        step_logits = model.coda.project_logits(step_readout)
        weight = float(t) / float(steps)
        intermediate_ce.append(weight * nn.losses.cross_entropy(step_logits, target_first))

    l_intermediate = (
        mx.mean(mx.stack(intermediate_ce, axis=-1))
        if intermediate_ce
        else mx.array(0.0)
    )

    if len(trajectory) >= 3:
        vel_penalties: list[mx.array] = []
        for t in range(2, len(trajectory)):
            v_curr = mx.mean(mx.abs(trajectory[t] - trajectory[t - 1]))
            v_prev = mx.mean(mx.abs(trajectory[t - 1] - trajectory[t - 2]))
            vel_penalties.append(mx.maximum(mx.array(0.0), v_curr - rho_contractive * v_prev))
        l_vel = mx.mean(mx.stack(vel_penalties))
    else:
        l_vel = mx.array(0.0)

    l_aux = l_intermediate + l_vel
    total_loss = l_ce + lambda_align * l_align + lambda_aux * l_aux
    return total_loss, (l_ce, l_align, l_aux)


class PRLRBPTTTrainer:
    """Native MLX BPTT Latent Distillation Engine for Parallel Latent Reasoner."""

    def __init__(
        self,
        model: MLXCompactGemmaModel,
        config: TrainerConfig | None = None,
        optimizer: optim.Optimizer | None = None,
    ):
        self.model = model
        self.config = config if config is not None else TrainerConfig()

        self.model.freeze_base_model()

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = self._build_optimizer()

        self._loss_and_grad_fn = nn.value_and_grad(self.model, _compute_bptt_loss)

        self.current_step: int = 0
        self.current_epoch: int = 0
        self.training_history: list[dict[str, float]] = []

    def _build_optimizer(self) -> optim.Optimizer:
        cfg = self.config
        if cfg.warmup_steps > 0:
            warmup = optim.schedulers.linear_schedule(
                init=cfg.min_learning_rate,
                end=cfg.learning_rate,
                steps=cfg.warmup_steps,
            )
            cosine = optim.schedulers.cosine_decay(
                init=cfg.learning_rate,
                decay_steps=max(1, cfg.total_steps - cfg.warmup_steps),
                end=cfg.min_learning_rate,
            )
            lr_schedule = optim.schedulers.join_schedules(
                [warmup, cosine], [cfg.warmup_steps]
            )
        else:
            lr_schedule = optim.schedulers.cosine_decay(
                init=cfg.learning_rate,
                decay_steps=cfg.total_steps,
                end=cfg.min_learning_rate,
            )

        return optim.AdamW(
            learning_rate=lr_schedule,
            betas=list(cfg.betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )

    def _parse_batch(
        self,
        batch: dict[str, mx.array] | tuple[mx.array, ...],
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids")
            if input_ids is None:
                input_ids = batch.get("prompt")
            if input_ids is None:
                raise KeyError("Batch dictionary must contain 'input_ids' or 'prompt'.")

            target_tokens = batch.get("target_tokens")
            if target_tokens is None:
                target_tokens = batch.get("target_ids")
            if target_tokens is None:
                target_tokens = batch.get("targets")
            if target_tokens is None:
                raise KeyError("Batch dictionary must contain 'target_tokens', 'target_ids', or 'targets'.")

            teacher_latents = batch.get("teacher_latents")
            return input_ids, target_tokens, teacher_latents
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                return batch[0], batch[1], None
            elif len(batch) >= 3:
                return batch[0], batch[1], batch[2]
            else:
                raise ValueError("Batch tuple must contain at least (input_ids, target_tokens).")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def train_step(
        self,
        batch: dict[str, mx.array] | tuple[mx.array, ...],
        steps: int | None = None,
    ) -> tuple[float, dict[str, float]]:
        input_ids, target_tokens, teacher_latents = self._parse_batch(batch)
        unroll_steps = steps if steps is not None else self.config.deliberation_steps

        (loss, (ce_loss, align_loss, aux_loss)), grads = self._loss_and_grad_fn(
            self.model,
            input_ids,
            target_tokens,
            teacher_latents=teacher_latents,
            steps=unroll_steps,
            lambda_align=self.config.lambda_align,
            lambda_aux=self.config.lambda_aux,
            rho_contractive=self.config.rho_contractive,
        )

        clipped_grads, grad_norm = optim.clip_grad_norm(
            grads, max_norm=self.config.max_grad_norm
        )

        self.optimizer.update(self.model, clipped_grads)
        mx.eval(self.model.parameters(), self.optimizer.state, loss, grad_norm)

        self.current_step += 1

        effective_lr = float(self.optimizer.learning_rate)
        loss_val = float(loss)
        ce_val = float(ce_loss)
        align_val = float(align_loss)
        aux_val = float(aux_loss)
        grad_norm_val = float(grad_norm)

        metrics = {
            "loss": loss_val,
            "ce_loss": ce_val,
            "align_loss": align_val,
            "aux_loss": aux_val,
            "grad_norm": grad_norm_val,
            "learning_rate": effective_lr,
            "step": self.current_step,
        }

        self.training_history.append(metrics)
        return loss_val, metrics

    def train_epoch(
        self,
        dataset: Sequence[Any] | Iterable[Any],
        steps: int | None = None,
    ) -> dict[str, float]:
        epoch_losses: list[float] = []
        epoch_ce: list[float] = []
        epoch_align: list[float] = []
        epoch_aux: list[float] = []
        epoch_grad_norms: list[float] = []

        for batch in dataset:
            loss_val, metrics = self.train_step(batch, steps=steps)
            epoch_losses.append(loss_val)
            epoch_ce.append(metrics["ce_loss"])
            epoch_align.append(metrics["align_loss"])
            epoch_aux.append(metrics["aux_loss"])
            epoch_grad_norms.append(metrics["grad_norm"])

        self.current_epoch += 1

        avg_metrics = {
            "epoch": float(self.current_epoch),
            "loss": sum(epoch_losses) / max(1, len(epoch_losses)),
            "ce_loss": sum(epoch_ce) / max(1, len(epoch_ce)),
            "align_loss": sum(epoch_align) / max(1, len(epoch_align)),
            "aux_loss": sum(epoch_aux) / max(1, len(epoch_aux)),
            "grad_norm": sum(epoch_grad_norms) / max(1, len(epoch_grad_norms)),
            "step": float(self.current_step),
        }

        if (
            self.config.save_every_epochs > 0
            and self.current_epoch % self.config.save_every_epochs == 0
        ):
            self.save_checkpoint()

        return avg_metrics

    def evaluate(
        self,
        val_dataset: Sequence[Any] | Iterable[Any],
        steps: int | None = None,
    ) -> dict[str, float]:
        unroll_steps = steps if steps is not None else self.config.deliberation_steps
        losses: list[float] = []
        ce_losses: list[float] = []
        align_losses: list[float] = []
        aux_losses: list[float] = []
        correct_tokens: int = 0
        total_tokens: int = 0

        for batch in val_dataset:
            input_ids, target_tokens, teacher_latents = self._parse_batch(batch)
            loss, (ce, align, aux) = _compute_bptt_loss(
                self.model,
                input_ids,
                target_tokens,
                teacher_latents=teacher_latents,
                steps=unroll_steps,
                lambda_align=self.config.lambda_align,
                lambda_aux=self.config.lambda_aux,
                rho_contractive=self.config.rho_contractive,
            )
            mx.eval(loss, ce, align, aux)
            losses.append(float(loss))
            ce_losses.append(float(ce))
            align_losses.append(float(align))
            aux_losses.append(float(aux))

            delib = self.model.deliberate(input_ids, steps=unroll_steps)
            logits = self.model.coda(delib.final_states, pool=True)
            preds = mx.argmax(logits, axis=-1)
            target_first = target_tokens if target_tokens.ndim == 1 else target_tokens[:, 0]
            correct_tokens += int(mx.sum(preds == target_first))
            total_tokens += input_ids.shape[0]

        n_batches = max(1, len(losses))
        return {
            "val_loss": sum(losses) / n_batches,
            "val_ce_loss": sum(ce_losses) / n_batches,
            "val_align_loss": sum(align_losses) / n_batches,
            "val_aux_loss": sum(aux_losses) / n_batches,
            "val_accuracy": float(correct_tokens) / max(1, total_tokens),
        }

    def save_checkpoint(
        self,
        filepath: str | Path | None = None,
    ) -> Path:
        if filepath is not None:
            save_path = Path(filepath)
        else:
            ckpt_dir = Path(self.config.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_path = ckpt_dir / "prlr_latent_adapter.npz"

            step_path = ckpt_dir / f"prlr_latent_adapter_step_{self.current_step}.npz"
            self.model.save_adapter_weights(step_path)

        self.model.save_adapter_weights(save_path)
        return save_path

    def load_checkpoint(self, filepath: str | Path) -> dict[str, mx.array]:
        return self.model.load_adapter_weights(filepath)


__all__ = [
    "TrainerConfig",
    "TrainMetrics",
    "PRLRBPTTTrainer",
    "_compute_bptt_loss",
]
