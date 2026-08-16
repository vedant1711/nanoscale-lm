"""The pretraining loop (spec B4).

What this loop actually does, in order, per optimizer step:

1. Accumulate gradients over ``grad_accum`` micro-batches, scaling each micro-batch
   loss by ``1/grad_accum`` so the accumulated gradient equals the gradient of the mean
   loss over the full effective batch (rather than its sum, which would silently scale
   the learning rate by ``grad_accum``).
2. Unscale (under fp16) and clip the global gradient norm.
3. Set the LR from the schedule multiplier and the weight-decay scale from the cautious
   schedule.
4. Step, then zero gradients.

Mixed precision
---------------
Autocast runs the forward pass in bf16/fp16 while parameters and optimizer state stay
fp32 — the "fp32 master weights" path the spec asks for. Under fp16 a
:class:`torch.amp.GradScaler` is required because fp16's exponent range underflows small
gradients to zero; bf16 has fp32's exponent range and needs no scaler. On CPU everything
falls back to fp32 (see :mod:`nanoscale.utils.device`), which is why the ``nano`` tier is
exactly reproducible.

Stopping
--------
Training stops at ``max_steps`` **or** when ``token_budget`` tokens have been consumed,
whichever comes first. The token budget is the compute-honest criterion (spec E1); the
step cap is a guard so a misconfigured run cannot loop forever.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nanoscale.config import ExperimentConfig
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.optim import CompositeOptimizer, build_optimizer, split_parameters
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train.checkpoint import TrainState, load_checkpoint, save_checkpoint
from nanoscale.train.data import Batch, PackedTokens, TokenBatcher, build_packed_tokens
from nanoscale.train.schedule import lr_multiplier, weight_decay_multiplier
from nanoscale.utils import (
    MetricLogger,
    autocast_context,
    get_logger,
    resolve_device,
    resolve_dtype,
    seed_all,
    write_manifest,
)

__all__ = ["TrainResult", "Trainer", "evaluate_loss", "grad_global_norm"]

log = get_logger("nanoscale.train")


@dataclass(slots=True)
class TrainResult:
    """The outcome of a training run."""

    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    steps: int
    tokens: int
    wall_clock_s: float
    tokens_per_second: float
    history: list[dict[str, float]] = field(default_factory=list)
    checkpoint_path: Path | None = None

    def summary(self) -> dict[str, float | int | str]:
        """Flat headline numbers for the manifest and the results table."""
        return {
            "final_train_loss": round(self.final_train_loss, 5),
            "final_val_loss": round(self.final_val_loss, 5),
            "best_val_loss": round(self.best_val_loss, 5),
            "final_val_perplexity": round(math.exp(min(self.final_val_loss, 20.0)), 4),
            "steps": self.steps,
            "tokens": self.tokens,
            "wall_clock_s": round(self.wall_clock_s, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
        }


def grad_global_norm(parameters: Sequence[torch.Tensor]) -> float:
    """L2 norm of the concatenated gradients, for logging."""
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    batches: list[Batch],
    *,
    device: torch.device,
    amp_dtype: torch.dtype = torch.float32,
) -> float:
    """Mean cross-entropy over a fixed list of batches.

    Weighted by token count rather than by batch, so a trailing short batch cannot
    distort the average.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for batch in batches:
            batch = batch.to(device)
            with autocast_context(device, amp_dtype):
                out = model(batch.inputs, targets=batch.targets)
            assert out.loss is not None
            total_loss += float(out.loss) * batch.n_tokens
            total_tokens += batch.n_tokens
    finally:
        model.train(was_training)
    return total_loss / max(1, total_tokens)


class Trainer:
    """Pretraining trainer for a :class:`NanoScaleLM`."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        tokenizer: BPETokenizer | None = None,
        model: NanoScaleLM | None = None,
        data: PackedTokens | None = None,
        out_dir: str | Path | None = None,
        phase: str = "phase4-pretrain",
        run_name: str | None = None,
    ) -> None:
        """Assemble model, data, optimizer and logging for a run."""
        self.config = config
        self.phase = phase
        self.run_name = run_name or f"{config.name}-{phase}"
        train_cfg = config.train

        seed_all(train_cfg.seed, deterministic=train_cfg.deterministic)

        self.device = resolve_device(train_cfg.device)
        self.amp_dtype = resolve_dtype(train_cfg.amp_dtype, self.device)
        self.out_dir = Path(out_dir or train_cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = tokenizer or BPETokenizer(config=config.tokenizer)
        self.model = (model or build_model(config.model)).to(self.device)
        self.data = data or build_packed_tokens(config.data, self.tokenizer)

        self.train_batcher = TokenBatcher(
            self.data.train,
            seq_len=config.data.seq_len,
            batch_size=train_cfg.batch_size,
            seed=train_cfg.seed,
            shuffle=True,
        )
        self.val_batcher = TokenBatcher(
            self.data.val,
            seq_len=config.data.seq_len,
            batch_size=train_cfg.batch_size,
            seed=train_cfg.seed,
            shuffle=False,
        )
        self.val_batches = self.val_batcher.take(train_cfg.eval_batches)

        self.optimizer: CompositeOptimizer = build_optimizer(self.model, train_cfg.optim)
        self.param_split = split_parameters(self.model)
        # fp16 needs loss scaling; bf16 and fp32 do not.
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp_dtype is torch.float16
        )
        self.state = TrainState()
        self.metrics = MetricLogger(self.out_dir, name="metrics")

        self.total_steps = self._planned_steps()
        log.info(
            "%s | %s params | device=%s dtype=%s | %d steps x %d seq x %d batch "
            "(accum %d) = %s tokens",
            self.run_name,
            f"{self.model.num_parameters():,}",
            self.device,
            self.amp_dtype,
            self.total_steps,
            config.data.seq_len,
            train_cfg.batch_size,
            train_cfg.grad_accum,
            f"{self.total_steps * self.tokens_per_step:,}",
        )

    # ------------------------------------------------------------------ planning

    def _consumed_batches(self) -> int:
        """Micro-batches consumed so far, which fixes the position in the data stream."""
        return self.state.step * self.config.train.grad_accum

    @property
    def tokens_per_step(self) -> int:
        """Supervised tokens consumed per optimizer step."""
        cfg = self.config.train
        return cfg.batch_size * cfg.grad_accum * self.config.data.seq_len

    def _planned_steps(self) -> int:
        """Steps implied by ``max_steps`` and ``token_budget``, whichever binds first."""
        cfg = self.config.train
        steps = cfg.max_steps
        if cfg.token_budget is not None:
            steps = min(steps, max(1, cfg.token_budget // self.tokens_per_step))
        return steps

    # -------------------------------------------------------------------- resume

    def maybe_resume(self, path: str | Path | None = None) -> None:
        """Resume from a checkpoint if one is configured or supplied."""
        target = path or self.config.train.resume
        if target is None:
            return
        self.state, _ = load_checkpoint(
            target, model=self.model, optimizer=self.optimizer, restore_rng=True
        )
        # Data position is derived from the step count, never stored: see
        # TokenBatcher.stream. Restoring only the epoch would replay the epoch from
        # its first batch and diverge from an uninterrupted run.
        self.train_batcher.set_epoch(self._consumed_batches() // max(1, len(self.train_batcher)))

    # ------------------------------------------------------------------ one step

    def _micro_step(self, batch: Batch) -> tuple[float, int]:
        """Forward/backward one micro-batch; returns ``(unscaled_loss, n_tokens)``."""
        batch = batch.to(self.device)
        with autocast_context(self.device, self.amp_dtype):
            out = self.model(batch.inputs, targets=batch.targets)
        assert out.loss is not None
        # Divide by grad_accum so the accumulated gradient is the gradient of the *mean*
        # loss over the effective batch, not its sum.
        self.scaler.scale(out.loss / self.config.train.grad_accum).backward()
        return float(out.loss.detach()), batch.n_tokens

    def train(self, *, stop_at_step: int | None = None) -> TrainResult:
        """Run the training loop and return the result.

        Args:
            stop_at_step: Stop after this many optimizer steps *without* changing the
                planned schedule. This models a real interruption — a pre-empted Colab
                session, a wall-clock limit — where the LR schedule was planned for the
                full run and the run simply did not get there. Resuming from such a
                checkpoint then continues on the original schedule, which is what makes
                "resume == uninterrupted" testable at all: shortening ``max_steps``
                instead would compress the cosine into the first segment and change the
                weights the checkpoint was taken from.
        """
        cfg = self.config.train
        stop_at = min(self.total_steps, stop_at_step or self.total_steps)
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        batch_iter = self.train_batcher.stream(self._consumed_batches())
        start = time.perf_counter()
        last_loss = float("nan")
        val_loss = float("nan")
        ckpt_path: Path | None = None

        while self.state.step < stop_at:
            if cfg.token_budget is not None and self.state.tokens >= cfg.token_budget:
                log.info("token budget of %s reached", f"{cfg.token_budget:,}")
                break

            step_loss = 0.0
            step_tokens = 0
            for _ in range(cfg.grad_accum):
                loss, n_tokens = self._micro_step(next(batch_iter))
                step_loss += loss / cfg.grad_accum
                step_tokens += n_tokens

            if self.scaler.is_enabled():
                self.scaler.unscale_(next(iter(self.optimizer)))
                for opt in list(self.optimizer)[1:]:
                    self.scaler.unscale_(opt)

            grad_norm = grad_global_norm(params)
            if cfg.optim.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)

            progress = lr_multiplier(self.state.step, self.total_steps, cfg.schedule)
            lrs = self.optimizer.set_lr(progress)
            self.optimizer.set_weight_decay_scale(
                weight_decay_multiplier(
                    self.state.step, self.total_steps, enabled=cfg.optim.cautious_weight_decay
                )
            )

            if self.scaler.is_enabled():
                for opt in self.optimizer:
                    self.scaler.step(opt)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.state.step += 1
            self.state.tokens += step_tokens
            self.state.epoch = self.train_batcher.epoch
            last_loss = step_loss

            if self.state.step % cfg.log_interval == 0 or self.state.step == 1:
                elapsed = time.perf_counter() - start
                row = self.metrics.log(
                    step=self.state.step,
                    console=True,
                    loss=step_loss,
                    ppl=math.exp(min(step_loss, 20.0)),
                    grad_norm=grad_norm,
                    tokens=self.state.tokens,
                    tokens_per_s=self.state.tokens / max(1e-9, elapsed),
                    **lrs,
                )
                self.state.history.append({k: float(v) for k, v in row.items()})

            if self.state.step % cfg.eval_interval == 0 or self.state.step == self.total_steps:
                val_loss = evaluate_loss(
                    self.model, self.val_batches, device=self.device, amp_dtype=self.amp_dtype
                )
                self.state.best_val_loss = min(self.state.best_val_loss, val_loss)
                self.metrics.log(
                    step=self.state.step,
                    console=True,
                    val_loss=val_loss,
                    val_ppl=math.exp(min(val_loss, 20.0)),
                )

            if self.state.step % cfg.ckpt_interval == 0:
                ckpt_path = self.save(self.out_dir / "checkpoint.pt")

        wall = time.perf_counter() - start
        if math.isnan(val_loss):
            val_loss = evaluate_loss(
                self.model, self.val_batches, device=self.device, amp_dtype=self.amp_dtype
            )
            self.state.best_val_loss = min(self.state.best_val_loss, val_loss)

        ckpt_path = self.save(self.out_dir / "final.pt")
        result = TrainResult(
            final_train_loss=last_loss,
            final_val_loss=val_loss,
            best_val_loss=self.state.best_val_loss,
            steps=self.state.step,
            tokens=self.state.tokens,
            wall_clock_s=wall,
            tokens_per_second=self.state.tokens / max(1e-9, wall),
            history=self.state.history,
            checkpoint_path=ckpt_path,
        )
        self._finish(result)
        return result

    # ------------------------------------------------------------------- artifacts

    def save(self, path: str | Path) -> Path:
        """Write a resumable checkpoint."""
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            state=self.state,
            config=self.config,
            extra={"tier": self.config.name, "phase": self.phase},
        )

    def _finish(self, result: TrainResult) -> None:
        summary: dict[str, Any] = dict(result.summary())
        summary.update(self.data.summary())
        summary.update(self.param_split.summary())
        summary["params"] = self.model.num_parameters()
        summary["chinchilla_fraction"] = round(
            result.tokens / max(1, self.config.model.chinchilla_token_budget()), 4
        )
        self.metrics.summary(**summary)
        self.metrics.close()
        write_manifest(
            self.out_dir,
            run_name=self.run_name,
            phase=self.phase,
            seed=self.config.train.seed,
            config=self.config,
            token_budget=self.config.train.token_budget,
            metrics={k: float(v) for k, v in summary.items() if isinstance(v, int | float)},
        )
        log.info(
            "done: %d steps, %s tokens, val loss %.4f (ppl %.2f), %.1f tok/s",
            result.steps,
            f"{result.tokens:,}",
            result.final_val_loss,
            math.exp(min(result.final_val_loss, 20.0)),
            result.tokens_per_second,
        )
