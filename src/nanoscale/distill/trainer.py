"""Distillation trainer for all three objectives (spec B6).

The comparison this module exists to support is a *controlled* one: the same student
architecture, the same teacher, the same step budget and the same seed, differing only
in the objective. That is what makes "reverse-KL on-policy beats forward-KL" a claim
about the objective rather than about a training-budget difference.

Warm-start
----------
``DistillConfig.warmup_steps`` runs plain MLE on the training data before the configured
objective takes over. This is not a convenience: MiniLLM prescribes exactly this, and the
reason is structural. On-policy reverse KL estimates its gradient from trajectories the
**student** generates, so a randomly-initialised student is sampling noise and scoring it
against a teacher that finds all of it equally unlikely; the reward carries no usable
signal and the policy gradient is pure variance. Skipping the warm-start does not make
reverse KL slightly worse; it makes it not work at all (measured: student perplexity
~1000 without, versus the teacher's ~3).

The warm-start is applied identically to **every** objective, so the comparison stays
controlled: all three arms start from the same warm-started weights.

Cost note, which is part of the result: the three objectives are not equally expensive
per step. Forward KL runs one teacher forward pass per batch; SeqKD runs none (the
teacher's cost is a one-off sampling phase); reverse-KL on-policy runs a student
*generation* plus a teacher forward pass per batch, and generation is sequential. The
benchmark harness reports wall-clock alongside quality so that trade is visible.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from nanoscale.config import DistillConfig, ExperimentConfig, ScheduleConfig
from nanoscale.distill.losses import (
    DistillLossOutput,
    forward_kl_loss,
    reverse_kl_policy_gradient,
    sequence_kd_loss,
)
from nanoscale.model import NanoScaleLM
from nanoscale.optim import AdamW
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train.checkpoint import TrainState, save_checkpoint
from nanoscale.train.data import Batch, TokenBatcher
from nanoscale.train.schedule import lr_multiplier
from nanoscale.utils import (
    MetricLogger,
    backward,
    get_logger,
    resolve_device,
    seed_all,
    write_manifest,
)

__all__ = ["DistillResult", "DistillTrainer"]

log = get_logger("nanoscale.distill")


@dataclass(slots=True)
class DistillResult:
    """Outcome of a distillation run."""

    method: str
    final_loss: float
    student_val_loss: float
    teacher_val_loss: float
    student_params: int
    teacher_params: int
    student_non_embedding: int
    teacher_non_embedding: int
    steps: int
    warmup_steps: int
    wall_clock_s: float
    checkpoint_path: Path | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Teacher parameters per student parameter, counting everything."""
        return self.teacher_params / max(1, self.student_params)

    @property
    def non_embedding_compression(self) -> float:
        """Compression of the *non-embedding* parameters.

        This is the number that describes what distillation actually shrank. Teacher and
        student must share a tokenizer, so the embedding table and LM head are the same
        width in both and their cost is irreducible, at small vocabularies they can
        dominate the total and make the headline ratio look far worse than the depth and
        width reduction really is.
        """
        return self.teacher_non_embedding / max(1, self.student_non_embedding)

    def summary(self) -> dict[str, float | int | str]:
        """Headline numbers for the manifest and the comparison table."""
        return {
            "method": self.method,
            "final_loss": round(self.final_loss, 5),
            "student_val_loss": round(self.student_val_loss, 5),
            "student_val_ppl": round(math.exp(min(self.student_val_loss, 20.0)), 4),
            "teacher_val_loss": round(self.teacher_val_loss, 5),
            "teacher_val_ppl": round(math.exp(min(self.teacher_val_loss, 20.0)), 4),
            "student_params": self.student_params,
            "teacher_params": self.teacher_params,
            "compression_ratio": round(self.compression_ratio, 3),
            "non_embedding_compression": round(self.non_embedding_compression, 3),
            "steps": self.steps,
            "warmup_steps": self.warmup_steps,
            "wall_clock_s": round(self.wall_clock_s, 2),
        }


class DistillTrainer:
    """Distils a frozen teacher into a smaller student under one of three objectives."""

    def __init__(
        self,
        teacher: NanoScaleLM,
        student: NanoScaleLM,
        tokenizer: BPETokenizer,
        config: DistillConfig,
        *,
        train_batcher: TokenBatcher | None = None,
        val_batches: list[Batch] | None = None,
        out_dir: str | Path | None = None,
        experiment_config: ExperimentConfig | None = None,
    ) -> None:
        """Freeze the teacher, prepare the student, the data and the optimizer."""
        self.config = config
        self.tokenizer = tokenizer
        seed_all(config.seed)
        self.device = resolve_device(config.device)

        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = student.to(self.device)

        if train_batcher is None:
            raise ValueError(
                "train_batcher is required; build one from packed tokens with "
                "nanoscale.train.TokenBatcher."
            )
        self.batcher = train_batcher
        self.val_batches = val_batches or []

        self.experiment_config = experiment_config
        self.out_dir = Path(out_dir or config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(
            [p for p in self.student.parameters() if p.requires_grad], lr=config.lr
        )
        self.metrics = MetricLogger(self.out_dir, name="metrics")
        self.schedule = ScheduleConfig(name="cosine", warmup_frac=0.05, min_lr_frac=0.1)
        self._baseline = 0.0

    # ------------------------------------------------------------------ helpers

    @torch.no_grad()
    def _teacher_samples(self, prompts: Tensor, n_new: int) -> tuple[Tensor, Tensor]:
        """Sample continuations from the teacher; returns ``(sequences, response_mask)``."""
        gen = torch.Generator().manual_seed(self.config.seed + int(prompts.sum()) % 10_000)
        out = self.teacher.generate(
            prompts,
            max_new_tokens=n_new,
            temperature=1.0,
            top_p=self.config.top_p,
            generator=gen,
        )
        mask = torch.zeros_like(out, dtype=torch.float32)
        mask[:, prompts.shape[1] :] = 1.0
        return out, mask

    @torch.no_grad()
    def _student_rollout(self, prompts: Tensor, n_new: int) -> tuple[Tensor, Tensor]:
        """Sample continuations from the *student*, the on-policy part of MiniLLM."""
        gen = torch.Generator().manual_seed(self.config.seed + int(prompts.sum()) % 10_000)
        out = self.student.generate(
            prompts,
            max_new_tokens=n_new,
            temperature=1.0,
            top_p=self.config.top_p,
            generator=gen,
        )
        mask = torch.zeros_like(out, dtype=torch.float32)
        mask[:, prompts.shape[1] :] = 1.0
        return out, mask

    def _mle_loss(self, batch: Batch) -> DistillLossOutput:
        """Plain next-token MLE, used for the warm-start phase."""
        batch = batch.to(self.device)
        out = self.student(batch.inputs, targets=batch.targets)
        assert out.loss is not None
        return DistillLossOutput(loss=out.loss, ce=out.loss, kd=out.loss.new_zeros(()), extra={})

    def _step_loss(self, batch: Batch, *, step: int) -> DistillLossOutput:
        cfg = self.config
        if step < cfg.warmup_steps:
            return self._mle_loss(batch)
        batch = batch.to(self.device)

        if cfg.method == "forward_kl":
            student_logits = self.student(batch.inputs).logits
            with torch.no_grad():
                teacher_logits = self.teacher(batch.inputs).logits
            mask = torch.ones_like(batch.targets, dtype=torch.float32)
            return forward_kl_loss(
                student_logits,
                teacher_logits,
                batch.targets,
                mask,
                temperature=cfg.temperature,
                alpha=cfg.alpha_ce,
            )

        if cfg.method == "seqkd":
            prompt_len = max(1, batch.inputs.shape[1] // 4)
            prompts = batch.inputs[:, :prompt_len]
            sequences, mask = self._teacher_samples(prompts, cfg.max_new_tokens)
            student_logits = self.student(sequences[:, :-1]).logits
            return sequence_kd_loss(student_logits, sequences[:, 1:], mask[:, 1:])

        # reverse_kl: on-policy MiniLLM
        prompt_len = max(1, batch.inputs.shape[1] // 4)
        prompts = batch.inputs[:, :prompt_len]
        sequences, mask = self._student_rollout(prompts, cfg.max_new_tokens)
        student_logits = self.student(sequences[:, :-1]).logits
        with torch.no_grad():
            teacher_logits = self.teacher(sequences[:, :-1]).logits
        out = reverse_kl_policy_gradient(
            student_logits,
            teacher_logits,
            sequences[:, 1:],
            mask[:, 1:],
            baseline=self._baseline,
            length_normalize=cfg.length_norm,
            single_step_reg=cfg.single_step_reg,
        )
        # EMA baseline over the observed mean reward, which is the standard
        # variance-reduction trick for a REINFORCE-style estimator.
        reward = out.extra.get("mean_reward", 0.0)
        self._baseline = cfg.baseline_ema * self._baseline + (1 - cfg.baseline_ema) * reward
        return out

    @torch.no_grad()
    def evaluate(self, model: NanoScaleLM) -> float:
        """Token-weighted validation cross-entropy."""
        if not self.val_batches:
            return float("nan")
        was_training = model.training
        model.eval()
        total, count = 0.0, 0
        for batch in self.val_batches:
            batch = batch.to(self.device)
            out = model(batch.inputs, targets=batch.targets)
            assert out.loss is not None
            total += float(out.loss) * batch.n_tokens
            count += batch.n_tokens
        model.train(was_training)
        return total / max(1, count)

    # -------------------------------------------------------------------- train

    def train(self) -> DistillResult:
        """Run distillation."""
        cfg = self.config
        self.student.train()
        stream = self.batcher.stream()
        start = time.perf_counter()
        history: list[dict[str, float]] = []
        last = float("nan")

        for step in range(cfg.max_steps):
            self.optimizer.zero_grad(set_to_none=True)
            out = self._step_loss(next(stream), step=step)
            backward(out.loss)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)

            scale = lr_multiplier(step, cfg.max_steps, self.schedule)
            if cfg.method == "reverse_kl" and step >= cfg.warmup_steps:
                # Policy-gradient steps are much noisier than supervised ones.
                scale *= cfg.onpolicy_lr_scale
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", cfg.lr)
                group["lr"] = group["initial_lr"] * scale
            self.optimizer.step()
            last = float(out.loss.detach())

            if (step + 1) % cfg.log_interval == 0 or step == 0:
                row = self.metrics.log(
                    step=step + 1,
                    console=True,
                    phase=1.0 if step < cfg.warmup_steps else 0.0,
                    **out.stats(),
                )
                history.append({k: float(v) for k, v in row.items()})

        wall = time.perf_counter() - start
        student_val = self.evaluate(self.student)
        teacher_val = self.evaluate(self.teacher)
        ckpt = save_checkpoint(
            self.out_dir / "final.pt",
            model=self.student,
            state=TrainState(step=cfg.max_steps),
            config=self.experiment_config,
            extra={"phase": f"phase7-distill-{cfg.method}"},
        )
        result = DistillResult(
            method=cfg.method,
            final_loss=last,
            student_val_loss=student_val,
            teacher_val_loss=teacher_val,
            student_params=self.student.num_parameters(),
            teacher_params=self.teacher.num_parameters(),
            student_non_embedding=self.student.num_parameters(non_embedding=True),
            teacher_non_embedding=self.teacher.num_parameters(non_embedding=True),
            steps=cfg.max_steps,
            warmup_steps=cfg.warmup_steps,
            wall_clock_s=wall,
            checkpoint_path=ckpt,
            history=history,
        )
        self.metrics.summary(**result.summary())
        self.metrics.close()
        write_manifest(
            self.out_dir,
            run_name=f"distill-{cfg.method}",
            phase="phase7-distill",
            seed=cfg.seed,
            config=cfg,
            metrics={
                k: float(v) for k, v in result.summary().items() if isinstance(v, int | float)
            },
        )
        log.info(
            "distill(%s) done: student val %.4f vs teacher %.4f (%.1fx smaller)",
            cfg.method,
            student_val,
            teacher_val,
            result.compression_ratio,
        )
        return result
