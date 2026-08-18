"""Supervised fine-tuning with completion-only loss masking (spec B5).

The one thing that has to be right
-----------------------------------
SFT trains the model to *produce assistant turns*, not to *predict the whole
transcript*. If the loss covers prompt tokens too, the model spends capacity learning to
generate plausible user messages, and, worse; it learns that the ``<user>`` role marker
is followed by user-like text, which makes it prone to continuing the conversation on
the user's behalf at inference time.

So the loss is masked to completion tokens only. The mask comes from
:func:`~nanoscale.tokenizer.chat.render_chat`, which marks the assistant reply and its
terminating ``<eot>`` (the model must learn where to stop) and nothing else.

Spec Phase 6 requires this to be *verified by test*, and
``tests/unit/test_align.py::test_prompt_tokens_receive_no_gradient`` does it the
strongest way available: it perturbs the prompt logits and checks the gradient with
respect to them is exactly zero.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from nanoscale.config import ExperimentConfig, ScheduleConfig, SFTConfig
from nanoscale.data.instruct import InstructExample, iter_instructions
from nanoscale.model import IGNORE_INDEX, NanoScaleLM
from nanoscale.optim import AdamW
from nanoscale.tokenizer import BPETokenizer, Message, render_chat
from nanoscale.train.checkpoint import TrainState, save_checkpoint
from nanoscale.train.schedule import lr_multiplier
from nanoscale.utils import (
    MetricLogger,
    autocast_context,
    backward,
    get_logger,
    resolve_device,
    resolve_dtype,
    seed_all,
    write_manifest,
)

__all__ = ["SFTBatch", "SFTResult", "SFTTrainer", "build_sft_batches", "encode_example"]

log = get_logger("nanoscale.align.sft")


@dataclass(frozen=True, slots=True)
class SFTBatch:
    """A padded batch of chat examples with a completion-only loss mask."""

    inputs: Tensor  # (B, T)
    targets: Tensor  # (B, T) with IGNORE_INDEX on masked positions
    completion_mask: Tensor  # (B, T) 1 where supervised

    @property
    def n_supervised(self) -> int:
        """Number of loss-carrying tokens."""
        return int(self.completion_mask.sum())

    def to(self, device: torch.device) -> SFTBatch:
        """Move every tensor to ``device``."""
        return SFTBatch(
            self.inputs.to(device),
            self.targets.to(device),
            self.completion_mask.to(device),
        )


def encode_example(
    tokenizer: BPETokenizer, example: InstructExample, *, seq_len: int
) -> tuple[list[int], list[int]]:
    """Render one instruction example to ``(ids, completion_mask)``, truncated to ``seq_len``.

    Truncation is from the **left** of the prompt rather than the right of the response:
    losing the start of a long instruction is recoverable, but truncating the response
    teaches the model to stop mid-sentence.
    """
    messages: list[Message] = []
    if example.system:
        messages.append(Message("system", example.system))
    messages.append(Message("user", example.instruction))
    messages.append(Message("assistant", example.response))

    rendered = render_chat(tokenizer, messages)
    ids, mask = rendered.ids, rendered.completion_mask
    if len(ids) > seq_len:
        ids = ids[-seq_len:]
        mask = mask[-seq_len:]
    return ids, mask


def build_sft_batches(
    tokenizer: BPETokenizer,
    examples: list[InstructExample],
    *,
    seq_len: int,
    batch_size: int,
) -> list[SFTBatch]:
    """Encode and pad instruction examples into fixed-shape batches.

    Padding uses ``<pad>`` for the inputs and :data:`IGNORE_INDEX` for the targets, so
    padded positions contribute nothing to the loss. The inputs/targets shift is applied
    here: ``inputs = ids[:-1]``, ``targets = ids[1:]``, with the mask aligned to the
    targets.
    """
    encoded = [encode_example(tokenizer, ex, seq_len=seq_len + 1) for ex in examples]
    encoded = [(ids, mask) for ids, mask in encoded if len(ids) >= 2]
    batches: list[SFTBatch] = []

    for start in range(0, len(encoded) - batch_size + 1, batch_size):
        chunk = encoded[start : start + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        inputs = torch.full((len(chunk), width - 1), tokenizer.pad_id, dtype=torch.long)
        targets = torch.full((len(chunk), width - 1), IGNORE_INDEX, dtype=torch.long)
        masks = torch.zeros((len(chunk), width - 1), dtype=torch.long)
        for row, (ids, mask) in enumerate(chunk):
            n = len(ids) - 1
            inputs[row, :n] = torch.tensor(ids[:-1], dtype=torch.long)
            supervised = torch.tensor(mask[1:], dtype=torch.long)
            masks[row, :n] = supervised
            tgt = torch.tensor(ids[1:], dtype=torch.long)
            targets[row, :n] = torch.where(
                supervised.bool(), tgt, torch.full_like(tgt, IGNORE_INDEX)
            )
        batches.append(SFTBatch(inputs=inputs, targets=targets, completion_mask=masks))
    return batches


@dataclass(slots=True)
class SFTResult:
    """Outcome of an SFT run."""

    final_loss: float
    best_val_loss: float
    steps: int
    supervised_tokens: int
    wall_clock_s: float
    checkpoint_path: Path | None = None

    def summary(self) -> dict[str, float | int]:
        """Headline numbers for the manifest."""
        return {
            "final_loss": round(self.final_loss, 5),
            "best_val_loss": round(self.best_val_loss, 5),
            "final_perplexity": round(math.exp(min(self.final_loss, 20.0)), 4),
            "steps": self.steps,
            "supervised_tokens": self.supervised_tokens,
            "wall_clock_s": round(self.wall_clock_s, 2),
        }


class SFTTrainer:
    """Instruction-tunes a pretrained model with completion-masked cross-entropy."""

    def __init__(
        self,
        model: NanoScaleLM,
        tokenizer: BPETokenizer,
        config: SFTConfig,
        *,
        out_dir: str | Path | None = None,
        experiment_config: ExperimentConfig | None = None,
        examples: list[InstructExample] | None = None,
        n_examples: int = 2400,
    ) -> None:
        """Prepare the model, batches and optimizer for supervised fine-tuning."""
        self.config = config
        self.tokenizer = tokenizer
        seed_all(config.seed)
        self.device = resolve_device(config.device)
        self.amp_dtype = resolve_dtype("fp32", self.device)
        self.model = model.to(self.device)
        self.experiment_config = experiment_config
        self.out_dir = Path(out_dir or config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        data = examples or list(iter_instructions(seed=config.seed, n=n_examples))
        split = max(1, int(len(data) * 0.9))
        self.train_batches = build_sft_batches(
            tokenizer, data[:split], seq_len=config.seq_len, batch_size=config.batch_size
        )
        self.val_batches = build_sft_batches(
            tokenizer, data[split:], seq_len=config.seq_len, batch_size=config.batch_size
        )
        if not self.train_batches:
            raise ValueError("no SFT batches were produced; check seq_len and batch_size.")

        # SFT is a short, low-LR pass over a small dataset: plain AdamW on everything is
        # the standard choice, and Muon's advantage does not apply over ~100 steps.
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.metrics = MetricLogger(self.out_dir, name="metrics")
        self.schedule = ScheduleConfig(
            name="cosine", warmup_frac=config.warmup_frac, min_lr_frac=0.05
        )

    def _loss(self, batch: SFTBatch) -> tuple[Tensor, int]:
        batch = batch.to(self.device)
        with autocast_context(self.device, self.amp_dtype):
            out = self.model(batch.inputs, targets=batch.targets)
        assert out.loss is not None
        return out.loss, batch.n_supervised

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean completion-masked loss over the held-out split."""
        if not self.val_batches:
            return float("nan")
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.val_batches:
            loss, n = self._loss(batch)
            total += float(loss) * n
            count += n
        self.model.train()
        return total / max(1, count)

    def train(self) -> SFTResult:
        """Run supervised fine-tuning."""
        cfg = self.config
        self.model.train()
        start = time.perf_counter()
        best = float("inf")
        last = float("nan")
        supervised = 0
        n_batches = len(self.train_batches)

        for step in range(cfg.max_steps):
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro in range(cfg.grad_accum):
                batch = self.train_batches[(step * cfg.grad_accum + micro) % n_batches]
                loss, n = self._loss(batch)
                backward(loss / cfg.grad_accum)
                step_loss += float(loss.detach()) / cfg.grad_accum
                supervised += n

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scale = lr_multiplier(step, cfg.max_steps, self.schedule)
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", cfg.lr)
                group["lr"] = group["initial_lr"] * scale
            self.optimizer.step()
            last = step_loss

            if (step + 1) % cfg.log_interval == 0 or step == 0:
                self.metrics.log(
                    step=step + 1,
                    console=True,
                    loss=step_loss,
                    ppl=math.exp(min(step_loss, 20.0)),
                    lr=self.optimizer.param_groups[0]["lr"],
                    supervised_tokens=supervised,
                )
            if (step + 1) % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
                val = self.evaluate()
                best = min(best, val)
                self.metrics.log(step=step + 1, console=True, val_loss=val)

        wall = time.perf_counter() - start
        ckpt = save_checkpoint(
            self.out_dir / "final.pt",
            model=self.model,
            state=TrainState(step=cfg.max_steps),
            config=self.experiment_config,
            extra={"phase": "phase6-sft"},
        )
        result = SFTResult(
            final_loss=last,
            best_val_loss=best,
            steps=cfg.max_steps,
            supervised_tokens=supervised,
            wall_clock_s=wall,
            checkpoint_path=ckpt,
        )
        self.metrics.summary(**result.summary())
        self.metrics.close()
        write_manifest(
            self.out_dir,
            run_name="sft",
            phase="phase6-sft",
            seed=cfg.seed,
            config=cfg,
            metrics={
                k: float(v) for k, v in result.summary().items() if isinstance(v, int | float)
            },
        )
        log.info("SFT done: loss %.4f (val %.4f) over %d steps", last, best, cfg.max_steps)
        return result
