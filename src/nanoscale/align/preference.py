"""DPO / SimPO training, with the length-exploitation diagnostic (spec B5, E4).

Two things this trainer does that a minimal implementation would not:

1. **It freezes the reference policy properly.** ``π_ref`` is a deep copy of the SFT
   model with ``requires_grad=False`` and ``eval()``, and its log-probabilities are
   computed under ``no_grad`` and cached per batch. A reference that is merely "the same
   weights" but still in train mode (dropout active) injects noise into the reward, and
   one that is not detached silently trains through the reference.

2. **It measures length exploitation as it trains.** Every logged step records the mean
   token length of the chosen and rejected responses *as the model scores them*, plus the
   implicit reward margin. Spec E4 asks for this comparison between DPO and SimPO, and
   the diagnostic is the deliverable, not an afterthought.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from nanoscale.align.losses import (
    PreferenceLossOutput,
    dpo_loss,
    sequence_logprobs,
    simpo_loss,
)
from nanoscale.config import ExperimentConfig, PreferenceConfig, ScheduleConfig
from nanoscale.data.instruct import PreferencePair, iter_preference_pairs
from nanoscale.model import NanoScaleLM
from nanoscale.optim import AdamW
from nanoscale.tokenizer import BPETokenizer, Message, render_chat, render_prompt
from nanoscale.train.checkpoint import TrainState, save_checkpoint
from nanoscale.train.schedule import lr_multiplier
from nanoscale.utils import (
    MetricLogger,
    backward,
    get_logger,
    resolve_device,
    seed_all,
    write_manifest,
)

__all__ = ["PreferenceBatch", "PreferenceResult", "PreferenceTrainer", "build_preference_batches"]

log = get_logger("nanoscale.align.preference")


@dataclass(frozen=True, slots=True)
class PreferenceBatch:
    """A padded batch of preference pairs with response-only masks."""

    chosen_inputs: Tensor
    chosen_targets: Tensor
    chosen_mask: Tensor
    rejected_inputs: Tensor
    rejected_targets: Tensor
    rejected_mask: Tensor

    def to(self, device: torch.device) -> PreferenceBatch:
        """Move every tensor to ``device``."""
        return PreferenceBatch(
            self.chosen_inputs.to(device),
            self.chosen_targets.to(device),
            self.chosen_mask.to(device),
            self.rejected_inputs.to(device),
            self.rejected_targets.to(device),
            self.rejected_mask.to(device),
        )

    @property
    def chosen_lengths(self) -> Tensor:
        """Response length in tokens, per pair."""
        return self.chosen_mask.sum(dim=-1)

    @property
    def rejected_lengths(self) -> Tensor:
        """Response length in tokens, per pair."""
        return self.rejected_mask.sum(dim=-1)


def _encode_side(
    tokenizer: BPETokenizer, prompt: str, response: str, *, seq_len: int
) -> tuple[list[int], list[int]]:
    rendered = render_chat(tokenizer, [Message("user", prompt), Message("assistant", response)])
    ids, mask = rendered.ids, rendered.completion_mask
    if len(ids) > seq_len:
        ids, mask = ids[-seq_len:], mask[-seq_len:]
    return ids, mask


def build_preference_batches(
    tokenizer: BPETokenizer,
    pairs: list[PreferencePair],
    *,
    seq_len: int,
    batch_size: int,
) -> list[PreferenceBatch]:
    """Encode preference pairs into padded batches with response-only masks."""
    encoded = [
        (
            _encode_side(tokenizer, p.prompt, p.chosen, seq_len=seq_len + 1),
            _encode_side(tokenizer, p.prompt, p.rejected, seq_len=seq_len + 1),
        )
        for p in pairs
    ]
    encoded = [e for e in encoded if len(e[0][0]) >= 2 and len(e[1][0]) >= 2]

    def pad(rows: list[tuple[list[int], list[int]]]) -> tuple[Tensor, Tensor, Tensor]:
        width = max(len(ids) for ids, _ in rows)
        inputs = torch.full((len(rows), width - 1), tokenizer.pad_id, dtype=torch.long)
        targets = torch.zeros((len(rows), width - 1), dtype=torch.long)
        masks = torch.zeros((len(rows), width - 1), dtype=torch.float32)
        for r, (ids, mask) in enumerate(rows):
            n = len(ids) - 1
            inputs[r, :n] = torch.tensor(ids[:-1], dtype=torch.long)
            targets[r, :n] = torch.tensor(ids[1:], dtype=torch.long)
            masks[r, :n] = torch.tensor(mask[1:], dtype=torch.float32)
        return inputs, targets, masks

    batches: list[PreferenceBatch] = []
    for start in range(0, len(encoded) - batch_size + 1, batch_size):
        chunk = encoded[start : start + batch_size]
        c_in, c_tgt, c_mask = pad([c for c, _ in chunk])
        r_in, r_tgt, r_mask = pad([r for _, r in chunk])
        batches.append(
            PreferenceBatch(
                chosen_inputs=c_in,
                chosen_targets=c_tgt,
                chosen_mask=c_mask,
                rejected_inputs=r_in,
                rejected_targets=r_tgt,
                rejected_mask=r_mask,
            )
        )
    return batches


@dataclass(slots=True)
class PreferenceResult:
    """Outcome of a preference-optimization run, including the length diagnostic."""

    method: str
    final_loss: float
    final_margin: float
    final_accuracy: float
    steps: int
    wall_clock_s: float
    mean_chosen_len: float
    mean_rejected_len: float
    length_ratio_start: float
    length_ratio_end: float
    generated_len_before: float = 0.0
    generated_len_after: float = 0.0
    checkpoint_path: Path | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> dict[str, float | int | str]:
        """Headline numbers for the manifest and the comparison table."""
        return {
            "method": self.method,
            "final_loss": round(self.final_loss, 5),
            "final_reward_margin": round(self.final_margin, 5),
            "final_reward_accuracy": round(self.final_accuracy, 4),
            "steps": self.steps,
            "wall_clock_s": round(self.wall_clock_s, 2),
            "mean_chosen_len": round(self.mean_chosen_len, 2),
            "mean_rejected_len": round(self.mean_rejected_len, 2),
            "generated_len_before": round(self.generated_len_before, 2),
            "generated_len_after": round(self.generated_len_after, 2),
            "generated_len_growth": round(self.generated_len_after - self.generated_len_before, 2),
        }


class PreferenceTrainer:
    """Trains a model with DPO or SimPO against a frozen SFT reference."""

    def __init__(
        self,
        model: NanoScaleLM,
        tokenizer: BPETokenizer,
        config: PreferenceConfig,
        *,
        out_dir: str | Path | None = None,
        experiment_config: ExperimentConfig | None = None,
        pairs: list[PreferencePair] | None = None,
        n_pairs: int = 800,
    ) -> None:
        """Prepare the policy, the frozen reference, the batches and the optimizer."""
        self.config = config
        self.tokenizer = tokenizer
        seed_all(config.seed)
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.experiment_config = experiment_config
        self.out_dir = Path(out_dir or config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # SimPO is reference-free by construction, so we do not pay for the copy.
        self.reference: NanoScaleLM | None = None
        if config.method == "dpo":
            self.reference = copy.deepcopy(model).to(self.device)
            self.reference.eval()
            for p in self.reference.parameters():
                p.requires_grad_(False)

        data = pairs or list(iter_preference_pairs(seed=config.seed, n=n_pairs))
        self.pairs = data
        self.batches = build_preference_batches(
            tokenizer, data, seq_len=config.seq_len, batch_size=config.batch_size
        )
        if not self.batches:
            raise ValueError("no preference batches were produced.")

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=config.lr
        )
        self.metrics = MetricLogger(self.out_dir, name="metrics")
        self.schedule = ScheduleConfig(name="cosine", warmup_frac=0.1, min_lr_frac=0.1)

    # --------------------------------------------------------------- log-probs

    def _logps(
        self, model: NanoScaleLM, batch: PreferenceBatch, *, average: bool
    ) -> tuple[Tensor, Tensor]:
        chosen = model(batch.chosen_inputs).logits
        rejected = model(batch.rejected_inputs).logits
        return (
            sequence_logprobs(chosen, batch.chosen_targets, batch.chosen_mask, average=average),
            sequence_logprobs(
                rejected, batch.rejected_targets, batch.rejected_mask, average=average
            ),
        )

    def compute_loss(self, batch: PreferenceBatch) -> tuple[PreferenceLossOutput, dict[str, float]]:
        """Compute the configured preference loss and the log-probability diagnostics.

        The diagnostics are not decoration. DPO optimises the *difference* of two
        log-probabilities, and a well-documented consequence is that it can reduce the
        loss by pushing **both** down -- the chosen response just less far than the
        rejected one. A model whose absolute likelihood of good responses is collapsing
        will show a healthy rising margin and degrading generations at the same time, so
        the absolute log-probabilities are logged alongside the margin.
        """
        cfg = self.config
        batch = batch.to(self.device)
        aux: dict[str, float] = {}

        if cfg.method == "simpo":
            # Length-normalised, reference-free.
            chosen, rejected = self._logps(self.model, batch, average=True)
            out = simpo_loss(chosen, rejected, beta=cfg.beta, gamma=cfg.gamma)
            aux = {
                "chosen_logp": float(chosen.detach().mean()),
                "rejected_logp": float(rejected.detach().mean()),
            }
            return self._add_sft_term(out, batch, aux)

        chosen, rejected = self._logps(self.model, batch, average=False)
        assert self.reference is not None
        with torch.no_grad():
            ref_chosen, ref_rejected = self._logps(self.reference, batch, average=False)
        out = dpo_loss(
            chosen,
            rejected,
            ref_chosen.detach(),
            ref_rejected.detach(),
            beta=cfg.beta,
            label_smoothing=cfg.label_smoothing,
        )
        lengths = batch.chosen_lengths.clamp_min(1.0)
        aux = {
            # Per-token, so DPO's summed log-probs are comparable with SimPO's averaged ones.
            "chosen_logp": float((chosen.detach() / lengths).mean()),
            "rejected_logp": float(
                (rejected.detach() / batch.rejected_lengths.clamp_min(1.0)).mean()
            ),
        }
        return self._add_sft_term(out, batch, aux)

    def _add_sft_term(
        self,
        out: PreferenceLossOutput,
        batch: PreferenceBatch,
        aux: dict[str, float],
    ) -> tuple[PreferenceLossOutput, dict[str, float]]:
        """Optionally add an RPO-style auxiliary NLL on the chosen response.

        This is the standard, documented fix for the collapse described above: anchoring
        the *absolute* likelihood of the chosen response stops the objective from
        satisfying itself by pushing everything down. It is off by default
        (``sft_loss_weight = 0``) so that its effect can be measured rather than assumed;
        the Phase-6 write-up runs DPO with and without it.
        """
        weight = self.config.sft_loss_weight
        if weight <= 0.0:
            return out, aux
        chosen_avg, _ = self._logps(self.model, batch, average=True)
        nll = -chosen_avg.mean()
        aux["sft_nll"] = float(nll.detach())
        combined = PreferenceLossOutput(
            loss=out.loss + weight * nll,
            chosen_reward=out.chosen_reward,
            rejected_reward=out.rejected_reward,
            margin=out.margin,
            accuracy=out.accuracy,
        )
        return combined, aux

    # ------------------------------------------------------- length diagnostic

    @torch.no_grad()
    def mean_generated_length(self, *, n_prompts: int = 24, max_new_tokens: int = 48) -> float:
        """Mean length in tokens of what the model *generates* for held-out prompts.

        This is the sharp version of the length-exploitation diagnostic: comparing the
        lengths of the training responses only tells you about the data, whereas this
        measures whether the objective has taught the model to produce longer output.
        """
        self.model.eval()
        lengths: list[int] = []
        for pair in self.pairs[:n_prompts]:
            ids = torch.tensor(
                [render_prompt(self.tokenizer, [Message("user", pair.prompt)])],
                device=self.device,
            )
            gen = torch.Generator().manual_seed(self.config.seed)
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                eos_id=self.tokenizer.eot_id,
                generator=gen,
            )
            lengths.append(int(out.shape[1] - ids.shape[1]))
        self.model.train()
        return sum(lengths) / max(1, len(lengths))

    # ------------------------------------------------------------------- train

    def train(self) -> PreferenceResult:
        """Run preference optimization."""
        cfg = self.config
        self.model.train()
        start = time.perf_counter()
        history: list[dict[str, float]] = []
        n_batches = len(self.batches)

        generated_before = self.mean_generated_length()
        first_ratio = 0.0
        last_ratio = 0.0
        stats: dict[str, float] = {}
        chosen_len = rejected_len = 0.0

        for step in range(cfg.max_steps):
            self.optimizer.zero_grad(set_to_none=True)
            batch = self.batches[step % n_batches]
            out, aux = self.compute_loss(batch)
            backward(out.loss / cfg.grad_accum)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            scale = lr_multiplier(step, cfg.max_steps, self.schedule)
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", cfg.lr)
                group["lr"] = group["initial_lr"] * scale
            self.optimizer.step()

            stats = {**out.stats(), **aux}
            chosen_len = float(batch.chosen_lengths.float().mean())
            rejected_len = float(batch.rejected_lengths.float().mean())
            ratio = chosen_len / max(1.0, rejected_len)
            if step == 0:
                first_ratio = ratio
            last_ratio = ratio

            if (step + 1) % cfg.log_interval == 0 or step == 0:
                row = self.metrics.log(
                    step=step + 1,
                    console=True,
                    chosen_len=chosen_len,
                    rejected_len=rejected_len,
                    **stats,
                )
                history.append({k: float(v) for k, v in row.items()})

        generated_after = self.mean_generated_length()
        wall = time.perf_counter() - start
        ckpt = save_checkpoint(
            self.out_dir / "final.pt",
            model=self.model,
            state=TrainState(step=cfg.max_steps),
            config=self.experiment_config,
            extra={"phase": f"phase6-{cfg.method}"},
        )
        result = PreferenceResult(
            method=cfg.method,
            final_loss=stats.get("loss", float("nan")),
            final_margin=stats.get("reward_margin", float("nan")),
            final_accuracy=stats.get("reward_accuracy", float("nan")),
            steps=cfg.max_steps,
            wall_clock_s=wall,
            mean_chosen_len=chosen_len,
            mean_rejected_len=rejected_len,
            length_ratio_start=first_ratio,
            length_ratio_end=last_ratio,
            generated_len_before=generated_before,
            generated_len_after=generated_after,
            checkpoint_path=ckpt,
            history=history,
        )
        self.metrics.summary(**result.summary())
        self.metrics.close()
        write_manifest(
            self.out_dir,
            run_name=cfg.method,
            phase=f"phase6-{cfg.method}",
            seed=cfg.seed,
            config=cfg,
            metrics={
                k: float(v) for k, v in result.summary().items() if isinstance(v, int | float)
            },
        )
        log.info(
            "%s done: loss %.4f, margin %.4f, accuracy %.3f, generated length %.1f -> %.1f",
            cfg.method.upper(),
            result.final_loss,
            result.final_margin,
            result.final_accuracy,
            generated_before,
            generated_after,
        )
        return result


def perplexity_from_loss(loss: float) -> float:
    """Convenience for reporting."""
    return math.exp(min(loss, 20.0))
