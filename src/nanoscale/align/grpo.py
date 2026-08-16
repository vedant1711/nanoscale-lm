r"""GRPO on a verifiable reward (spec B5, the optional RLVR track).

Reference: Shao et al., *DeepSeekMath* (arXiv:2402.03300).

The idea
--------
PPO needs a learned value function to compute advantages — a second network of
comparable size, trained on a moving target. GRPO removes it. For each prompt, sample a
**group** of ``G`` completions, score them all, and use the group itself as the
baseline:

.. math::  A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G}) + \epsilon}

A completion that beat its siblings on the same prompt gets a positive advantage. There
is no critic to train, no critic to go stale, and the baseline is exactly matched to the
prompt's difficulty.

The policy-gradient step is the PPO-clipped surrogate on token log-probability ratios,
plus a KL penalty toward the frozen reference so the policy cannot drift arbitrarily far
to farm the reward.

Verifiable rewards
------------------
The reward here is **programmatic, not learned**: arithmetic problems whose answers are
checked by evaluating them. That is what "RLVR" means, and it is why this track is worth
including at ``nano`` scale — with a learned reward model there would be nothing to
distinguish "the policy improved" from "the policy found a hole in the reward model",
whereas an arithmetic checker cannot be gamed.

What this track is not
----------------------
This is a scoped demonstration on a task a ~5M-parameter model can actually learn, not a
reasoning-RL result. The 2026 successors to cite — and the documented next step — are
**GSPO** (sequence-level importance ratios, which fix the token-level ratio's variance
at long horizons) and **DHPO** (hybrid token+sequence). Neither is implemented here.
"""

from __future__ import annotations

import copy
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from nanoscale.config import GRPOConfig
from nanoscale.model import NanoScaleLM
from nanoscale.optim import AdamW
from nanoscale.tokenizer import BPETokenizer, Message, render_prompt
from nanoscale.train.checkpoint import TrainState, save_checkpoint
from nanoscale.utils import (
    MetricLogger,
    backward,
    get_logger,
    resolve_device,
    seed_all,
    write_manifest,
)

__all__ = [
    "ArithmeticTask",
    "GRPOResult",
    "GRPOTrainer",
    "group_relative_advantages",
    "make_arithmetic_tasks",
    "verify_arithmetic",
]

log = get_logger("nanoscale.align.grpo")

_ANSWER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True, slots=True)
class ArithmeticTask:
    """One programmatically-verifiable arithmetic problem."""

    prompt: str
    answer: int


def make_arithmetic_tasks(
    *, seed: int = 1337, n: int = 256, max_value: int = 9
) -> list[ArithmeticTask]:
    """Generate small addition/subtraction problems with known answers."""
    rng = random.Random(seed)
    tasks: list[ArithmeticTask] = []
    for _ in range(n):
        a = rng.randint(0, max_value)
        b = rng.randint(0, max_value)
        if rng.random() < 0.5:
            tasks.append(ArithmeticTask(prompt=f"What is {a} plus {b}?", answer=a + b))
        else:
            hi, lo = max(a, b), min(a, b)
            tasks.append(ArithmeticTask(prompt=f"What is {hi} minus {lo}?", answer=hi - lo))
    return tasks


def verify_arithmetic(completion: str, answer: int) -> float:
    """Programmatic 0/1 reward: does the completion state the correct number?

    Deliberately strict about *which* number: the first integer in the completion must
    be the answer. Accepting "the answer is somewhere in this text" would reward a model
    that lists every number it knows.
    """
    match = _ANSWER_RE.search(completion)
    if match is None:
        return 0.0
    try:
        return 1.0 if int(match.group()) == answer else 0.0
    except ValueError:  # pragma: no cover - the regex guarantees an int
        return 0.0


def group_relative_advantages(rewards: Tensor, *, eps: float = 1e-4) -> Tensor:
    """Normalise rewards within each group — GRPO's critic-free baseline.

    Args:
        rewards: ``(n_prompts, group_size)`` rewards.
        eps: Floor on the standard deviation.

    Returns:
        ``(n_prompts, group_size)`` advantages, zero-mean within each group.

    A group where every completion scored the same carries no learning signal at all
    (the advantage is exactly zero), which is the correct behaviour: if all ``G`` samples
    are right, or all wrong, the comparison says nothing about which action to reinforce.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps)


@dataclass(slots=True)
class GRPOResult:
    """Outcome of a GRPO run."""

    accuracy_before: float
    accuracy_after: float
    final_reward: float
    steps: int
    wall_clock_s: float
    checkpoint_path: Path | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> dict[str, float | int]:
        """Headline numbers for the manifest."""
        return {
            "accuracy_before": round(self.accuracy_before, 4),
            "accuracy_after": round(self.accuracy_after, 4),
            "accuracy_delta": round(self.accuracy_after - self.accuracy_before, 4),
            "final_reward": round(self.final_reward, 4),
            "steps": self.steps,
            "wall_clock_s": round(self.wall_clock_s, 2),
        }


class GRPOTrainer:
    """Group-relative policy optimization against a verifiable arithmetic reward."""

    def __init__(
        self,
        model: NanoScaleLM,
        tokenizer: BPETokenizer,
        config: GRPOConfig,
        *,
        out_dir: str | Path | None = None,
        tasks: list[ArithmeticTask] | None = None,
    ) -> None:
        """Prepare the policy, the frozen reference and the task pool."""
        self.config = config
        self.tokenizer = tokenizer
        seed_all(config.seed)
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.reference = copy.deepcopy(model).to(self.device).eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

        self.tasks = tasks or make_arithmetic_tasks(seed=config.seed)
        self.eval_tasks = make_arithmetic_tasks(seed=config.seed + 1, n=64)
        self.out_dir = Path(out_dir or config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=config.lr
        )
        self.metrics = MetricLogger(self.out_dir, name="metrics")
        self._rng = random.Random(config.seed)

    # ------------------------------------------------------------------ rollout

    def _prompt_ids(self, task: ArithmeticTask) -> list[int]:
        return render_prompt(self.tokenizer, [Message("user", task.prompt)])

    @torch.no_grad()
    def sample_group(self, task: ArithmeticTask) -> tuple[Tensor, Tensor, Tensor]:
        """Sample ``group_size`` completions for one prompt.

        Returns:
            ``(sequences, response_mask, rewards)`` where ``sequences`` is
            ``(G, prompt + generated)``.
        """
        cfg = self.config
        prompt = torch.tensor([self._prompt_ids(task)], device=self.device)
        prompt = prompt.expand(cfg.group_size, -1).contiguous()
        gen = torch.Generator().manual_seed(self._rng.randrange(2**31))
        out = self.model.generate(
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=0.95,
            generator=gen,
        )
        prompt_len = prompt.shape[1]
        mask = torch.zeros_like(out, dtype=torch.float32)
        mask[:, prompt_len:] = 1.0
        rewards = torch.tensor(
            [
                verify_arithmetic(
                    self.tokenizer.decode(row[prompt_len:].tolist(), skip_special=True),
                    task.answer,
                )
                for row in out
            ],
            device=self.device,
        )
        return out, mask, rewards

    def _token_logps(self, model: NanoScaleLM, sequences: Tensor) -> Tensor:
        """Per-token log-probabilities of ``sequences[:, 1:]`` under ``model``."""
        logits = model(sequences[:, :-1]).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        return logprobs.gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)

    # -------------------------------------------------------------------- eval

    @torch.no_grad()
    def evaluate(self, n: int = 32) -> float:
        """Greedy accuracy on held-out arithmetic problems."""
        self.model.eval()
        correct = 0.0
        for task in self.eval_tasks[:n]:
            ids = torch.tensor([self._prompt_ids(task)], device=self.device)
            out = self.model.generate(
                ids, max_new_tokens=self.config.max_new_tokens, temperature=0.0
            )
            text = self.tokenizer.decode(out[0, ids.shape[1] :].tolist(), skip_special=True)
            correct += verify_arithmetic(text, task.answer)
        self.model.train()
        return correct / max(1, n)

    # ------------------------------------------------------------------- train

    def train(self) -> GRPOResult:
        """Run GRPO."""
        cfg = self.config
        start = time.perf_counter()
        before = self.evaluate()
        history: list[dict[str, float]] = []
        mean_reward = 0.0

        for step in range(cfg.max_steps):
            self.optimizer.zero_grad(set_to_none=True)
            batch_tasks = [self._rng.choice(self.tasks) for _ in range(cfg.n_prompts)]

            all_rewards: list[Tensor] = []
            total_loss = torch.zeros((), device=self.device)
            total_kl = 0.0

            for task in batch_tasks:
                sequences, mask, rewards = self.sample_group(task)
                all_rewards.append(rewards)
                advantages = group_relative_advantages(rewards.unsqueeze(0)).squeeze(0)
                if float(advantages.abs().max()) == 0.0:
                    continue  # a unanimous group carries no signal

                logps = self._token_logps(self.model, sequences)
                with torch.no_grad():
                    old_logps = logps.detach()
                    ref_logps = self._token_logps(self.reference, sequences)

                resp_mask = mask[:, 1:]
                ratio = torch.exp(logps - old_logps)
                adv = advantages.unsqueeze(-1)
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                pg = -torch.min(ratio * adv, clipped * adv)

                # k3 KL estimator (Schulman): unbiased and always non-negative, unlike
                # the naive (logp_ref - logp) difference which can go negative per token.
                log_ratio = ref_logps - logps
                kl = torch.exp(log_ratio) - log_ratio - 1.0

                per_token = pg + cfg.kl_coef * kl
                denom = resp_mask.sum().clamp_min(1.0)
                loss = (per_token * resp_mask).sum() / denom
                total_loss = total_loss + loss / cfg.n_prompts
                total_kl += float(((kl * resp_mask).sum() / denom).detach())

            if total_loss.requires_grad:
                backward(total_loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            mean_reward = float(torch.stack(all_rewards).mean()) if all_rewards else 0.0
            if (step + 1) % cfg.log_interval == 0 or step == 0:
                row = self.metrics.log(
                    step=step + 1,
                    console=True,
                    reward=mean_reward,
                    loss=float(total_loss.detach()),
                    kl=total_kl / max(1, cfg.n_prompts),
                )
                history.append({k: float(v) for k, v in row.items()})

        after = self.evaluate()
        wall = time.perf_counter() - start
        ckpt = save_checkpoint(
            self.out_dir / "final.pt",
            model=self.model,
            state=TrainState(step=cfg.max_steps),
            extra={"phase": "phase6-grpo"},
        )
        result = GRPOResult(
            accuracy_before=before,
            accuracy_after=after,
            final_reward=mean_reward,
            steps=cfg.max_steps,
            wall_clock_s=wall,
            checkpoint_path=ckpt,
            history=history,
        )
        self.metrics.summary(**result.summary())
        self.metrics.close()
        write_manifest(
            self.out_dir,
            run_name="grpo",
            phase="phase6-grpo",
            seed=cfg.seed,
            config=cfg,
            metrics={
                k: float(v) for k, v in result.summary().items() if isinstance(v, int | float)
            },
        )
        log.info("GRPO done: verified accuracy %.3f -> %.3f", before, after)
        return result
