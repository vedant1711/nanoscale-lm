"""A tiny, committed, offline benchmark suite (spec A6).

Scope, stated up front
----------------------
This is **not** HellaSwag or ARC. A 5M-parameter model trained on a synthetic story
corpus with a 1024-token vocabulary cannot attempt those, and running it on them would
produce a number indistinguishable from chance that told you nothing. Reporting such a
number would be worse than reporting none.

What this suite does instead is test the things the ``nano`` model *could* plausibly have
learned from its training distribution, using the same multiple-choice-by-log-likelihood
protocol the real benchmarks use:

* **agreement**: does the model prefer the pronoun matching the protagonist's gender?
* **coreference**: does it prefer the continuation that refers to the entity actually
  introduced?
* **schema**: does it prefer a grammatical continuation over a scrambled one?
* **arithmetic**: the GSM8K-style verifiable-answer task from the GRPO track.

Scoring protocol
----------------
For each question, every candidate completion is scored by its **length-normalised**
total log-likelihood given the context, and the highest wins. Length normalisation
matters: without it the model would systematically prefer whichever candidate happens to
be shorter, and the benchmark would measure candidate length rather than model
preference.

Determinism
-----------
The questions are hard-coded, not sampled, so the suite is byte-identical across runs and
machines. A score from it is comparable to any other score from it at the same commit.

It saturates, and that is what it is for
-----------------------------------------
The trained ``nano`` base model scores **100% on all four tasks**. A saturated benchmark
cannot rank models that are all good at it, so this suite is **a degradation detector**,
not a quality ladder: its job in Arc 2 is to answer "did 4-bit quantization break
something?", and a drop from 100% is unambiguous. It cannot answer "is the distilled
student better than the quantized one" and is not used for that. The `n=28` binomial
standard error (~9 points near 50%) is reported so that any difference under about 10
points reads as the noise it is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from nanoscale.tokenizer import BPETokenizer

__all__ = [
    "TASKS",
    "BenchmarkResult",
    "MultipleChoiceQuestion",
    "run_tiny_bench",
    "score_choice",
]


@dataclass(frozen=True, slots=True)
class MultipleChoiceQuestion:
    """One question: a context, several continuations, and the index of the right one."""

    task: str
    context: str
    choices: tuple[str, ...]
    answer: int

    def __post_init__(self) -> None:
        """Validate the question."""
        if not 0 <= self.answer < len(self.choices):
            raise ValueError(f"answer index {self.answer} out of range for {self.choices}.")
        if len(self.choices) < 2:
            raise ValueError("a multiple-choice question needs at least two choices.")


def _agreement_context(name: str) -> str:
    """A story prefix positioned exactly where a pronoun sentence occurs in training.

    The context length and structure are not incidental. An earlier version of this probe
    used a single sentence of context (``"{name} went to the park with a small dog."``)
    and the model scored at chance, 3/6, which read as "the model has not learned
    agreement". It had: the corpus never places a pronoun sentence that early, so the
    probe was off-distribution in *structure* even though every word in it was in
    vocabulary. With the prefix positioned where pronouns actually occur, the same
    checkpoint scores 12/12.

    The lesson generalises past this repo: a benchmark that presents a model with a
    context shape it never saw measures the mismatch, not the capability.
    """
    return (
        f"It was a sunny day. {name} went to the park with a small dog. "
        f"{name} wanted to find a red ball. But a red ball was stuck under a tree."
    )


def _agreement() -> list[MultipleChoiceQuestion]:
    female = ("Lily", "Mia", "Anna", "Nora", "Ella", "Ruby")
    male = ("Tom", "Ben", "Sam", "Max", "Leo", "Finn")
    choices = (
        " She counted to three and tried again.",
        " He counted to three and tried again.",
    )
    return [
        MultipleChoiceQuestion(
            task="agreement",
            context=_agreement_context(name),
            choices=choices,
            answer=0 if name in female else 1,
        )
        for name in (*female, *male)
    ]


def _coreference() -> list[MultipleChoiceQuestion]:
    pairs = (
        ("a red ball", "a blue box"),
        ("a shiny key", "a paper boat"),
        ("a big book", "a warm hat"),
        ("a long rope", "a glass jar"),
        ("a silver spoon", "a torn map"),
    )
    questions: list[MultipleChoiceQuestion] = []
    for right, wrong in pairs:
        questions.append(
            MultipleChoiceQuestion(
                task="coreference",
                context=f"Lily wanted to find {right}.",
                choices=(
                    f" But {right} was stuck under a tree.",
                    f" But {wrong} was stuck under a tree.",
                ),
                answer=0,
            )
        )
    return questions


def _schema() -> list[MultipleChoiceQuestion]:
    good_bad = (
        ("It was a sunny day.", "day sunny a was It."),
        ("The wind was cold.", "cold was wind The."),
        ("Tom smiled and went home.", "home went and smiled Tom."),
        ("At last the key was free.", "free was key the last At."),
        ("She counted to three and tried again.", "again tried and three to counted She."),
    )
    return [
        MultipleChoiceQuestion(
            task="schema",
            context="Here is a sentence from the story.",
            choices=(f" {good}", f" {bad}"),
            answer=0,
        )
        for good, bad in good_bad
    ]


def _arithmetic() -> list[MultipleChoiceQuestion]:
    problems = ((2, 3), (4, 1), (5, 4), (7, 2), (6, 3), (8, 1))
    return [
        MultipleChoiceQuestion(
            task="arithmetic",
            context=f"What is {a} plus {b}?",
            choices=(f" The answer is {a + b}.", f" The answer is {a + b + 1}."),
            answer=0,
        )
        for a, b in problems
    ]


#: The committed question set. Hard-coded so scores are comparable across runs.
TASKS: tuple[MultipleChoiceQuestion, ...] = tuple(
    _agreement() + _coreference() + _schema() + _arithmetic()
)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Accuracy overall and per task, with a binomial standard error."""

    accuracy: float
    n_questions: int
    per_task: dict[str, float]

    @property
    def stderr(self) -> float:
        """Binomial standard error of the overall accuracy.

        With ~30 questions this is around 0.09, which is the point: a 5-point difference
        between two models on this suite is noise, and the interval says so.
        """
        p = self.accuracy
        return math.sqrt(max(0.0, p * (1 - p)) / max(1, self.n_questions))

    @property
    def chance(self) -> float:
        """Chance accuracy for this question set (all questions are binary here)."""
        return 0.5

    def summary(self) -> dict[str, float | int]:
        """Flat numbers for the results table."""
        return {
            "accuracy": round(self.accuracy, 4),
            "accuracy_stderr": round(self.stderr, 4),
            "n_questions": self.n_questions,
            "chance": self.chance,
            **{f"acc_{task}": round(score, 4) for task, score in sorted(self.per_task.items())},
        }

    def __str__(self) -> str:
        """Render accuracy with its interval and the chance baseline."""
        return (
            f"{self.accuracy:.1%} ± {self.stderr:.1%} "
            f"(n={self.n_questions}, chance={self.chance:.0%})"
        )


@torch.no_grad()
def score_choice(
    model: nn.Module,
    tokenizer: BPETokenizer,
    context: str,
    choice: str,
    *,
    device: torch.device | None = None,
) -> float:
    """Length-normalised log-likelihood of ``choice`` given ``context``.

    Normalising by the number of scored tokens is what stops the benchmark from
    measuring candidate length instead of model preference.
    """
    context_ids = tokenizer.encode(context, add_bos=True)
    choice_ids = tokenizer.encode(choice)
    if not choice_ids:
        return float("-inf")

    ids = torch.tensor([context_ids + choice_ids], device=device)
    logits = model(ids[:, :-1]).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[:, 1:]
    gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Score only the choice tokens: the context is identical across candidates, so
    # including it would add the same constant to every score.
    start = len(context_ids) - 1
    scored = gathered[0, start:]
    return float(scored.sum() / max(1, scored.numel()))


@torch.no_grad()
def run_tiny_bench(
    model: nn.Module,
    tokenizer: BPETokenizer,
    questions: Sequence[MultipleChoiceQuestion] = TASKS,
    *,
    device: torch.device | None = None,
) -> BenchmarkResult:
    """Score a model on the committed question set."""
    was_training = model.training
    model.eval()
    try:
        correct = 0
        per_task_correct: dict[str, int] = {}
        per_task_total: dict[str, int] = {}

        for question in questions:
            scores = [
                score_choice(model, tokenizer, question.context, choice, device=device)
                for choice in question.choices
            ]
            predicted = max(range(len(scores)), key=lambda i: scores[i])
            hit = int(predicted == question.answer)
            correct += hit
            per_task_correct[question.task] = per_task_correct.get(question.task, 0) + hit
            per_task_total[question.task] = per_task_total.get(question.task, 0) + 1

        return BenchmarkResult(
            accuracy=correct / max(1, len(questions)),
            n_questions=len(questions),
            per_task={
                task: per_task_correct[task] / per_task_total[task] for task in per_task_total
            },
        )
    finally:
        model.train(was_training)
