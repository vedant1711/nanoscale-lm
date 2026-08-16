"""Knowledge distillation: forward-KL, SeqKD and MiniLLM-style on-policy reverse KL."""

from __future__ import annotations

from nanoscale.distill.losses import (
    DistillLossOutput,
    forward_kl_loss,
    reverse_kl_policy_gradient,
    sequence_kd_loss,
    token_kl,
)
from nanoscale.distill.trainer import DistillResult, DistillTrainer

__all__ = [
    "DistillLossOutput",
    "DistillResult",
    "DistillTrainer",
    "forward_kl_loss",
    "reverse_kl_policy_gradient",
    "sequence_kd_loss",
    "token_kl",
]
