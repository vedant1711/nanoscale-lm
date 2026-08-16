"""Post-training: SFT, DPO, SimPO and the optional GRPO-RLVR track."""

from __future__ import annotations

from nanoscale.align.grpo import (
    ArithmeticTask,
    GRPOResult,
    GRPOTrainer,
    group_relative_advantages,
    make_arithmetic_tasks,
    verify_arithmetic,
)
from nanoscale.align.losses import (
    PreferenceLossOutput,
    dpo_loss,
    sequence_logprobs,
    simpo_loss,
)
from nanoscale.align.preference import (
    PreferenceBatch,
    PreferenceResult,
    PreferenceTrainer,
    build_preference_batches,
)
from nanoscale.align.sft import (
    SFTBatch,
    SFTResult,
    SFTTrainer,
    build_sft_batches,
    encode_example,
)

__all__ = [
    "ArithmeticTask",
    "GRPOResult",
    "GRPOTrainer",
    "PreferenceBatch",
    "PreferenceLossOutput",
    "PreferenceResult",
    "PreferenceTrainer",
    "SFTBatch",
    "SFTResult",
    "SFTTrainer",
    "build_preference_batches",
    "build_sft_batches",
    "dpo_loss",
    "encode_example",
    "group_relative_advantages",
    "make_arithmetic_tasks",
    "sequence_logprobs",
    "simpo_loss",
    "verify_arithmetic",
]
