"""Training: data pipeline, LR schedules, checkpointing and the pretraining loop."""

from __future__ import annotations

from nanoscale.train.checkpoint import (
    CHECKPOINT_VERSION,
    TrainState,
    load_checkpoint,
    load_config_from_checkpoint,
    save_checkpoint,
)
from nanoscale.train.data import (
    Batch,
    PackedTokens,
    TokenBatcher,
    build_packed_tokens,
    iter_hf_documents,
    iter_text_documents,
    tokenize_documents,
)
from nanoscale.train.loop import Trainer, TrainResult, evaluate_loss, grad_global_norm
from nanoscale.train.schedule import lr_multiplier, make_schedule, weight_decay_multiplier

__all__ = [
    "CHECKPOINT_VERSION",
    "Batch",
    "PackedTokens",
    "TokenBatcher",
    "TrainResult",
    "TrainState",
    "Trainer",
    "build_packed_tokens",
    "evaluate_loss",
    "grad_global_norm",
    "iter_hf_documents",
    "iter_text_documents",
    "load_checkpoint",
    "load_config_from_checkpoint",
    "lr_multiplier",
    "make_schedule",
    "save_checkpoint",
    "tokenize_documents",
    "weight_decay_multiplier",
]
