"""Multi-token prediction heads (spec B2, optional).

Standard LM training supervises one objective per position: predict token ``t+1``.
Multi-token prediction (Gloeckle et al., *Better & Faster Large Language Models via
Multi-token Prediction*; also in the modded-nanoGPT stack and DeepSeek-V3) adds
auxiliary heads that predict ``t+2``, ``t+3``, … from the same final hidden state. Two
things come out of it:

1. **A denser training signal.** Each position now carries several prediction targets,
   which empirically improves sample efficiency at fixed token budget.
2. **A free draft model.** At inference the extra heads propose the next few tokens
   without a second network — which is precisely the Medusa-style self-speculation used
   in Phase 9. This is the seam that ties Arc 1 to Arc 2: an architecture choice made
   during pretraining pays off as an inference optimisation later.

Each head is deliberately tiny — one ``d_model → d_model`` transform, a nonlinearity, a
parameter-free norm, and its own unembedding — so that the auxiliary objective adds
capacity for prediction rather than a second model.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nanoscale.model.norm import RMSNorm

__all__ = ["MTPHead", "MultiTokenPredictionHeads"]


class MTPHead(nn.Module):
    """One auxiliary head predicting the token ``offset`` steps ahead."""

    def __init__(
        self, d_model: int, vocab_size: int, offset: int, *, norm_eps: float = 1e-5
    ) -> None:
        """Build a head that predicts token ``t + offset``."""
        super().__init__()
        self.offset = offset
        self.transform = nn.Linear(d_model, d_model, bias=False)
        # Parameter-free norm: the analytic parameter count in ModelConfig assumes each
        # head costs exactly d_model^2 + vocab*d_model, and a test enforces that.
        self.norm = RMSNorm(d_model, eps=norm_eps, elementwise_affine=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map final hidden states to logits for position ``t + offset``."""
        logits: torch.Tensor = self.head(self.norm(F.silu(self.transform(hidden))))
        return logits


class MultiTokenPredictionHeads(nn.Module):
    """A stack of MTP heads predicting ``t+2 … t+1+n_heads``."""

    def __init__(
        self, d_model: int, vocab_size: int, n_heads: int, *, norm_eps: float = 1e-5
    ) -> None:
        """Build ``n_heads`` auxiliary heads at increasing offsets."""
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList(
            MTPHead(d_model, vocab_size, offset=i + 2, norm_eps=norm_eps) for i in range(n_heads)
        )

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Return one logit tensor per auxiliary head."""
        return [head(hidden) for head in self.heads]

    def loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        *,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Mean cross-entropy over all auxiliary heads.

        Args:
            hidden: ``(B, T, d_model)`` final hidden states.
            targets: ``(B, T)`` next-token targets (i.e. already shifted by one, so
                ``targets[:, i]`` is the token at position ``i+1``).
            ignore_index: Label to skip.

        Returns:
            A scalar. Head ``j`` (offset ``j+2``) is supervised against
            ``targets[:, j+1:]``, and positions with no target are dropped.
        """
        if self.n_heads == 0:
            return hidden.new_zeros(())
        losses: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = self(hidden)
        for j, logits in enumerate(all_logits):
            shift = j + 1  # head j predicts t+(j+2); targets holds t+1
            if targets.shape[1] <= shift:
                continue
            pred = logits[:, : targets.shape[1] - shift]
            tgt = targets[:, shift:]
            losses.append(
                F.cross_entropy(
                    pred.reshape(-1, pred.shape[-1]),
                    tgt.reshape(-1),
                    ignore_index=ignore_index,
                )
            )
        if not losses:
            return hidden.new_zeros(())
        return torch.stack(losses).mean()
