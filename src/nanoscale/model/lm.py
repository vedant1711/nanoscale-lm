"""The NanoScale-LM decoder-only language model (spec B2).

Architecture: token embedding → N pre-norm blocks (GQA + RoPE + QK-norm attention,
SwiGLU MLP) → final RMSNorm → untied LM head, with optional tanh logit soft-capping and
optional multi-token-prediction heads.

Initialisation
--------------
Two schemes, both seeded and both documented:

* **Zero-init output projections** (default, muP-like, from the modded-nanoGPT
  speedrun). Every projection that *writes into the residual stream* — attention
  ``o_proj``, MLP ``down_proj`` — starts at exactly zero, as does the LM head. The
  network therefore starts as the identity function on the residual stream and outputs
  uniform logits, so the initial loss is exactly ``ln(vocab_size)``. Layers switch
  themselves on as their gradients justify it, rather than starting with a random
  perturbation that has to be unlearned first. A test asserts the ``ln(V)`` starting
  loss, which is a sharp check that nothing is silently non-zero.

  This interacts with weight tying: a **tied** head *is* the embedding matrix, so
  zeroing it would destroy the input embeddings too. Tied models therefore keep a
  randomly-initialised head and do not start at exactly ``ln(V)``. That is one more
  reason the default here is an untied head, as in the speedrun stack.
* **Scaled residual init** (the ablation): those same projections get
  ``std / √(2·n_layers)``, the GPT-2 recipe for keeping residual-stream variance from
  growing with depth.

Logit soft-capping
------------------
Optional ``c · tanh(logits / c)`` (Gemma-2). It bounds logits smoothly instead of
letting a few dimensions run away, which is a cheap guard against the loss spikes small
models are prone to. Off by default so its effect can be measured, not assumed.
"""

from __future__ import annotations

import math
from typing import NamedTuple, cast

import torch
from torch import nn
from torch.nn import functional as F

from nanoscale.config import ModelConfig
from nanoscale.model.block import TransformerBlock
from nanoscale.model.kv_cache import KVCache
from nanoscale.model.mtp import MTPHead, MultiTokenPredictionHeads
from nanoscale.model.norm import build_norm
from nanoscale.model.rope import RotaryCache

__all__ = ["IGNORE_INDEX", "LMOutput", "NanoScaleLM", "build_model"]

#: Label value skipped by the loss, matching PyTorch's cross-entropy default.
IGNORE_INDEX = -100


class LMOutput(NamedTuple):
    """Result of a forward pass."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None


class NanoScaleLM(nn.Module):
    """A modern decoder-only transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the model described by ``config`` and initialise its weights."""
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
        )
        self.final_norm = build_norm(config.norm_type, config.d_model, eps=config.norm_eps)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.mtp: MultiTokenPredictionHeads | None = (
            MultiTokenPredictionHeads(
                config.d_model, config.vocab_size, config.n_mtp_heads, norm_eps=config.norm_eps
            )
            if config.n_mtp_heads > 0
            else None
        )

        self.embed_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.rope = RotaryCache(
            config.head_dim,
            config.max_seq_len,
            theta=config.rope_theta,
            scaling=config.rope_scaling,
        )

        self.apply(self._init_weights)
        self._apply_output_init()

    # ------------------------------------------------------------------ init

    def _init_weights(self, module: nn.Module) -> None:
        """Truncated-normal init for every weight matrix and embedding."""
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def _apply_output_init(self) -> None:
        """Zero-init (or depth-scale) the projections that write to the residual stream."""
        cfg = self.config
        residual_writers: list[nn.Linear] = []
        for module in self.blocks:
            block = cast(TransformerBlock, module)
            residual_writers.append(block.attn.output_projection)
            residual_writers.append(cast(nn.Linear, block.mlp.output_projection))

        if cfg.zero_init_output:
            for linear in residual_writers:
                nn.init.zeros_(linear.weight)
            if not cfg.tie_embeddings:
                nn.init.zeros_(self.lm_head.weight)
            if self.mtp is not None:
                for module in self.mtp.heads:
                    head = cast(MTPHead, module)
                    nn.init.zeros_(head.head.weight)
        else:
            scale = cfg.init_std / math.sqrt(2 * cfg.n_layers)
            for linear in residual_writers:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=scale, a=-3 * scale, b=3 * scale)

    # ------------------------------------------------------------ introspection

    def num_parameters(self, *, non_embedding: bool = False) -> int:
        """Count parameters, optionally excluding the embedding and untied head."""
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return total
        total -= self.embed_tokens.weight.numel()
        if not self.config.tie_embeddings:
            total -= self.lm_head.weight.numel()
        return total

    @property
    def device(self) -> torch.device:
        """Device the parameters live on."""
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the parameters."""
        return self.embed_tokens.weight.dtype

    # ------------------------------------------------------------------ caching

    def make_cache(
        self,
        batch_size: int,
        *,
        max_seq_len: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Allocate a KV cache sized for this model."""
        return KVCache(
            n_layers=self.config.n_layers,
            batch_size=batch_size,
            n_kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            max_seq_len=max_seq_len or self.config.max_seq_len,
            device=self.device,
            dtype=dtype or self.dtype,
        )

    # ------------------------------------------------------------------ forward

    def _soft_cap(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Gemma-2-style ``c·tanh(logits/c)`` when configured."""
        cap = self.config.logit_soft_cap
        if cap is None:
            return logits
        return cap * torch.tanh(logits / cap)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        positions: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        loss_mask: torch.Tensor | None = None,
    ) -> LMOutput:
        """Run the model.

        Args:
            input_ids: ``(B, T)`` token IDs.
            targets: ``(B, T)`` next-token targets. Positions equal to
                :data:`IGNORE_INDEX` are skipped by the loss.
            cache: KV cache for incremental decoding. When supplied, ``positions``
                defaults to continuing from the cache length.
            positions: ``(T,)`` or ``(B, T)`` absolute positions for RoPE. Defaults to
                ``arange(T)`` offset by the cache length.
            attn_mask: Additive attention mask broadcastable to ``(B, H, T, T_kv)``.
            return_hidden: Also return the final hidden states (used by distillation
                and by the speculative-decoding drafters).
            loss_mask: ``(B, T)`` 0/1 mask restricting which positions contribute to the
                loss. Equivalent to setting targets to ``IGNORE_INDEX``, but keeps the
                caller's target tensor intact.

        Returns:
            An :class:`LMOutput`.
        """
        t = input_ids.shape[1]
        if t > self.config.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.config.max_seq_len}.")

        offset = cache.length if cache is not None else 0
        if positions is None:
            positions = torch.arange(offset, offset + t, device=input_ids.device)
        cos, sin = self.rope.get(positions)

        x = self.embed_dropout(self.embed_tokens(input_ids))
        for i, module in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x = cast(TransformerBlock, module)(x, cos, sin, cache=layer_cache, attn_mask=attn_mask)
        hidden = self.final_norm(x)

        logits = self._soft_cap(self.lm_head(hidden))

        loss: torch.Tensor | None = None
        mtp_loss: torch.Tensor | None = None
        if targets is not None:
            effective = targets
            if loss_mask is not None:
                effective = targets.masked_fill(loss_mask == 0, IGNORE_INDEX)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                effective.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            if self.mtp is not None:
                mtp_loss = self.mtp.loss(hidden, effective, ignore_index=IGNORE_INDEX)
                loss = loss + self.config.mtp_loss_weight * mtp_loss

        return LMOutput(
            logits=logits,
            loss=loss,
            hidden=hidden if return_hidden else None,
            mtp_loss=mtp_loss,
        )

    # --------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_id: int | None = None,
        use_cache: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample a continuation.

        This is the reference decoding loop: Phase 9's speculative decoder and Phase
        10's server are both measured against it, so it stays deliberately simple.

        Args:
            input_ids: ``(B, T)`` prompt.
            max_new_tokens: Tokens to generate.
            temperature: ``0`` means greedy.
            top_k: Keep only the ``k`` most likely tokens (``0`` disables).
            top_p: Nucleus cutoff (``1.0`` disables).
            eos_id: Stop once every sequence in the batch has emitted this token.
            use_cache: Use the KV cache. ``False`` recomputes the full prefix each step
                and must produce identical output — that equivalence is a test.
            generator: RNG for reproducible sampling.

        Returns:
            ``(B, T + generated)`` token IDs.
        """
        was_training = self.training
        self.eval()
        try:
            batch, prompt_len = input_ids.shape
            budget = min(self.config.max_seq_len, prompt_len + max_new_tokens)
            cache = self.make_cache(batch, max_seq_len=budget) if use_cache else None
            tokens = input_ids
            finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)

            step_input = tokens
            for _ in range(max_new_tokens):
                if tokens.shape[1] >= self.config.max_seq_len:
                    break
                out = self.forward(step_input if cache is not None else tokens, cache=cache)
                next_token = sample_next_token(
                    out.logits[:, -1],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                )
                if eos_id is not None:
                    next_token = torch.where(
                        finished, torch.full_like(next_token, eos_id), next_token
                    )
                    finished = finished | (next_token == eos_id)
                tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
                step_input = next_token.unsqueeze(1)
                if eos_id is not None and bool(finished.all()):
                    break
            return tokens
        finally:
            self.train(was_training)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row from ``(B, vocab)`` logits.

    Temperature is applied first, then top-k, then top-p, which is the conventional
    order: each filter narrows the already-tempered distribution.
    """
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled = logits.float() / temperature

    if top_k > 0:
        k = min(top_k, scaled.shape[-1])
        threshold = torch.topk(scaled, k, dim=-1).values[..., -1, None]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

    if top_p < 1.0:
        ordered, order = torch.sort(scaled, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        # Keep every token up to and including the one that crosses the threshold, so
        # the nucleus is never empty even when one token holds more than top_p mass.
        drop = cumulative - torch.softmax(ordered, dim=-1) >= top_p
        drop[..., 0] = False
        ordered = ordered.masked_fill(drop, float("-inf"))
        scaled = torch.empty_like(scaled).scatter_(-1, order, ordered)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def build_model(config: ModelConfig, *, device: torch.device | None = None) -> NanoScaleLM:
    """Construct a model and move it to ``device``.

    Asserts that the built module's parameter count matches the analytic count in
    :meth:`ModelConfig.param_breakdown`. The two are derived independently, so a
    mismatch means either the config table or the architecture drifted.
    """
    model = NanoScaleLM(config)
    expected = config.param_breakdown()
    actual = model.num_parameters()
    if actual != expected["total"]:
        raise AssertionError(
            f"parameter count mismatch: module has {actual:,}, config predicts "
            f"{expected['total']:,}. One of them is wrong."
        )
    if device is not None:
        model = model.to(device)
    return model
