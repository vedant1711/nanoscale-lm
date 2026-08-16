"""Medusa-style self-speculation with tree attention (spec B8).

Reference: Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple
Decoding Heads* (arXiv:2401.10774).

The idea
--------
Draft–target speculation needs a *second model*: another set of weights to train, store,
load and keep in sync with the target's tokenizer. Medusa removes it. Extra lightweight
heads are attached to the target's own final hidden state, head ``k`` predicting the
token ``k+1`` steps ahead. Drafting costs one small matmul per head instead of ``γ``
forward passes of a second network.

This is where Arc 1 pays Arc 2 back: the multi-token-prediction heads implemented in
:mod:`nanoscale.model.mtp` as a *training* signal are exactly the drafting mechanism
here. One architecture decision, two uses.

Tree attention
--------------
A single head is a weak predictor, so Medusa takes the top-``k`` candidates from each
head and considers their **Cartesian product** as a tree of continuations. Verifying
those independently would cost one forward pass per path. Instead the paths are packed
into one sequence and given a **custom attention mask** in which each node attends only
to its own ancestors — so sibling branches cannot see each other, and one forward pass
over ``N`` tree nodes evaluates every root-to-leaf path simultaneously.

:func:`build_tree_attention_mask` constructs that mask from the tree's parent pointers,
and a test verifies the resulting logits match evaluating each path on its own.

**Positions matter as much as the mask.** A node's index in the packed sequence is *not*
its position in the sequence it belongs to: the third node of the packed tree might be
the first child of the second root, i.e. at depth 1. Feeding the packed sequence with
default positions gives every node a RoPE angle derived from its packing index, which
silently scores continuations at the wrong distance from the prefix.
:func:`tree_position_ids` supplies positions derived from each node's **depth**, and the
equivalence test only passes with them — it caught exactly this bug.

What is and is not claimed
---------------------------
Medusa's accept rule here is **exact-match against the target's own greedy/sampled
token**, which is the typical Medusa formulation and is *not* the distribution-preserving
rule used for draft–target. So: draft–target is lossless by construction, Medusa is a
heuristic that trades a little distributional fidelity for not needing a second model.
The tests hold each to its own standard rather than pretending they are the same thing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nanoscale.model import NanoScaleLM
from nanoscale.model.mtp import MultiTokenPredictionHeads
from nanoscale.specdec.spec_sampling import SpeculativeResult, apply_sampling_transforms
from nanoscale.utils.logging import get_logger

__all__ = [
    "MedusaSampler",
    "TreeCandidates",
    "build_candidate_tree",
    "build_tree_attention_mask",
    "tree_position_ids",
]

log = get_logger("nanoscale.specdec.medusa")


@dataclass(frozen=True, slots=True)
class TreeCandidates:
    """A packed candidate tree: tokens, parent pointers and root-to-leaf paths."""

    tokens: Tensor  # (N,) the token at each node, in breadth-first order
    parents: Tensor  # (N,) index of each node's parent, -1 for a root
    depths: Tensor  # (N,) distance from the root
    paths: list[list[int]]  # node indices along each root-to-leaf path

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the tree."""
        return int(self.tokens.shape[0])


def build_candidate_tree(
    head_topk: list[Tensor],
    *,
    max_nodes: int = 16,
) -> TreeCandidates:
    """Build a candidate tree from each head's top-k tokens.

    Args:
        head_topk: One ``(k,)`` tensor of candidate token IDs per head, ordered by
            decreasing probability. Head ``i`` predicts depth ``i``.
        max_nodes: Cap on total nodes, applied breadth-first so shallow, high-probability
            continuations are kept in preference to deep speculative ones.

    Returns:
        A :class:`TreeCandidates`.
    """
    if not head_topk:
        raise ValueError("build_candidate_tree needs at least one head's candidates.")

    tokens: list[int] = []
    parents: list[int] = []
    depths: list[int] = []
    frontier: list[int] = [-1]  # parent indices at the current depth; -1 is the root

    for depth, candidates in enumerate(head_topk):
        next_frontier: list[int] = []
        for parent in frontier:
            for token in candidates.tolist():
                if len(tokens) >= max_nodes:
                    break
                tokens.append(int(token))
                parents.append(parent)
                depths.append(depth)
                next_frontier.append(len(tokens) - 1)
            if len(tokens) >= max_nodes:
                break
        frontier = next_frontier
        if not frontier or len(tokens) >= max_nodes:
            break

    # Every node that is nobody's parent terminates a path.
    has_child = {p for p in parents if p >= 0}
    paths: list[list[int]] = []
    for node in range(len(tokens)):
        if node in has_child:
            continue
        path: list[int] = []
        cursor = node
        while cursor >= 0:
            path.append(cursor)
            cursor = parents[cursor]
        paths.append(list(reversed(path)))

    return TreeCandidates(
        tokens=torch.tensor(tokens, dtype=torch.long),
        parents=torch.tensor(parents, dtype=torch.long),
        depths=torch.tensor(depths, dtype=torch.long),
        paths=paths,
    )


def build_tree_attention_mask(
    tree: TreeCandidates,
    prefix_len: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Additive attention mask letting each tree node see only the prefix and its ancestors.

    Args:
        tree: The candidate tree.
        prefix_len: Number of already-committed positions the whole tree may attend to.
        device: Mask device.
        dtype: Mask dtype (the fill value is that dtype's most negative finite value).

    Returns:
        ``(1, 1, N, prefix_len + N)``, additive: ``0`` where attention is allowed.

    Without this mask, packing the tree into one sequence would let sibling branches
    attend to each other and the verification would be scoring continuations that no
    single path actually contains.
    """
    n = tree.n_nodes
    neg = torch.finfo(dtype).min
    mask = torch.full((n, prefix_len + n), neg, device=device, dtype=dtype)
    mask[:, :prefix_len] = 0.0  # every node sees the committed prefix

    parents = tree.parents.tolist()
    for node in range(n):
        mask[node, prefix_len + node] = 0.0  # itself
        cursor = parents[node]
        while cursor >= 0:
            mask[node, prefix_len + cursor] = 0.0
            cursor = parents[cursor]
    return mask.view(1, 1, n, prefix_len + n)


def tree_position_ids(
    tree: TreeCandidates, prefix_len: int, *, device: torch.device | None = None
) -> Tensor:
    """Absolute positions for a packed ``[prefix, tree]`` sequence.

    Prefix positions are ``0 … prefix_len-1``; a tree node at depth ``d`` gets
    ``prefix_len + d``, regardless of where it sits in the packed order. Without this,
    RoPE would place sibling branches at different distances from the prefix purely
    because of packing order.
    """
    prefix = torch.arange(prefix_len, device=device)
    nodes = tree.depths.to(device=device) + prefix_len
    return torch.cat([prefix, nodes])


class MedusaSampler:
    """Self-speculative decoding using multi-token-prediction heads as the drafter."""

    def __init__(
        self,
        model: NanoScaleLM,
        *,
        heads: MultiTokenPredictionHeads | None = None,
        topk: int = 3,
        max_nodes: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        """Attach a Medusa drafter to a model.

        Args:
            model: The target model.
            heads: MTP heads to draft with. Defaults to the model's own ``mtp`` heads —
                the Arc-1/Arc-2 seam — and raises if the model has none.
            topk: Candidates taken from each head.
            max_nodes: Cap on tree size.
            temperature: Sampling temperature for the target's own tokens.
            top_p: Nucleus cutoff.
        """
        self.model = model.eval()
        resolved = heads if heads is not None else model.mtp
        if resolved is None or resolved.n_heads == 0:
            raise ValueError(
                "Medusa needs multi-token-prediction heads. Train the model with "
                "model.n_mtp_heads > 0, or pass heads= explicitly."
            )
        self.heads: MultiTokenPredictionHeads = resolved
        self.topk = topk
        self.max_nodes = max_nodes
        self.temperature = temperature
        self.top_p = top_p

    def _probs(self, logits: Tensor) -> Tensor:
        return apply_sampling_transforms(logits, temperature=self.temperature, top_p=self.top_p)

    @torch.no_grad()
    def propose(self, hidden: Tensor) -> TreeCandidates:
        """Draft a candidate tree from one final hidden state ``(1, d_model)``."""
        head_topk: list[Tensor] = []
        for module in self.heads.heads:
            head = module
            assert isinstance(head, nn.Module)
            logits = head(hidden)
            k = min(self.topk, logits.shape[-1])
            head_topk.append(torch.topk(logits, k, dim=-1).indices.reshape(-1)[:k])
        return build_candidate_tree(head_topk, max_nodes=self.max_nodes)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 64,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SpeculativeResult:
        """Generate with Medusa-style self-speculation.

        The acceptance rule is exact match: a drafted token is accepted iff it equals the
        token the target itself would have emitted at that position. That is Medusa's
        formulation and is *not* distribution-preserving; see the module docstring.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("Medusa decoding here is single-sequence.")

        prompt_len = input_ids.shape[1]
        tokens = input_ids
        model_calls = 0
        accepted_total = proposed_total = 0
        per_round: list[int] = []
        start = time.perf_counter()
        finished = False

        while tokens.shape[1] - prompt_len < max_new_tokens and not finished:
            if tokens.shape[1] >= self.model.config.max_seq_len:
                break

            out = self.model(tokens, return_hidden=True)
            model_calls += 1
            assert out.hidden is not None
            hidden = out.hidden[:, -1]

            # The target's own next token always comes first and is never speculative.
            next_token = torch.multinomial(
                self._probs(out.logits[:, -1]), 1, generator=generator
            ).squeeze(-1)
            emitted = [next_token]

            tree = self.propose(hidden)
            if tree.n_nodes and tokens.shape[1] + 1 + tree.n_nodes <= self.model.config.max_seq_len:
                committed = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
                prefix_len = committed.shape[1]
                node_tokens = tree.tokens.to(tokens.device).view(1, -1)
                packed = torch.cat([committed, node_tokens], dim=1)
                mask = build_tree_attention_mask(
                    tree, prefix_len, device=tokens.device, dtype=torch.float32
                )
                full_mask = torch.zeros(
                    1,
                    1,
                    packed.shape[1],
                    packed.shape[1],
                    dtype=torch.float32,
                    device=tokens.device,
                )
                full_mask[:, :, prefix_len:, :] = mask
                verify = self.model(packed, attn_mask=full_mask).logits
                model_calls += 1

                # Walk the best path greedily: accept a node iff its token matches what
                # the target would have produced at its parent position.
                best = max(tree.paths, key=len) if tree.paths else []
                n_accepted = 0
                position = prefix_len - 1
                for node in best:
                    target_token = int(verify[0, position].argmax())
                    if target_token != int(tree.tokens[node]):
                        break
                    emitted.append(tree.tokens[node : node + 1].to(tokens.device))
                    n_accepted += 1
                    position = prefix_len + node
                proposed_total += len(best)
                accepted_total += n_accepted
                per_round.append(n_accepted)

            new_tokens = torch.cat([t.reshape(1, 1) for t in emitted], dim=1)
            tokens = torch.cat([tokens, new_tokens], dim=1)
            if eos_id is not None and bool((new_tokens == eos_id).any()):
                finished = True

        wall = time.perf_counter() - start
        tokens = tokens[:, : prompt_len + max_new_tokens]
        return SpeculativeResult(
            tokens=tokens,
            generated=tokens.shape[1] - prompt_len,
            target_calls=model_calls,
            draft_calls=0,
            accepted_tokens=accepted_total,
            proposed_tokens=proposed_total,
            wall_clock_s=wall,
            per_round_accepted=per_round,
        )
