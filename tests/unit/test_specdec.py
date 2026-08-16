"""Speculative-decoding correctness (spec D1) — the most important tests in the repo.

The claim under test is not "it is fast" but "it is **exactly** the target distribution".
A speculative decoder with a subtly wrong acceptance rule still produces fluent text; the
only way to catch it is to measure the emitted distribution against direct sampling from
the target, which is what :func:`test_accepted_tokens_match_target_sampling` does over
120k samples with a total-variation bound.

Three deliberately different levels of evidence:

1. **Algebraic** — the branch probabilities sum to ``p(x)`` exactly, checked in fp64.
2. **Statistical** — the empirical distribution over 120k samples matches, and a
   deliberately *broken* rule fails the same test (so the test has teeth).
3. **End-to-end** — greedy speculative decoding produces token-for-token identical
   output to greedy autoregressive decoding on a real model.
"""

from __future__ import annotations

import math

import pytest
import torch

from nanoscale.config import ModelConfig, draft_model_config
from nanoscale.model import build_model
from nanoscale.model.attention import CausalSelfAttention
from nanoscale.specdec import (
    MedusaSampler,
    SpeculativeSampler,
    acceptance_probability,
    apply_sampling_transforms,
    autoregressive_baseline,
    build_candidate_tree,
    build_tree_attention_mask,
    expected_acceptance_rate,
    residual_distribution,
    sample_accept_reject,
    tree_position_ids,
)

VOCAB = 12
N_SAMPLES = 120_000
#: Total-variation tolerance. With N samples the per-bin standard error is about
#: sqrt(p(1-p)/N) ~ 1.4e-3 here, so a 1.5e-2 TV budget across 12 bins is roughly 4 sigma:
#: loose enough not to flake, tight enough that a wrong rule cannot slip through (the
#: broken-rule control test below confirms it does not).
TV_TOLERANCE = 1.5e-2


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(20260816)


def random_distribution(n: int = VOCAB, *, temperature: float = 1.0) -> torch.Tensor:
    return torch.softmax(torch.randn(n, dtype=torch.float64) / temperature, dim=-1)


def total_variation(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(0.5 * (a - b).abs().sum())


def empirical(samples: torch.Tensor, n: int = VOCAB) -> torch.Tensor:
    return torch.bincount(samples, minlength=n).double() / samples.numel()


# =====================================================================================
# 1. Algebraic: the two branches sum to p(x)
# =====================================================================================


def test_the_two_branches_sum_to_the_target_probability() -> None:
    r"""``min(q, p) + max(0, p - q) == p``, elementwise, exactly.

    This is the whole proof. If it holds, the emitted distribution is ``p``.
    """
    for _ in range(20):
        p = random_distribution()
        q = random_distribution()
        accept_mass = torch.minimum(p, q)  # q(x) * min(1, p/q)
        rejection_prob = 1.0 - accept_mass.sum()
        resample_mass = residual_distribution(p, q) * rejection_prob
        torch.testing.assert_close(accept_mass + resample_mass, p, atol=1e-12, rtol=1e-12)


def test_residual_is_a_valid_distribution() -> None:
    for _ in range(20):
        p, q = random_distribution(), random_distribution()
        residual = residual_distribution(p, q)
        assert float(residual.sum()) == pytest.approx(1.0, abs=1e-10)
        assert bool((residual >= 0).all())


def test_residual_falls_back_when_the_distributions_are_identical() -> None:
    p = random_distribution()
    torch.testing.assert_close(residual_distribution(p, p), p, atol=1e-10, rtol=1e-10)


def test_acceptance_probability_is_min_one_p_over_q() -> None:
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.25, 0.6, 0.15]])
    tokens = torch.tensor([0, 1, 2])
    got = torch.stack([acceptance_probability(p, q, tokens[i : i + 1]) for i in range(3)])
    expected = torch.tensor([[1.0], [0.5], [1.0]])  # 0.5/0.25 clamped, 0.3/0.6, 0.2/0.15 clamped
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_expected_acceptance_rate_is_one_minus_total_variation() -> None:
    for _ in range(10):
        p, q = random_distribution(), random_distribution()
        assert float(expected_acceptance_rate(p, q)) == pytest.approx(
            1.0 - total_variation(p, q), abs=1e-10
        )


def test_identical_distributions_always_accept() -> None:
    p = random_distribution()
    assert float(expected_acceptance_rate(p, p)) == pytest.approx(1.0, abs=1e-10)


# =====================================================================================
# 2. Statistical: the distributional-equivalence test
# =====================================================================================


def test_accepted_tokens_match_target_sampling() -> None:
    """The correctness crown jewel (spec D1).

    Draw N tokens through the full accept/reject procedure and N directly from ``p``.
    The two empirical distributions must agree within the sampling tolerance.
    """
    p = random_distribution(temperature=1.2)
    q = random_distribution(temperature=1.2)
    gen = torch.Generator().manual_seed(4242)

    draft_tokens = torch.multinomial(
        q.expand(N_SAMPLES, VOCAB), 1, replacement=True, generator=gen
    ).squeeze(-1)
    emitted, accepted = sample_accept_reject(
        p.expand(N_SAMPLES, VOCAB).contiguous(),
        q.expand(N_SAMPLES, VOCAB).contiguous(),
        draft_tokens,
        generator=gen,
    )

    direct = torch.multinomial(
        p.expand(N_SAMPLES, VOCAB), 1, replacement=True, generator=gen
    ).squeeze(-1)

    tv = total_variation(empirical(emitted), empirical(direct))
    assert tv < TV_TOLERANCE, f"speculative TV {tv:.5f} exceeds tolerance {TV_TOLERANCE}"

    # The observed acceptance rate must match the analytic sum(min(p, q)).
    observed = float(accepted.double().mean())
    assert observed == pytest.approx(float(expected_acceptance_rate(p, q)), abs=0.01)


def test_the_equivalence_test_rejects_a_deliberately_broken_rule() -> None:
    """A control: if the test cannot fail, it proves nothing.

    The classic wrong implementation resamples from ``p`` instead of the residual
    ``(p - q)₊``. It is *almost* right, produces perfectly fluent text, and biases the
    output toward tokens the draft already favours. The same statistical test must catch
    it.
    """
    p = random_distribution(temperature=1.2)
    q = random_distribution(temperature=1.2)
    gen = torch.Generator().manual_seed(99)

    draft_tokens = torch.multinomial(
        q.expand(N_SAMPLES, VOCAB), 1, replacement=True, generator=gen
    ).squeeze(-1)
    alpha = acceptance_probability(
        p.expand(N_SAMPLES, VOCAB), q.expand(N_SAMPLES, VOCAB), draft_tokens
    )
    accepted = torch.rand(N_SAMPLES, generator=gen, dtype=torch.float64) < alpha
    # The bug: resample from p rather than from the normalised residual.
    wrong_replacement = torch.multinomial(
        p.expand(N_SAMPLES, VOCAB), 1, replacement=True, generator=gen
    ).squeeze(-1)
    broken = torch.where(accepted, draft_tokens, wrong_replacement)

    direct = torch.multinomial(
        p.expand(N_SAMPLES, VOCAB), 1, replacement=True, generator=gen
    ).squeeze(-1)
    tv = total_variation(empirical(broken), empirical(direct))
    assert tv > TV_TOLERANCE, (
        f"the broken rule produced TV {tv:.5f}, which the test would have accepted — "
        "the tolerance is too loose to be meaningful"
    )


def test_equivalence_holds_even_with_a_terrible_draft() -> None:
    """Losslessness does not depend on draft quality; only the speedup does."""
    p = random_distribution()
    q = torch.full((VOCAB,), 1.0 / VOCAB, dtype=torch.float64)  # uniform: a useless draft
    gen = torch.Generator().manual_seed(7)

    draft_tokens = torch.multinomial(
        q.expand(N_SAMPLES, VOCAB), 1, replacement=True, generator=gen
    ).squeeze(-1)
    emitted, accepted = sample_accept_reject(
        p.expand(N_SAMPLES, VOCAB).contiguous(),
        q.expand(N_SAMPLES, VOCAB).contiguous(),
        draft_tokens,
        generator=gen,
    )
    direct = torch.multinomial(
        p.expand(N_SAMPLES, VOCAB), 1, replacement=True, generator=gen
    ).squeeze(-1)

    assert total_variation(empirical(emitted), empirical(direct)) < TV_TOLERANCE
    # A useless draft still emits the right distribution, it is just rarely accepted.
    assert float(accepted.double().mean()) < 0.95


def test_equivalence_holds_when_the_draft_equals_the_target() -> None:
    p = random_distribution()
    gen = torch.Generator().manual_seed(11)
    draft_tokens = torch.multinomial(
        p.expand(20_000, VOCAB), 1, replacement=True, generator=gen
    ).squeeze(-1)
    emitted, accepted = sample_accept_reject(
        p.expand(20_000, VOCAB).contiguous(),
        p.expand(20_000, VOCAB).contiguous(),
        draft_tokens,
        generator=gen,
    )
    assert bool(accepted.all()), "an identical draft must always be accepted"
    assert torch.equal(emitted, draft_tokens)


# =====================================================================================
# 3. End-to-end on a real model
# =====================================================================================


def make_models() -> tuple[object, object]:
    cfg = ModelConfig(
        vocab_size=64,
        n_layers=3,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=96,
        zero_init_output=False,
    )
    torch.manual_seed(3)
    target = build_model(cfg)
    draft = build_model(draft_model_config(cfg))
    return target, draft


def test_greedy_speculative_equals_greedy_autoregressive() -> None:
    """At temperature 0 both are deterministic, so equality must be token-for-token.

    This is the sharpest end-to-end check available: any cache off-by-one, any wrong
    position in the verification pass, changes the output immediately.
    """
    target, draft = make_models()
    prompt = torch.randint(0, 64, (1, 5))

    spec = SpeculativeSampler(target, draft, gamma=4, temperature=0.0)  # type: ignore[arg-type]
    speculative = spec.generate(prompt, max_new_tokens=24)
    baseline = autoregressive_baseline(target, prompt, max_new_tokens=24, temperature=0.0)  # type: ignore[arg-type]

    assert torch.equal(speculative.tokens, baseline.tokens), (
        "greedy speculative decoding diverged from greedy autoregressive decoding"
    )


def test_speculative_uses_far_fewer_target_calls() -> None:
    """The efficiency claim, stated in the currency that actually matters."""
    target, draft = make_models()
    prompt = torch.randint(0, 64, (1, 4))
    spec = SpeculativeSampler(target, draft, gamma=4, temperature=1.0)  # type: ignore[arg-type]
    result = spec.generate(prompt, max_new_tokens=32, generator=torch.Generator().manual_seed(1))

    assert result.generated == 32
    assert result.target_calls < 32, "speculation did not reduce target forward passes"
    assert result.mean_accepted_length > 1.0
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_greedy_speculation_with_a_disagreeing_draft_is_a_net_loss() -> None:
    """An honest negative case, pinned rather than hidden.

    At temperature 0 both distributions are one-hot. If the draft's argmax differs from
    the target's, ``p(x_draft) = 0`` and the token is rejected with certainty — so every
    round costs a full target pass and yields exactly one token, plus the drafting work
    on top. Speculation only pays when the draft *agrees often*, which for an untrained
    draft at temperature 0 it does not. This is a property of the method, not a bug:
    correctness is unconditional, speedup is not.
    """
    target, draft = make_models()
    prompt = torch.randint(0, 64, (1, 4))
    spec = SpeculativeSampler(target, draft, gamma=4, temperature=0.0)  # type: ignore[arg-type]
    result = spec.generate(prompt, max_new_tokens=16)

    assert result.acceptance_rate == 0.0
    assert result.target_calls >= result.generated
    # ...and the output is still exactly right, which is the point.
    baseline = autoregressive_baseline(target, prompt, max_new_tokens=16, temperature=0.0)  # type: ignore[arg-type]
    assert torch.equal(result.tokens, baseline.tokens)


def test_a_self_draft_accepts_everything() -> None:
    """Drafting with the target itself must accept every token: q == p exactly."""
    target, _ = make_models()
    spec = SpeculativeSampler(target, target, gamma=4, temperature=1.0)  # type: ignore[arg-type]
    result = spec.generate(torch.randint(0, 64, (1, 4)), max_new_tokens=20)
    assert result.acceptance_rate == pytest.approx(1.0, abs=1e-9)
    assert all(n == 4 for n in result.per_round_accepted), result.per_round_accepted
    # Each verification round yields gamma+1 = 5 tokens; the prompt prefill is one extra
    # target call, so the average over all calls is 20/5 = 4.0 rather than 5.0.
    assert result.mean_accepted_length >= 4.0


def test_speculative_generation_is_reproducible_under_a_seed() -> None:
    target, draft = make_models()
    prompt = torch.randint(0, 64, (1, 4))
    spec = SpeculativeSampler(target, draft, gamma=3, temperature=1.0)  # type: ignore[arg-type]
    a = spec.generate(prompt, max_new_tokens=16, generator=torch.Generator().manual_seed(5))
    b = spec.generate(prompt, max_new_tokens=16, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a.tokens, b.tokens)


def test_speculative_respects_the_token_budget() -> None:
    target, draft = make_models()
    spec = SpeculativeSampler(target, draft, gamma=5, temperature=1.0)  # type: ignore[arg-type]
    for n in (1, 3, 7, 16):
        result = spec.generate(torch.randint(0, 64, (1, 3)), max_new_tokens=n)
        assert result.generated == n, f"asked for {n}, got {result.generated}"


def test_speculative_validates_its_inputs() -> None:
    target, draft = make_models()
    with pytest.raises(ValueError, match="single-sequence"):
        SpeculativeSampler(target, draft).generate(torch.zeros(2, 4, dtype=torch.long))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gamma must be at least 1"):
        SpeculativeSampler(target, draft, gamma=0)  # type: ignore[arg-type]

    other_vocab = ModelConfig(
        vocab_size=32, n_layers=1, d_model=32, n_heads=2, n_kv_heads=1, max_seq_len=32
    )
    with pytest.raises(ValueError, match="must share a tokenizer"):
        SpeculativeSampler(target, build_model(other_vocab))  # type: ignore[arg-type]


def test_sampling_transforms_are_shared_between_the_models() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, -5.0]])
    greedy = apply_sampling_transforms(logits, temperature=0.0)
    assert float(greedy[0, 0]) == 1.0 and float(greedy.sum()) == 1.0

    tempered = apply_sampling_transforms(logits, temperature=2.0)
    assert float(tempered.sum()) == pytest.approx(1.0)
    assert float(tempered[0, 0]) < float(apply_sampling_transforms(logits, temperature=0.5)[0, 0])

    nucleus = apply_sampling_transforms(logits, temperature=1.0, top_p=0.5)
    assert float(nucleus[0, -1]) == 0.0
    assert float(nucleus.sum()) == pytest.approx(1.0)


# =====================================================================================
# Medusa and tree attention
# =====================================================================================


def test_candidate_tree_structure() -> None:
    heads = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    tree = build_candidate_tree(heads, max_nodes=16)
    # depth 0: 2 roots; depth 1: 2 children each = 4; total 6
    assert tree.n_nodes == 6
    assert tree.parents[:2].tolist() == [-1, -1]
    assert set(tree.parents[2:].tolist()) == {0, 1}
    assert len(tree.paths) == 4  # four leaves
    assert all(len(path) == 2 for path in tree.paths)


def test_candidate_tree_respects_the_node_cap() -> None:
    heads = [torch.arange(5), torch.arange(5), torch.arange(5)]
    tree = build_candidate_tree(heads, max_nodes=7)
    assert tree.n_nodes <= 7


def test_candidate_tree_rejects_no_heads() -> None:
    with pytest.raises(ValueError, match="at least one head"):
        build_candidate_tree([])


def test_tree_mask_lets_each_node_see_only_its_ancestors() -> None:
    heads = [torch.tensor([1, 2]), torch.tensor([3])]
    tree = build_candidate_tree(heads, max_nodes=16)
    prefix = 3
    mask = build_tree_attention_mask(tree, prefix)[0, 0]
    allowed = mask == 0.0

    assert bool(allowed[:, :prefix].all()), "every node must see the committed prefix"
    for node in range(tree.n_nodes):
        assert bool(allowed[node, prefix + node]), "a node must see itself"
        ancestors = set()
        cursor = int(tree.parents[node])
        while cursor >= 0:
            ancestors.add(cursor)
            cursor = int(tree.parents[cursor])
        for other in range(tree.n_nodes):
            expected = other == node or other in ancestors
            assert bool(allowed[node, prefix + other]) is expected, (
                f"node {node} visibility of {other} should be {expected}"
            )


def test_tree_attention_matches_evaluating_each_path_separately() -> None:
    """One packed forward pass must equal one forward pass per root-to-leaf path."""
    cfg = ModelConfig(
        vocab_size=32,
        n_layers=2,
        d_model=32,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=64,
        zero_init_output=False,
    )
    torch.manual_seed(1)
    model = build_model(cfg).double().eval()

    prefix = torch.randint(0, 32, (1, 5))
    heads = [torch.tensor([7, 11]), torch.tensor([13])]
    tree = build_candidate_tree(heads, max_nodes=16)

    packed = torch.cat([prefix, tree.tokens.view(1, -1)], dim=1)
    prefix_len = prefix.shape[1]
    node_mask = build_tree_attention_mask(tree, prefix_len, dtype=torch.float64)
    full = torch.zeros(1, 1, packed.shape[1], packed.shape[1], dtype=torch.float64)
    full[:, :, :prefix_len, :] = CausalSelfAttention.build_causal_mask(
        prefix_len, packed.shape[1], device=packed.device, dtype=torch.float64
    )
    full[:, :, prefix_len:, :] = node_mask
    # Positions come from tree *depth*, not packing order: see tree_position_ids.
    positions = tree_position_ids(tree, prefix_len)
    packed_logits = model(packed, attn_mask=full, positions=positions).logits

    for path in tree.paths:
        path_tokens = torch.tensor([[int(tree.tokens[n]) for n in path]])
        sequence = torch.cat([prefix, path_tokens], dim=1)
        separate = model(sequence).logits
        for depth, node in enumerate(path):
            torch.testing.assert_close(
                packed_logits[0, prefix_len + node],
                separate[0, prefix_len + depth],
                atol=1e-9,
                rtol=1e-9,
            )


def test_medusa_requires_mtp_heads() -> None:
    cfg = ModelConfig(
        vocab_size=32, n_layers=1, d_model=32, n_heads=2, n_kv_heads=1, max_seq_len=32
    )
    with pytest.raises(ValueError, match="multi-token-prediction heads"):
        MedusaSampler(build_model(cfg))


def test_medusa_generates_with_the_models_own_mtp_heads() -> None:
    """The Arc-1/Arc-2 seam: heads trained as an auxiliary objective become the drafter."""
    cfg = ModelConfig(
        vocab_size=32,
        n_layers=2,
        d_model=32,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=96,
        n_mtp_heads=2,
        zero_init_output=False,
    )
    torch.manual_seed(2)
    model = build_model(cfg)
    sampler = MedusaSampler(model, topk=2, max_nodes=6, temperature=0.0)
    result = sampler.generate(torch.randint(0, 32, (1, 4)), max_new_tokens=12)
    assert result.generated == 12
    assert result.tokens.shape == (1, 16)
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_medusa_is_reproducible() -> None:
    cfg = ModelConfig(
        vocab_size=32,
        n_layers=2,
        d_model=32,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=96,
        n_mtp_heads=2,
        zero_init_output=False,
    )
    torch.manual_seed(2)
    model = build_model(cfg)
    sampler = MedusaSampler(model, topk=2, max_nodes=6, temperature=1.0)
    prompt = torch.randint(0, 32, (1, 4))
    a = sampler.generate(prompt, max_new_tokens=8, generator=torch.Generator().manual_seed(0))
    b = sampler.generate(prompt, max_new_tokens=8, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a.tokens, b.tokens)


# =====================================================================================
# Composition with quantization (spec Phase 9 acceptance)
# =====================================================================================


def test_speculation_composes_with_a_quantized_target() -> None:
    """The two levers stack: quantize the target, then speculate over it.

    Losslessness is *relative to the target you have*. Quantizing changes the target
    distribution, and speculative decoding then reproduces that new distribution exactly.
    The test asserts exactly that, rather than the stronger and false claim that the
    output is unchanged by quantization.
    """
    from nanoscale.quantize import quantize_rtn

    target, draft = make_models()
    quantize_rtn(target, bits=4, group_size=32)  # type: ignore[arg-type]

    prompt = torch.randint(0, 64, (1, 4))
    spec = SpeculativeSampler(target, draft, gamma=4, temperature=0.0)  # type: ignore[arg-type]
    speculative = spec.generate(prompt, max_new_tokens=20)
    baseline = autoregressive_baseline(target, prompt, max_new_tokens=20, temperature=0.0)  # type: ignore[arg-type]

    assert torch.equal(speculative.tokens, baseline.tokens)
    assert speculative.target_calls < baseline.target_calls


def test_mean_accepted_length_bounds_the_speedup() -> None:
    """Sanity on the reported metric: it can never exceed gamma + 1."""
    target, draft = make_models()
    gamma = 4
    spec = SpeculativeSampler(target, draft, gamma=gamma, temperature=1.0)  # type: ignore[arg-type]
    result = spec.generate(
        torch.randint(0, 64, (1, 4)),
        max_new_tokens=24,
        generator=torch.Generator().manual_seed(1),
    )
    assert result.mean_accepted_length <= gamma + 1 + 1e-9
    assert all(0 <= n <= gamma for n in result.per_round_accepted)
    assert math.isfinite(result.tokens_per_second)
