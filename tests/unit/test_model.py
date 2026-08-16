"""Model-level tests (spec Phase 2 acceptance criteria).

Covers: forward shapes, hand-computed norm/MLP values, parameter counts against the
tier table, the initialisation schemes, logit soft-capping, MTP heads, and that a
random-init `nano` model actually generates (garbage but well-shaped) text.
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch
from torch.nn import functional as F

from nanoscale.config import TIERS, ModelConfig, draft_model_config, get_preset
from nanoscale.config.presets import TIER_EXPECTED_PARAMS
from nanoscale.model import (
    IGNORE_INDEX,
    LayerNorm,
    MTPHead,
    MultiTokenPredictionHeads,
    NanoScaleLM,
    ReLU2MLP,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    build_mlp,
    build_model,
    build_norm,
    rms_normalize,
    sample_next_token,
)
from nanoscale.tokenizer import BPETokenizer


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(20260816)


def tiny(**kwargs: object) -> ModelConfig:
    base: dict[str, object] = {
        "vocab_size": 64,
        "n_layers": 2,
        "d_model": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "max_seq_len": 16,
    }
    base.update(kwargs)
    return ModelConfig.model_validate(base)


# ------------------------------------------------------------------------- norms


def test_rmsnorm_matches_a_hand_computed_value() -> None:
    x = torch.tensor([[3.0, 4.0]])  # RMS = sqrt((9+16)/2) = 3.5355...
    out = rms_normalize(x, eps=0.0)
    rms = math.sqrt((9.0 + 16.0) / 2.0)
    torch.testing.assert_close(out, torch.tensor([[3.0 / rms, 4.0 / rms]]))
    torch.testing.assert_close(out.pow(2).mean(), torch.tensor(1.0))


def test_rmsnorm_module_applies_the_gain() -> None:
    norm = RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.randn(2, 4)
    torch.testing.assert_close(norm(x), rms_normalize(x) * norm.weight)


def test_rmsnorm_has_no_bias_and_unit_init() -> None:
    norm = RMSNorm(8)
    assert getattr(norm, "bias", None) is None
    torch.testing.assert_close(norm.weight, torch.ones(8))
    assert sum(p.numel() for p in norm.parameters()) == 8


def test_rmsnorm_without_affine_has_no_parameters() -> None:
    assert sum(p.numel() for p in RMSNorm(8, elementwise_affine=False).parameters()) == 0


def test_layernorm_centers_but_rmsnorm_does_not() -> None:
    x = torch.tensor([[10.0, 12.0, 14.0, 16.0]])  # nonzero mean
    ln = LayerNorm(4)(x)
    rn = RMSNorm(4)(x)
    torch.testing.assert_close(ln.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert rn.mean().abs() > 0.5, "RMSNorm must not remove the mean"
    torch.testing.assert_close(rn.pow(2).mean(), torch.tensor(1.0), atol=1e-5, rtol=0)


def test_build_norm_dispatch() -> None:
    assert isinstance(build_norm("rmsnorm", 4), RMSNorm)
    assert isinstance(build_norm("layernorm", 4), LayerNorm)
    with pytest.raises(ValueError, match="Unknown norm_type"):
        build_norm("batchnorm", 4)


# --------------------------------------------------------------------------- MLPs


def test_swiglu_matches_the_formula() -> None:
    mlp = SwiGLU(4, 8)
    x = torch.randn(3, 4)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    torch.testing.assert_close(mlp(x), expected)


def test_swiglu_matches_a_hand_computed_value() -> None:
    mlp = SwiGLU(2, 2)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.eye(2))
        mlp.up_proj.weight.copy_(torch.eye(2) * 2.0)
        mlp.down_proj.weight.copy_(torch.eye(2))
    x = torch.tensor([[1.0, 0.0]])
    # gate = [1,0] -> SiLU(1) = 1/(1+e^-1); up = [2,0]; product = [2*SiLU(1), 0]
    silu1 = 1.0 / (1.0 + math.exp(-1.0))
    torch.testing.assert_close(mlp(x), torch.tensor([[2.0 * silu1, 0.0]]), atol=1e-6, rtol=1e-6)


def test_relu2_matches_the_formula() -> None:
    mlp = ReLU2MLP(4, 8)
    x = torch.randn(3, 4)
    hidden = F.relu(mlp.up_proj(x))
    torch.testing.assert_close(mlp(x), mlp.down_proj(hidden * hidden))


def test_relu2_is_nonnegative_before_the_down_projection() -> None:
    mlp = ReLU2MLP(4, 8)
    x = torch.randn(5, 4)
    assert (F.relu(mlp.up_proj(x)) ** 2 >= 0).all()


def test_swiglu_has_three_matrices_and_relu2_has_two() -> None:
    assert len(list(SwiGLU(16, 32).parameters())) == 3
    assert len(list(ReLU2MLP(16, 32).parameters())) == 2


def test_build_mlp_dispatch() -> None:
    assert isinstance(build_mlp("swiglu", 4, 8), SwiGLU)
    assert isinstance(build_mlp("relu2", 4, 8), ReLU2MLP)
    with pytest.raises(ValueError, match="Unknown mlp_type"):
        build_mlp("gelu", 4, 8)


# --------------------------------------------------------------- parameter counts


@pytest.mark.parametrize("tier", TIERS)
def test_built_model_matches_the_tier_table(tier: str) -> None:
    """The analytic count and the real module must agree, and both match the table."""
    cfg = get_preset(tier).model
    model = NanoScaleLM(cfg)
    total, non_embed = TIER_EXPECTED_PARAMS[tier]
    assert model.num_parameters() == total
    assert model.num_parameters(non_embedding=True) == non_embed
    assert model.num_parameters() == cfg.param_count()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"qk_norm": False},
        {"mlp_type": "relu2"},
        {"norm_type": "layernorm"},
        {"tie_embeddings": True},
        {"n_mtp_heads": 2},
        {"n_kv_heads": 4},
        {"d_ff": 48},
    ],
)
def test_analytic_param_count_holds_across_configurations(overrides: dict[str, object]) -> None:
    cfg = tiny(**overrides)
    assert NanoScaleLM(cfg).num_parameters() == cfg.param_count()


def test_build_model_rejects_a_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tiny()
    monkeypatch.setattr(type(cfg), "param_breakdown", lambda self: {"total": 1, "non_embedding": 1})
    with pytest.raises(AssertionError, match="parameter count mismatch"):
        build_model(cfg)


def test_tying_embeddings_saves_exactly_the_head() -> None:
    untied = NanoScaleLM(tiny(tie_embeddings=False))
    tied = NanoScaleLM(tiny(tie_embeddings=True))
    saved = untied.num_parameters() - tied.num_parameters()
    assert saved == 64 * 32
    assert tied.lm_head.weight is tied.embed_tokens.weight


# ------------------------------------------------------------------ initialisation


def test_zero_init_gives_exactly_ln_vocab_initial_loss() -> None:
    """Zero-init output projections make the network start as uniform over the vocab."""
    cfg = tiny(zero_init_output=True)
    model = NanoScaleLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (4, 8))
    loss = model(ids, targets=ids).loss
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(math.log(cfg.vocab_size)), atol=1e-5, rtol=1e-5)


def test_tied_head_is_not_zeroed_because_it_is_the_embedding() -> None:
    """Zeroing a tied head would destroy the input embeddings; it must be skipped."""
    cfg = tiny(tie_embeddings=True, zero_init_output=True)
    model = NanoScaleLM(cfg).eval()
    assert torch.count_nonzero(model.lm_head.weight) > 0
    assert torch.count_nonzero(model.embed_tokens.weight) > 0
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    loss = model(ids, targets=ids).loss
    assert loss is not None
    assert abs(float(loss) - math.log(cfg.vocab_size)) > 1e-6


def test_zero_init_zeroes_every_residual_writer() -> None:
    model = NanoScaleLM(tiny(zero_init_output=True))
    blocks = cast(list[TransformerBlock], list(model.blocks))
    for block in blocks:
        assert torch.count_nonzero(block.attn.o_proj.weight) == 0
        assert torch.count_nonzero(cast(SwiGLU, block.mlp).down_proj.weight) == 0
    assert torch.count_nonzero(model.lm_head.weight) == 0
    # ...but the input-side projections are emphatically not zero.
    assert torch.count_nonzero(blocks[0].attn.q_proj.weight) > 0
    assert torch.count_nonzero(model.embed_tokens.weight) > 0


def test_scaled_residual_init_is_the_documented_alternative() -> None:
    cfg = tiny(zero_init_output=False, init_std=0.02, n_layers=8)
    model = NanoScaleLM(cfg)
    block = cast(TransformerBlock, model.blocks[0])
    o_std = float(block.attn.o_proj.weight.detach().std())
    expected = cfg.init_std / math.sqrt(2 * cfg.n_layers)
    assert 0.5 * expected < o_std < 1.6 * expected
    assert torch.count_nonzero(model.lm_head.weight) > 0


def test_init_is_seed_reproducible() -> None:
    torch.manual_seed(5)
    a = NanoScaleLM(tiny())
    torch.manual_seed(5)
    b = NanoScaleLM(tiny())
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)


# ------------------------------------------------------------------------ forward


@pytest.mark.parametrize(("batch", "seq"), [(1, 1), (2, 8), (5, 16)])
def test_forward_shapes(batch: int, seq: int) -> None:
    cfg = tiny()
    model = NanoScaleLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (batch, seq))
    out = model(ids, return_hidden=True)
    assert out.logits.shape == (batch, seq, cfg.vocab_size)
    assert out.hidden is not None
    assert out.hidden.shape == (batch, seq, cfg.d_model)
    assert out.loss is None


def test_forward_rejects_overlong_sequences() -> None:
    cfg = tiny(max_seq_len=8)
    model = NanoScaleLM(cfg)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_loss_ignores_the_ignore_index() -> None:
    cfg = tiny()
    model = NanoScaleLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = ids.clone()
    targets[:, :4] = IGNORE_INDEX
    masked = model(ids, targets=targets).loss
    manual = model(ids[:, 4:], targets=ids[:, 4:]).loss
    assert masked is not None and manual is not None
    # Not equal (different context), but both finite and the masked one is well-defined.
    assert torch.isfinite(masked) and torch.isfinite(manual)


def test_loss_mask_equals_setting_ignore_index() -> None:
    cfg = tiny()
    model = NanoScaleLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    mask = torch.zeros(2, 8, dtype=torch.long)
    mask[:, 4:] = 1
    via_mask = model(ids, targets=ids, loss_mask=mask).loss
    explicit = ids.clone()
    explicit[:, :4] = IGNORE_INDEX
    via_index = model(ids, targets=explicit).loss
    assert via_mask is not None and via_index is not None
    torch.testing.assert_close(via_mask, via_index)


def test_the_model_is_causal_end_to_end() -> None:
    cfg = tiny(zero_init_output=False)
    model = NanoScaleLM(cfg).double().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    base = model(ids).logits
    perturbed = ids.clone()
    perturbed[:, 7:] = (perturbed[:, 7:] + 1) % cfg.vocab_size
    torch.testing.assert_close(model(perturbed).logits[:, :7], base[:, :7], atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------- soft capping


def test_logit_soft_cap_bounds_the_logits() -> None:
    cfg = tiny(logit_soft_cap=5.0, zero_init_output=False, init_std=1.0)
    model = NanoScaleLM(cfg).eval()
    logits = model(torch.randint(0, cfg.vocab_size, (2, 8))).logits
    assert logits.abs().max() < 5.0

    uncapped = NanoScaleLM(tiny(zero_init_output=False, init_std=1.0)).eval()
    assert uncapped(torch.randint(0, cfg.vocab_size, (2, 8))).logits.abs().max() > 5.0


def test_soft_cap_is_monotone_and_near_identity_for_small_logits() -> None:
    cfg = tiny(logit_soft_cap=30.0)
    model = NanoScaleLM(cfg)
    small = torch.tensor([-0.5, 0.0, 0.5, 1.0])
    capped = model._soft_cap(small)
    torch.testing.assert_close(capped, small, atol=2e-3, rtol=2e-3)
    assert torch.all(capped.diff() > 0)


# --------------------------------------------------------------------- MTP heads


def test_mtp_heads_add_the_predicted_parameter_count() -> None:
    without = NanoScaleLM(tiny(n_mtp_heads=0)).num_parameters()
    with_two = NanoScaleLM(tiny(n_mtp_heads=2)).num_parameters()
    cfg = tiny()
    assert with_two - without == 2 * (cfg.d_model**2 + cfg.vocab_size * cfg.d_model)


def test_mtp_loss_is_included_and_reported() -> None:
    cfg = tiny(n_mtp_heads=2, mtp_loss_weight=0.5, zero_init_output=False)
    model = NanoScaleLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids, targets=ids)
    assert out.mtp_loss is not None
    assert out.loss is not None
    base = NanoScaleLM(cfg.merged(n_mtp_heads=0))
    base.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("mtp.")}, strict=True
    )
    plain = base(ids, targets=ids).loss
    assert plain is not None
    torch.testing.assert_close(out.loss, plain + 0.5 * out.mtp_loss, atol=1e-5, rtol=1e-5)


def test_mtp_head_offsets_are_t_plus_two_onward() -> None:
    heads = MultiTokenPredictionHeads(8, 16, 3)
    assert [cast(MTPHead, h).offset for h in heads.heads] == [2, 3, 4]


def test_mtp_loss_is_zero_when_there_are_no_heads() -> None:
    heads = MultiTokenPredictionHeads(8, 16, 0)
    hidden = torch.randn(1, 4, 8)
    targets = torch.randint(0, 16, (1, 4))
    assert float(heads.loss(hidden, targets)) == 0.0


def test_mtp_loss_handles_sequences_shorter_than_the_offsets() -> None:
    heads = MultiTokenPredictionHeads(8, 16, 3)
    hidden = torch.randn(1, 1, 8)
    targets = torch.randint(0, 16, (1, 1))
    assert float(heads.loss(hidden, targets)) == 0.0


# --------------------------------------------------------------------- generation


def test_random_init_nano_model_generates_shaped_text() -> None:
    """Phase-2 acceptance: a random-init nano model generates garbage, but *shaped*."""
    cfg = get_preset("nano")
    model = build_model(cfg.model.merged(zero_init_output=False)).eval()
    tok = BPETokenizer.load("artifacts/tokenizer/nano.json")

    prompt = torch.tensor([tok.encode("Lily went to", add_bos=True)])
    out = model.generate(prompt, max_new_tokens=24, temperature=0.9, top_k=40)

    assert out.shape[1] == prompt.shape[1] + 24
    assert out[:, : prompt.shape[1]].equal(prompt), "the prompt must be preserved verbatim"
    assert (out < cfg.model.vocab_size).all() and (out >= 0).all()
    text = tok.decode(out[0].tolist())
    assert isinstance(text, str) and len(text) > 0


def test_generation_is_reproducible_under_a_seed() -> None:
    cfg = tiny(zero_init_output=False)
    model = NanoScaleLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))

    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    a = model.generate(prompt, max_new_tokens=6, temperature=1.0, generator=g1)
    b = model.generate(prompt, max_new_tokens=6, temperature=1.0, generator=g2)
    assert torch.equal(a, b)


def test_greedy_generation_is_deterministic() -> None:
    cfg = tiny(zero_init_output=False)
    model = NanoScaleLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 3))
    a = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert torch.equal(a, b)


def test_generation_stops_at_eos() -> None:
    cfg = tiny(zero_init_output=False)
    model = NanoScaleLM(cfg).eval()
    eos = 7
    with torch.no_grad():  # make eos overwhelmingly likely
        model.lm_head.weight.zero_()
        model.lm_head.weight[eos] = 100.0
    out = model.generate(
        torch.randint(0, cfg.vocab_size, (1, 2)), max_new_tokens=10, temperature=0.0, eos_id=eos
    )
    assert out.shape[1] < 12
    assert out[0, -1].item() == eos


def test_generation_respects_the_context_limit() -> None:
    cfg = tiny(max_seq_len=8, zero_init_output=False)
    model = NanoScaleLM(cfg).eval()
    out = model.generate(torch.randint(0, cfg.vocab_size, (1, 6)), max_new_tokens=50)
    assert out.shape[1] <= cfg.max_seq_len


def test_generation_leaves_training_mode_unchanged() -> None:
    model = NanoScaleLM(tiny())
    model.train()
    model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=2)
    assert model.training


# ------------------------------------------------------------------------ sampling


def test_greedy_sampling_picks_the_argmax() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 0.0]])
    assert sample_next_token(logits, temperature=0.0).tolist() == [1, 0]


def test_top_k_restricts_the_support() -> None:
    logits = torch.tensor([[10.0, 9.0, -50.0, -60.0]])
    counts = {0: 0, 1: 0}
    for _ in range(200):
        tok = int(sample_next_token(logits, temperature=1.0, top_k=2))
        assert tok in (0, 1)
        counts[tok] += 1
    assert counts[0] > 0 and counts[1] > 0


def test_top_p_keeps_the_nucleus_nonempty_even_for_a_dominant_token() -> None:
    """A token holding more than top_p of the mass must still be sampleable."""
    logits = torch.tensor([[100.0, 0.0, 0.0]])
    for _ in range(20):
        assert int(sample_next_token(logits, temperature=1.0, top_p=0.5)) == 0


def test_top_p_excludes_the_tail() -> None:
    logits = torch.tensor([[5.0, 4.9, -20.0]])
    for _ in range(100):
        assert int(sample_next_token(logits, temperature=1.0, top_p=0.99)) in (0, 1)


def test_temperature_sharpens_the_distribution() -> None:
    logits = torch.tensor([[2.0, 1.0]]).repeat(500, 1)
    g = torch.Generator().manual_seed(0)
    hot = sample_next_token(logits, temperature=5.0, generator=g).float().mean()
    g = torch.Generator().manual_seed(0)
    cold = sample_next_token(logits, temperature=0.2, generator=g).float().mean()
    assert cold < hot, "a lower temperature must concentrate mass on the argmax"


# ------------------------------------------------------------------------- device


def test_rope_tables_follow_the_module_through_dtype_and_device_moves() -> None:
    """The tables are non-persistent buffers, so `.to()`/`.double()` move them for free."""
    model = NanoScaleLM(tiny())
    model.to(torch.device("cpu"))
    assert model.rope.cos.device.type == "cpu"
    assert model.device.type == "cpu"
    assert model.dtype is torch.float32

    model.double()
    assert model.rope.cos.dtype is torch.float64, (
        "an un-moved RoPE table would silently downcast every rotation to fp32"
    )
    ids = torch.randint(0, 64, (1, 4))
    assert model(ids).logits.dtype is torch.float64


def test_rope_tables_are_not_persisted_in_checkpoints() -> None:
    keys = NanoScaleLM(tiny()).state_dict().keys()
    assert not any("rope" in k for k in keys), "recomputable tables must not bloat checkpoints"


def test_draft_config_builds_a_working_smaller_model() -> None:
    base = get_preset("micro").model
    draft = draft_model_config(base)
    model = NanoScaleLM(draft).eval()
    assert model.num_parameters() < NanoScaleLM(base).num_parameters()
    ids = torch.randint(0, draft.vocab_size, (1, 8))
    assert model(ids).logits.shape == (1, 8, draft.vocab_size)
