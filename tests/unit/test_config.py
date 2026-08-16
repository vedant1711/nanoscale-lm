"""Phase-0 tests: config round-trip, presets, size ladder, overrides, manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nanoscale.config import (
    ALL_CONFIG_MODELS,
    TIERS,
    ExperimentConfig,
    ModelConfig,
    QuantConfig,
    TokenizerConfig,
    apply_overrides,
    deep_merge,
    draft_model_config,
    export_json_schemas,
    get_preset,
    load_experiment,
    parse_override,
    save_experiment,
)
from nanoscale.config.presets import TIER_EXPECTED_PARAMS, TOKENS_PER_PARAM

# ---------------------------------------------------------------------------------
# Presets and the size ladder
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_preset_builds_and_validates(tier: str) -> None:
    cfg = get_preset(tier)
    assert cfg.name == tier
    assert cfg.train.tier == tier
    assert cfg.model.vocab_size == cfg.tokenizer.vocab_size


@pytest.mark.parametrize("tier", TIERS)
def test_size_ladder_param_counts_are_pinned(tier: str) -> None:
    """The tier table is a contract; drift in any width/depth must fail here."""
    total, non_embed = TIER_EXPECTED_PARAMS[tier]
    breakdown = get_preset(tier).model.param_breakdown()
    assert breakdown["total"] == total
    assert breakdown["non_embedding"] == non_embed


@pytest.mark.parametrize(
    ("tier", "n_layers", "d_model", "n_heads", "ctx"),
    [("nano", 6, 256, 4, 256), ("micro", 8, 512, 8, 512), ("small", 12, 768, 12, 1024)],
)
def test_size_ladder_shapes_match_spec_table(
    tier: str, n_layers: int, d_model: int, n_heads: int, ctx: int
) -> None:
    m = get_preset(tier).model
    assert (m.n_layers, m.d_model, m.n_heads, m.max_seq_len) == (n_layers, d_model, n_heads, ctx)


@pytest.mark.parametrize("tier", ["micro", "small"])
def test_gpu_tiers_follow_the_20_to_1_token_budget(tier: str) -> None:
    """micro/small are compute-honest: their budget is the Chinchilla-style 20:1 figure."""
    cfg = get_preset(tier)
    assert cfg.train.token_budget == cfg.model.chinchilla_token_budget(TOKENS_PER_PARAM)


def test_nano_budget_is_step_driven_and_documented_as_sub_chinchilla() -> None:
    cfg = get_preset("nano")
    budget = cfg.train.token_budget
    assert budget is not None
    per_step = cfg.train.batch_size * cfg.train.grad_accum * cfg.data.seq_len
    assert budget == per_step * cfg.train.max_steps
    assert budget < cfg.model.chinchilla_token_budget()


def test_param_breakdown_is_internally_consistent() -> None:
    b = get_preset("micro").model.param_breakdown()
    assert b["total"] == b["embedding"] + b["blocks"] + b["final_norm"] + b["lm_head"] + b["mtp"]
    assert b["non_embedding"] == b["total"] - b["embedding"] - b["lm_head"]


def test_unknown_tier_raises() -> None:
    with pytest.raises(KeyError, match="Unknown tier"):
        get_preset("gigantic")


# ---------------------------------------------------------------------------------
# Round-trip and hashing
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_yaml_round_trip_preserves_config(tier: str, tmp_path: Path) -> None:
    cfg = get_preset(tier)
    path = save_experiment(cfg, tmp_path / f"{tier}.yaml")
    reloaded = load_experiment(path, tier=tier)
    assert reloaded == cfg
    assert reloaded.config_hash() == cfg.config_hash()


def test_config_hash_is_stable_and_sensitive() -> None:
    a = get_preset("nano")
    b = get_preset("nano")
    assert a.config_hash() == b.config_hash()
    changed = a.merged(name="nano-2")
    assert changed.config_hash() != a.config_hash()
    assert len(a.config_hash()) == 12


def test_configs_are_frozen() -> None:
    cfg = get_preset("nano")
    with pytest.raises(ValidationError):
        cfg.model.n_layers = 99  # type: ignore[misc]


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(nonexistent_field=3)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------------
# Overrides / merging
# ---------------------------------------------------------------------------------


def test_parse_override_types() -> None:
    assert parse_override("train.max_steps=10") == (["train", "max_steps"], 10)
    assert parse_override("train.optim.lr=1e-4") == (["train", "optim", "lr"], 1e-4)
    assert parse_override("model.qk_norm=false") == (["model", "qk_norm"], False)
    assert parse_override("model.logit_soft_cap=null") == (["model", "logit_soft_cap"], None)
    assert parse_override("name=abc") == (["name"], "abc")


def test_parse_override_requires_equals() -> None:
    with pytest.raises(ValueError, match=r"key\.path=value"):
        parse_override("train.max_steps")


def test_deep_merge_is_recursive_and_pure() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"c": 9}, "e": 4})
    assert out == {"a": {"b": 1, "c": 9}, "d": 3, "e": 4}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_apply_overrides_creates_missing_paths() -> None:
    out = apply_overrides({}, ["x.y.z=5"])
    assert out == {"x": {"y": {"z": 5}}}


def test_cli_overrides_win_over_yaml_and_preset(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"preset": "nano", "train": {"max_steps": 7}}), encoding="utf-8")
    cfg = load_experiment(path, overrides=["train.max_steps=3", "train.optim.name=adamw"])
    assert cfg.train.max_steps == 3
    assert cfg.train.optim.name == "adamw"
    assert cfg.model.n_layers == 6  # untouched preset value


def test_yaml_overrides_preset(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump({"preset": "micro", "train": {"max_steps": 11}}), encoding="utf-8"
    )
    cfg = load_experiment(path)
    assert cfg.name == "micro"
    assert cfg.train.max_steps == 11


def test_load_experiment_defaults_to_nano() -> None:
    assert load_experiment().name == "nano"


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_experiment(path)


# ---------------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------------


def test_gqa_divisibility_is_enforced() -> None:
    with pytest.raises(ValidationError, match="integer multiple"):
        ModelConfig(d_model=384, n_heads=6, n_kv_heads=4)
    with pytest.raises(ValidationError, match="n_kv_heads"):
        ModelConfig(d_model=256, n_heads=4, n_kv_heads=8)


def test_head_dim_must_be_even_for_rope() -> None:
    with pytest.raises(ValidationError, match="even"):
        ModelConfig(d_model=12, n_heads=4, n_kv_heads=2)  # head_dim = 3


def test_d_model_must_divide_by_heads() -> None:
    with pytest.raises(ValidationError, match="not divisible"):
        ModelConfig(d_model=100, n_heads=8, n_kv_heads=4)


def test_mtp_requires_untied_head() -> None:
    with pytest.raises(ValidationError, match="untied"):
        ModelConfig(tie_embeddings=True, n_mtp_heads=1)


def test_tokenizer_vocab_must_fit_bytes_and_specials() -> None:
    with pytest.raises(ValidationError, match="no room"):
        TokenizerConfig(vocab_size=100)


def test_tokenizer_merge_count() -> None:
    tok = TokenizerConfig(vocab_size=1000)
    assert tok.n_merges == 1000 - 256 - len(tok.special_tokens)


def test_experiment_cross_checks_vocab_and_seq_len() -> None:
    cfg = get_preset("nano")
    base = cfg.dump_inputs()
    with pytest.raises(ValidationError, match="must equal"):
        ExperimentConfig.model_validate(
            {**base, "model": {**cfg.model.dump_inputs(), "vocab_size": 99}}
        )
    with pytest.raises(ValidationError, match="exceeds"):
        ExperimentConfig.model_validate(
            {**base, "data": {**cfg.data.dump_inputs(), "seq_len": 100_000}}
        )


def test_dump_inputs_round_trips_through_validate() -> None:
    """dump_inputs must be exactly the inverse of model_validate for every config."""
    cfg = get_preset("micro")
    assert ExperimentConfig.model_validate(cfg.dump_inputs()) == cfg
    dumped = cfg.dump_inputs()
    assert "head_dim" not in dumped["model"]
    assert "ffn_dim" not in dumped["model"]
    assert "effective_batch" not in dumped["train"]
    assert "n_merges" not in dumped["tokenizer"]


def test_quant_group_size_validation() -> None:
    assert QuantConfig(group_size=-1).group_size == -1
    with pytest.raises(ValidationError):
        QuantConfig(group_size=0)


def test_ffn_dim_rule() -> None:
    swiglu = ModelConfig(d_model=256, mlp_type="swiglu")
    relu2 = ModelConfig(d_model=256, mlp_type="relu2")
    assert swiglu.ffn_dim % 64 == 0
    assert swiglu.ffn_dim == 704  # ceil(8/3 * 256) rounded up to a multiple of 64
    assert relu2.ffn_dim == 1024
    assert ModelConfig(d_model=256, d_ff=333).ffn_dim == 333


# ---------------------------------------------------------------------------------
# Draft/student derivation (used by Arc 2)
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_draft_config_is_smaller_and_valid(tier: str) -> None:
    base = get_preset(tier).model
    draft = draft_model_config(base)
    assert draft.param_count() < base.param_count()
    assert draft.vocab_size == base.vocab_size  # must share the tokenizer
    assert draft.max_seq_len == base.max_seq_len
    assert draft.head_dim % 2 == 0
    assert draft.n_heads % draft.n_kv_heads == 0


# ---------------------------------------------------------------------------------
# JSON Schema export
# ---------------------------------------------------------------------------------


def test_json_schema_export(tmp_path: Path) -> None:
    written = export_json_schemas(tmp_path)
    assert len(written) == len(ALL_CONFIG_MODELS)
    payload = json.loads((tmp_path / "ModelConfig.json").read_text(encoding="utf-8"))
    assert payload["title"] == "ModelConfig"
    assert "n_kv_heads" in payload["properties"]


def test_every_field_is_documented() -> None:
    """A config field without a description is an undocumented knob."""
    undocumented: list[str] = []
    for model in ALL_CONFIG_MODELS:
        for field_name, field in model.model_fields.items():
            if field.description is None and field.default_factory is None:
                undocumented.append(f"{model.__name__}.{field_name}")
    # Sub-config fields carry their documentation on the nested model itself.
    assert undocumented == [], f"undocumented config fields: {undocumented}"
