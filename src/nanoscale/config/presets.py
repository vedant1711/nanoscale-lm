"""The compute-honest size ladder (spec A4).

All tiers share one architecture; only depth/width/context/token-budget change. Token
budgets follow the Chinchilla-style 20:1 tokens:parameters heuristic (Hoffmann et al.,
arXiv:2203.15556) so that runs are compute-honest rather than arbitrary.

===========  ===========  ==========================  =========  ==============
Tier         Params       layers x d_model x heads    Context    Token budget
===========  ===========  ==========================  =========  ==============
``nano``     ~2-4 M       6 x 256 x 4                 256        ~60-80 M
``micro``    ~25 M        8 x 512 x 8                 512        ~500 M
``small``    ~120 M       12 x 768 x 12               1024       ~2-3 B
===========  ===========  ==========================  =========  ==============

``micro`` is the default tier for everything *reported*; ``nano`` is the tier that has
to stay runnable on a CPU for CI and the sub-10-minute smoke test; ``small`` is the
documented scale-up recipe.

Two honesty notes, both of which the docs repeat:

1. The spec's headline parameter figures are approximate. This module pins the
   **shapes** from the spec table exactly and reports the parameter counts that those
   shapes actually produce, for both the total and the non-embedding count (the latter
   is what the spec's "~25M" style figures track most closely). The exact expected
   counts are asserted in ``tests/unit/test_config.py`` so architecture drift is caught.
2. ``micro`` and ``small`` train to their full 20:1 Chinchilla budget. ``nano`` does
   **not**: 20:1 on ~5M parameters is ~99M tokens, which is hours of CPU time, and
   ``nano`` exists to be a <10-minute CI/teaching run. Its stopping budget is therefore
   set from its step count, and every ``nano`` manifest records what fraction of the
   compute-optimal budget it covered.
"""

from __future__ import annotations

from typing import Final, Literal

from nanoscale.config.schemas import (
    AlignConfig,
    BenchConfig,
    DataConfig,
    DistillConfig,
    ExperimentConfig,
    GenerateConfig,
    GRPOConfig,
    ModelConfig,
    OptimConfig,
    PreferenceConfig,
    QuantConfig,
    ScheduleConfig,
    SFTConfig,
    SpecConfig,
    TokenizerConfig,
    TrainConfig,
)

Tier = Literal["nano", "micro", "small"]

TIERS: Final[tuple[str, ...]] = ("nano", "micro", "small")

#: Chinchilla-style tokens-per-parameter ratio used to set every tier's token budget.
TOKENS_PER_PARAM: Final[int] = 20

#: Exact ``(total, non_embedding)`` parameter counts each tier must produce.
#:
#: These are a regression guard, asserted in ``tests/unit/test_config.py`` against both
#: the analytic count and the count of the built ``nn.Module``. Any accidental change to
#: a width, a depth or the FFN expansion rule fails that test loudly.
TIER_EXPECTED_PARAMS: Final[dict[str, tuple[int, int]]] = {
    "nano": (4_952_064, 4_427_776),
    "micro": (40_379_904, 23_602_688),
    "small": (125_849_856, 75_518_208),
}


def _nano() -> ExperimentConfig:
    """CPU/laptop tier: the CI, smoke-test and teaching configuration.

    The 1024-token vocabulary is not arbitrary: it is exactly what the offline toy
    corpus (:mod:`nanoscale.data.toy`) can fill. A larger vocabulary would leave dead
    embedding rows that cost parameters and CPU time while carrying no information.
    """
    tokenizer = TokenizerConfig(vocab_size=1024, max_train_bytes=1_500_000)
    model = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=6,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
    )
    data = DataConfig(source="toy", seq_len=256, val_fraction=0.05, shuffle_buffer=256)
    # Step-driven budget, not Chinchilla-driven: see the module docstring, honesty note 2.
    nano_steps = 400
    train = TrainConfig(
        tier="nano",
        batch_size=8,
        grad_accum=1,
        max_steps=nano_steps,
        token_budget=8 * 256 * nano_steps,
        eval_interval=50,
        eval_batches=8,
        log_interval=10,
        ckpt_interval=200,
        amp_dtype="fp32",
        out_dir="runs/nano/pretrain",
        optim=OptimConfig(name="muon", lr=2e-2, adamw_lr=3e-3, weight_decay=0.05),
        schedule=ScheduleConfig(name="cosine", warmup_frac=0.05, min_lr_frac=0.1),
    )
    align = AlignConfig(
        sft=SFTConfig(seq_len=256, batch_size=8, max_steps=150, lr=3e-4, out_dir="runs/nano/sft"),
        preference=PreferenceConfig(
            seq_len=256, batch_size=4, max_steps=120, lr=5e-5, beta=0.1, out_dir="runs/nano/align"
        ),
        grpo=GRPOConfig(group_size=6, n_prompts=4, max_steps=30, out_dir="runs/nano/grpo"),
    )
    distill = DistillConfig(
        seq_len=256, batch_size=8, max_steps=150, lr=3e-4, out_dir="runs/nano/distill"
    )
    quant = QuantConfig(
        bits=4, group_size=64, calib_samples=32, calib_seq_len=256, out_dir="runs/nano/quant"
    )
    return ExperimentConfig(
        name="nano",
        tokenizer=tokenizer,
        model=model,
        data=data,
        train=train,
        align=align,
        distill=distill,
        quant=quant,
        spec=SpecConfig(gamma=4, max_new_tokens=64),
        generate=GenerateConfig(max_new_tokens=64),
        bench=BenchConfig(prompt_len=16, max_new_tokens=48, out_dir="results/bench/nano"),
    )


def _micro() -> ExperimentConfig:
    """Free-GPU tier (T4/P100): the main reported model."""
    tokenizer = TokenizerConfig(vocab_size=16384, max_train_bytes=8_000_000)
    model = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=8,
        d_model=512,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=512,
    )
    data = DataConfig(
        source="hf",
        hf_dataset="HuggingFaceFW/fineweb-edu",
        hf_config="sample-10BT",
        seq_len=512,
        val_fraction=0.01,
        shuffle_buffer=2048,
    )
    # 20:1 budget / (batch*accum*seq_len) tokens per step.
    budget = model.chinchilla_token_budget(TOKENS_PER_PARAM)
    tokens_per_step = 24 * 4 * 512
    train = TrainConfig(
        tier="micro",
        batch_size=24,
        grad_accum=4,
        max_steps=max(1, budget // tokens_per_step),
        token_budget=budget,
        eval_interval=250,
        eval_batches=20,
        log_interval=20,
        ckpt_interval=1000,
        amp_dtype="bf16",
        out_dir="runs/micro/pretrain",
        optim=OptimConfig(name="muon", lr=1e-2, adamw_lr=6e-4, weight_decay=0.1),
        schedule=ScheduleConfig(name="cosine", warmup_frac=0.02, min_lr_frac=0.1),
    )
    align = AlignConfig(
        sft=SFTConfig(
            seq_len=512,
            batch_size=8,
            grad_accum=2,
            max_steps=1000,
            lr=1e-4,
            out_dir="runs/micro/sft",
        ),
        preference=PreferenceConfig(
            seq_len=512,
            batch_size=4,
            grad_accum=2,
            max_steps=600,
            lr=5e-6,
            beta=0.1,
            out_dir="runs/micro/align",
        ),
        grpo=GRPOConfig(group_size=8, n_prompts=8, max_steps=200, out_dir="runs/micro/grpo"),
    )
    distill = DistillConfig(
        seq_len=512, batch_size=8, max_steps=1500, lr=3e-4, out_dir="runs/micro/distill"
    )
    quant = QuantConfig(
        bits=4, group_size=128, calib_samples=128, calib_seq_len=512, out_dir="runs/micro/quant"
    )
    return ExperimentConfig(
        name="micro",
        tokenizer=tokenizer,
        model=model,
        data=data,
        train=train,
        align=align,
        distill=distill,
        quant=quant,
        spec=SpecConfig(gamma=5, max_new_tokens=128),
        generate=GenerateConfig(max_new_tokens=128),
        bench=BenchConfig(prompt_len=32, max_new_tokens=128, out_dir="results/bench/micro"),
    )


def _small() -> ExperimentConfig:
    """GPT-2-ish stretch tier: documented recipe, run it when you have the compute."""
    tokenizer = TokenizerConfig(vocab_size=32768, max_train_bytes=32_000_000)
    model = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=12,
        d_model=768,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=1024,
    )
    data = DataConfig(
        source="hf",
        hf_dataset="HuggingFaceFW/fineweb-edu",
        hf_config="sample-10BT",
        seq_len=1024,
        val_fraction=0.005,
        shuffle_buffer=4096,
    )
    budget = model.chinchilla_token_budget(TOKENS_PER_PARAM)
    tokens_per_step = 16 * 16 * 1024
    train = TrainConfig(
        tier="small",
        batch_size=16,
        grad_accum=16,
        max_steps=max(1, budget // tokens_per_step),
        token_budget=budget,
        eval_interval=500,
        eval_batches=40,
        log_interval=25,
        ckpt_interval=2000,
        amp_dtype="bf16",
        out_dir="runs/small/pretrain",
        optim=OptimConfig(name="muon", lr=6e-3, adamw_lr=3e-4, weight_decay=0.1),
        schedule=ScheduleConfig(name="wsd", warmup_frac=0.01, decay_frac=0.2, min_lr_frac=0.02),
    )
    align = AlignConfig(
        sft=SFTConfig(
            seq_len=1024,
            batch_size=4,
            grad_accum=8,
            max_steps=2000,
            lr=5e-5,
            out_dir="runs/small/sft",
        ),
        preference=PreferenceConfig(
            seq_len=1024,
            batch_size=2,
            grad_accum=8,
            max_steps=1000,
            lr=5e-7,
            beta=0.1,
            out_dir="runs/small/align",
        ),
        grpo=GRPOConfig(group_size=8, n_prompts=8, max_steps=500, out_dir="runs/small/grpo"),
    )
    distill = DistillConfig(
        seq_len=1024, batch_size=4, max_steps=3000, lr=2e-4, out_dir="runs/small/distill"
    )
    quant = QuantConfig(
        bits=4, group_size=128, calib_samples=256, calib_seq_len=1024, out_dir="runs/small/quant"
    )
    return ExperimentConfig(
        name="small",
        tokenizer=tokenizer,
        model=model,
        data=data,
        train=train,
        align=align,
        distill=distill,
        quant=quant,
        spec=SpecConfig(gamma=5, max_new_tokens=256),
        generate=GenerateConfig(max_new_tokens=256),
        bench=BenchConfig(prompt_len=64, max_new_tokens=256, out_dir="results/bench/small"),
    )


_BUILDERS: Final[dict[str, object]] = {"nano": _nano, "micro": _micro, "small": _small}


def get_preset(tier: str) -> ExperimentConfig:
    """Return the :class:`ExperimentConfig` for a size-ladder tier.

    Args:
        tier: One of ``"nano"``, ``"micro"``, ``"small"``.

    Raises:
        KeyError: If the tier is unknown.
    """
    if tier not in _BUILDERS:
        raise KeyError(f"Unknown tier {tier!r}; expected one of {TIERS}.")
    builder = _BUILDERS[tier]
    assert callable(builder)
    result = builder()
    assert isinstance(result, ExperimentConfig)
    return result


def draft_model_config(base: ModelConfig, *, shrink: int = 4) -> ModelConfig:
    """Derive a small draft/student config from a target/teacher config.

    Used by both Arc-2 tracks: the distillation student and the speculative-decoding
    draft model. Depth is halved and width divided by ``shrink`` (rounded to a multiple
    of the head count so the head dimension stays even for RoPE).

    Args:
        base: The teacher/target model config.
        shrink: Width divisor.
    """
    n_heads = max(1, base.n_heads // 2)
    raw_width = max(64, base.d_model // shrink)
    d_model = max(n_heads * 2, ((raw_width + n_heads - 1) // n_heads) * n_heads)
    while (d_model // n_heads) % 2 != 0:
        d_model += n_heads
    n_kv_heads = max(1, min(n_heads, base.n_kv_heads // 2))
    while n_heads % n_kv_heads != 0:
        n_kv_heads -= 1
    return ModelConfig(
        vocab_size=base.vocab_size,
        n_layers=max(2, base.n_layers // 2),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=base.max_seq_len,
        norm_type=base.norm_type,
        mlp_type=base.mlp_type,
        qk_norm=base.qk_norm,
        tie_embeddings=base.tie_embeddings,
        zero_init_output=base.zero_init_output,
        logit_soft_cap=base.logit_soft_cap,
        rope_theta=base.rope_theta,
        norm_eps=base.norm_eps,
        attn_impl=base.attn_impl,
    )
