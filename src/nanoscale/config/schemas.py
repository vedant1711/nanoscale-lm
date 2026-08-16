"""Pydantic v2 configuration schemas for every phase of NanoScale-LM.

Design rules for this module:

* One frozen, fully-validated config object per concern. Configs are hashable
  (see :meth:`BaseConfig.config_hash`) so that a run manifest can pin exactly which
  configuration produced a given artifact.
* Every field carries a description; the JSON Schema exported from these models is a
  committed artifact (``configs/schema/``) and doubles as user documentation.
* No config object ever touches the filesystem or torch. Loading/merging lives in
  :mod:`nanoscale.config.loader`, presets live in :mod:`nanoscale.config.presets`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# --------------------------------------------------------------------------------------
# Shared base
# --------------------------------------------------------------------------------------

PositiveInt = Annotated[int, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class BaseConfig(BaseModel):
    """Base class for all NanoScale configs: frozen, strict, and hashable."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )

    @classmethod
    def strip_computed(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively drop ``computed_field`` outputs from a dump of this model.

        Computed fields (``head_dim``, ``ffn_dim``, ...) are derived, so they appear in
        ``model_dump`` but are rejected as *inputs* by ``extra="forbid"``. Anything that
        round-trips a config through a dict — YAML save/load, ``merged``, the CLI
        override path — has to strip them first.
        """
        out: dict[str, Any] = {}
        for name, value in data.items():
            if name in cls.model_computed_fields:
                continue
            field = cls.model_fields.get(name)
            annotation = field.annotation if field is not None else None
            if (
                isinstance(value, dict)
                and isinstance(annotation, type)
                and issubclass(annotation, BaseConfig)
            ):
                out[name] = annotation.strip_computed(value)
            else:
                out[name] = value
        return out

    def dump_inputs(self, *, mode: Literal["python", "json"] = "python") -> dict[str, Any]:
        """Dump only the fields that can be fed back into ``model_validate``."""
        return type(self).strip_computed(self.model_dump(mode=mode))

    def config_hash(self) -> str:
        """Return a stable 12-hex-char hash of this config.

        The hash is computed over the canonical JSON dump of the *input* fields (sorted
        keys), so it is invariant to field ordering and to how the config was
        constructed. It is recorded in the run manifest to make every artifact traceable
        to the exact configuration that produced it.
        """
        payload = json.dumps(self.dump_inputs(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def merged(self, **overrides: Any) -> Self:
        """Return a copy of this config with ``overrides`` applied and re-validated."""
        data = self.dump_inputs()
        data.update(overrides)
        return type(self).model_validate(data)


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------


class TokenizerConfig(BaseConfig):
    """Byte-level BPE tokenizer configuration (spec B1)."""

    vocab_size: PositiveInt = Field(
        8192, description="Target vocabulary size including the 256 byte tokens and specials."
    )
    special_tokens: tuple[str, ...] = Field(
        ("<bos>", "<eos>", "<pad>", "<user>", "<assistant>", "<system>", "<eot>"),
        description="Special tokens; the chat roles are used by the SFT chat template.",
    )
    lowercase: bool = Field(False, description="Never lowercase by default: byte-level is exact.")
    split_pattern: Literal["gpt2", "gpt4", "none"] = Field(
        "gpt2",
        description=(
            "Pre-tokenization regex. GPT-2/GPT-4 style splitting keeps merges from crossing "
            "word/punctuation boundaries, which materially improves compression."
        ),
    )
    max_train_bytes: PositiveInt = Field(
        2_000_000, description="Cap on corpus bytes used for BPE training (keeps training quick)."
    )

    @model_validator(mode="after")
    def _check_vocab_room(self) -> TokenizerConfig:
        floor = 256 + len(self.special_tokens)
        if self.vocab_size < floor:
            raise ValueError(
                f"vocab_size={self.vocab_size} leaves no room for 256 byte tokens plus "
                f"{len(self.special_tokens)} specials (need >= {floor})."
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_merges(self) -> int:
        """Number of BPE merges implied by the target vocabulary size."""
        return self.vocab_size - 256 - len(self.special_tokens)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class ModelConfig(BaseConfig):
    """Decoder-only transformer configuration (spec B2).

    Defaults are the 2026 "architecture that won" for small decoder-only LMs:
    RoPE + RMSNorm + SwiGLU + GQA + QK-norm, untied head, zero-init output projections.
    The modded-nanoGPT speedrun extras (ReLU^2 MLP, logit soft-cap, MTP head) are
    toggles so that Phase 5 can *measure* them rather than assert them.
    """

    # --- shape -------------------------------------------------------------------
    vocab_size: PositiveInt = Field(8192, description="Must match the trained tokenizer.")
    n_layers: PositiveInt = Field(6, description="Number of transformer blocks.")
    d_model: PositiveInt = Field(256, description="Residual-stream width.")
    n_heads: PositiveInt = Field(4, description="Number of query heads.")
    n_kv_heads: PositiveInt = Field(
        2, description="Number of key/value heads (GQA). Must divide n_heads."
    )
    d_ff: PositiveInt | None = Field(
        None,
        description=(
            "Feed-forward hidden width. If None it is derived as a multiple of d_model "
            "(8/3 for SwiGLU so that parameter count matches a 4x non-gated MLP)."
        ),
    )
    max_seq_len: PositiveInt = Field(256, description="Maximum context length.")

    # --- architecture toggles ----------------------------------------------------
    norm_type: Literal["rmsnorm", "layernorm"] = Field("rmsnorm", description="Normalization.")
    mlp_type: Literal["swiglu", "relu2"] = Field(
        "swiglu", description="SwiGLU default; ReLU^2 is the modded-nanoGPT speedrun variant."
    )
    qk_norm: bool = Field(True, description="RMS-normalize q and k before the dot product.")
    tie_embeddings: bool = Field(False, description="Untied LM head by default (speedrun stack).")
    zero_init_output: bool = Field(
        True, description="Zero-init attention/MLP output projections and the LM head (muP-like)."
    )
    logit_soft_cap: float | None = Field(
        None,
        gt=0.0,
        description="If set, apply c*tanh(logits/c) (Gemma-2 style). None disables it.",
    )
    n_mtp_heads: int = Field(
        0,
        ge=0,
        le=4,
        description=(
            "Extra multi-token-prediction heads predicting t+2, t+3, ... Used both as a "
            "training-quality ablation and, in Arc 2, as a self-speculation draft."
        ),
    )
    mtp_loss_weight: NonNegFloat = Field(0.3, description="Weight of the auxiliary MTP loss.")

    # --- positional / numerics ---------------------------------------------------
    rope_theta: float = Field(10_000.0, gt=0.0, description="RoPE base frequency.")
    rope_scaling: float = Field(1.0, gt=0.0, description="Linear position-interpolation factor.")
    norm_eps: float = Field(1e-5, gt=0.0, description="Epsilon inside RMSNorm/LayerNorm.")
    dropout: UnitFloat = Field(0.0, description="Dropout is off by default at this scale.")
    init_std: float = Field(0.02, gt=0.0, description="Std of the truncated-normal init.")
    attn_impl: Literal["manual", "sdpa"] = Field(
        "manual",
        description=(
            "'manual' is the from-scratch reference path. 'sdpa' is an optional fast path "
            "that must match 'manual' numerically (tested in tests/unit/test_attention.py)."
        ),
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> ModelConfig:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}.")
        if self.n_kv_heads > self.n_heads:
            raise ValueError(f"n_kv_heads={self.n_kv_heads} > n_heads={self.n_heads}.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must be an integer multiple of "
                f"n_kv_heads={self.n_kv_heads} for grouped-query attention."
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE's pair rotation.")
        if self.tie_embeddings and self.n_mtp_heads > 0:
            raise ValueError("Multi-token-prediction heads require an untied LM head.")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def head_dim(self) -> int:
        """Per-head dimension."""
        return self.d_model // self.n_heads

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ffn_dim(self) -> int:
        """Resolved feed-forward hidden width.

        For SwiGLU we use 8/3*d_model rounded up to a multiple of 64, which keeps the
        parameter count comparable to a classic 4*d_model non-gated MLP (three matrices
        of width 8/3 vs two of width 4).
        """
        if self.d_ff is not None:
            return self.d_ff
        raw = int(8 * self.d_model / 3) if self.mlp_type == "swiglu" else 4 * self.d_model
        return ((raw + 63) // 64) * 64

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_kv_groups(self) -> int:
        """Number of query heads sharing each key/value head."""
        return self.n_heads // self.n_kv_heads

    def param_breakdown(self) -> dict[str, int]:
        """Analytic parameter breakdown, counted exactly as the modules build them.

        Returns a mapping with ``embedding``, ``blocks``, ``final_norm``, ``lm_head``,
        ``mtp``, ``total`` and ``non_embedding``. The non-embedding count is the one
        that scales with depth/width and is the number usually quoted when people
        describe a model as "an N-parameter transformer".
        """
        d, h, kv = self.d_model, self.head_dim, self.n_kv_heads
        embed = self.vocab_size * d
        attn = d * (self.n_heads * h) + 2 * d * (kv * h) + (self.n_heads * h) * d
        mlp = 3 * d * self.ffn_dim if self.mlp_type == "swiglu" else 2 * d * self.ffn_dim
        norms = 2 * d  # two pre-norm gains per block
        qk_gains = 2 * h if self.qk_norm else 0
        blocks = self.n_layers * (attn + mlp + norms + qk_gains)
        head = 0 if self.tie_embeddings else self.vocab_size * d
        mtp = self.n_mtp_heads * (d * d + self.vocab_size * d)
        total = embed + blocks + d + head + mtp
        return {
            "embedding": embed,
            "blocks": blocks,
            "final_norm": d,
            "lm_head": head,
            "mtp": mtp,
            "total": total,
            "non_embedding": total - embed - head,
        }

    def param_count(self) -> int:
        """Total analytic parameter count (see :meth:`param_breakdown`)."""
        return self.param_breakdown()["total"]

    def chinchilla_token_budget(self, ratio: int = 20) -> int:
        """Compute-optimal token budget under the Chinchilla-style ratio heuristic.

        Hoffmann et al. (arXiv:2203.15556) put the compute-optimal ratio near 20 tokens
        per parameter. We apply it to the **total** parameter count, which is the
        convention used by nanochat's budgeting.
        """
        return self.param_count() * ratio


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


class DataConfig(BaseConfig):
    """Streaming/packed data pipeline configuration (spec B4)."""

    source: Literal["toy", "textfile", "hf"] = Field(
        "toy",
        description=(
            "'toy' = the committed offline corpus (CI/laptop tier, no network); "
            "'textfile' = local .txt/.jsonl paths; 'hf' = streaming Hugging Face dataset."
        ),
    )
    hf_dataset: str = Field("HuggingFaceFW/fineweb-edu", description="HF dataset repo id.")
    hf_config: str | None = Field("sample-10BT", description="HF dataset config name.")
    hf_split: str = Field("train", description="HF split.")
    hf_text_field: str = Field("text", description="Field of the HF record holding raw text.")
    paths: tuple[str, ...] = Field((), description="Local paths when source='textfile'.")
    seq_len: PositiveInt = Field(256, description="Packed sequence length (must be <= ctx).")
    val_fraction: UnitFloat = Field(0.02, description="Fraction of tokens held out for validation.")
    shuffle_buffer: PositiveInt = Field(1024, description="Streaming shuffle buffer, in documents.")
    num_workers: int = Field(0, ge=0, description="Dataloader workers; 0 keeps CPU runs simple.")


# --------------------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------------------


class OptimConfig(BaseConfig):
    """Optimizer configuration (spec B3): Muon for hidden matrices, AdamW for the rest."""

    name: Literal["adamw", "muon"] = Field(
        "muon",
        description=(
            "'adamw' routes everything to AdamW. 'muon' uses the documented split: Muon for "
            "2D hidden matmul weights, AdamW for embeddings, LM head, norms and scalars."
        ),
    )
    lr: float = Field(3e-3, gt=0.0, description="Peak LR for the Muon group.")
    adamw_lr: float = Field(3e-4, gt=0.0, description="Peak LR for the AdamW group.")
    betas: tuple[float, float] = Field((0.9, 0.95), description="AdamW betas.")
    eps: float = Field(1e-8, gt=0.0, description="AdamW epsilon.")
    weight_decay: NonNegFloat = Field(0.1, description="Decoupled weight decay.")
    cautious_weight_decay: bool = Field(
        False,
        description=(
            "Feb-2026 improvement: decay only the components of the weight that agree in sign "
            "with the update, on a schedule that decays to zero. Flag so it can be A/B'd."
        ),
    )
    muon_momentum: float = Field(0.95, ge=0.0, lt=1.0, description="Muon momentum coefficient.")
    muon_nesterov: bool = Field(True, description="Nesterov-style momentum inside Muon.")
    muon_ns_steps: PositiveInt = Field(5, description="Newton-Schulz iterations per step.")
    grad_clip: NonNegFloat = Field(1.0, description="Global grad-norm clip; 0 disables.")


class ScheduleConfig(BaseConfig):
    """Learning-rate schedule (spec B4)."""

    name: Literal["cosine", "wsd", "constant"] = Field(
        "cosine", description="Cosine+warmup default; WSD (warmup-stable-decay) as a flag."
    )
    warmup_frac: UnitFloat = Field(0.02, description="Fraction of total steps spent warming up.")
    decay_frac: UnitFloat = Field(0.2, description="WSD only: fraction of steps in the decay tail.")
    min_lr_frac: UnitFloat = Field(0.1, description="Floor LR as a fraction of peak.")


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


class TrainConfig(BaseConfig):
    """Pretraining loop configuration (spec B4)."""

    tier: Literal["nano", "micro", "small"] = Field("nano", description="Size-ladder tier name.")
    seed: int = Field(1337, description="Global seed; controls init, dataloading and dropout.")
    batch_size: PositiveInt = Field(16, description="Micro-batch size (sequences per step).")
    grad_accum: PositiveInt = Field(1, description="Gradient-accumulation steps.")
    max_steps: PositiveInt = Field(200, description="Optimizer steps (upper bound).")
    token_budget: PositiveInt | None = Field(
        None,
        description=(
            "If set, training stops once this many tokens have been consumed. Presets set it "
            "from the Chinchilla-style 20:1 tokens:params heuristic."
        ),
    )
    eval_interval: PositiveInt = Field(50, description="Steps between validation passes.")
    eval_batches: PositiveInt = Field(10, description="Validation batches per evaluation.")
    log_interval: PositiveInt = Field(10, description="Steps between training-log rows.")
    ckpt_interval: PositiveInt = Field(100, description="Steps between checkpoint writes.")
    amp_dtype: Literal["fp32", "bf16", "fp16"] = Field(
        "fp32",
        description=(
            "Autocast dtype. fp32 on CPU (the nano tier); bf16 on Colab/Kaggle GPUs, always "
            "with an fp32 master-weight copy held by the optimizer."
        ),
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field("auto", description="Compute device.")
    compile_model: bool = Field(False, description="torch.compile the model (GPU tiers only).")
    out_dir: str = Field("runs/pretrain", description="Run directory for logs/checkpoints.")
    resume: str | None = Field(None, description="Checkpoint path to resume from.")
    deterministic: bool = Field(True, description="Force deterministic kernels and dataloading.")

    optim: OptimConfig = Field(default_factory=OptimConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_batch(self) -> int:
        """Sequences per optimizer step."""
        return self.batch_size * self.grad_accum


# --------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------


class SFTConfig(BaseConfig):
    """Supervised fine-tuning configuration (spec B5)."""

    seed: int = Field(1337, description="Seed for init, batching and any sampling.")
    dataset: Literal["toy", "jsonl"] = Field("toy", description="Instruction dataset source.")
    paths: tuple[str, ...] = Field((), description="JSONL paths when dataset='jsonl'.")
    seq_len: PositiveInt = Field(256, description="Chat sequences are padded/truncated to this.")
    batch_size: PositiveInt = Field(8, description="Conversations per micro-batch.")
    grad_accum: PositiveInt = Field(1, description="Gradient-accumulation steps.")
    max_steps: PositiveInt = Field(120, description="Optimizer steps.")
    lr: float = Field(1e-4, gt=0.0, description="Peak learning rate (AdamW).")
    weight_decay: NonNegFloat = Field(0.0, description="Decoupled weight decay.")
    warmup_frac: UnitFloat = Field(0.05, description="Warmup fraction of total steps.")
    mask_prompt: bool = Field(
        True, description="Loss on completion tokens only (verified by test)."
    )
    eval_interval: PositiveInt = Field(40, description="Steps between held-out evaluations.")
    log_interval: PositiveInt = Field(10, description="Steps between log rows.")
    out_dir: str = Field("runs/sft", description="Run directory.")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field("auto", description="Compute device.")


class PreferenceConfig(BaseConfig):
    """DPO / SimPO configuration (spec B5)."""

    method: Literal["dpo", "simpo"] = Field("dpo", description="Preference-optimization method.")
    seed: int = Field(1337, description="Seed for init and batching.")
    dataset: Literal["toy", "jsonl"] = Field("toy", description="Preference dataset source.")
    paths: tuple[str, ...] = Field((), description="JSONL paths when dataset='jsonl'.")
    seq_len: PositiveInt = Field(256, description="Max prompt+response length.")
    batch_size: PositiveInt = Field(4, description="Preference pairs per micro-batch.")
    grad_accum: PositiveInt = Field(1, description="Gradient-accumulation steps.")
    max_steps: PositiveInt = Field(100, description="Optimizer steps.")
    lr: float = Field(
        5e-6, gt=0.0, description="Peak learning rate; preference tuning wants a low LR."
    )
    beta: float = Field(0.1, gt=0.0, description="DPO/SimPO inverse-temperature on the reward.")
    gamma: NonNegFloat = Field(
        0.5, description="SimPO target reward margin (gamma/beta in the paper's parameterisation)."
    )
    label_smoothing: UnitFloat = Field(0.0, description="cDPO-style label smoothing.")
    sft_loss_weight: NonNegFloat = Field(
        0.0, description="Optional auxiliary NLL on the chosen response (RPO-style stabiliser)."
    )
    log_interval: PositiveInt = Field(10, description="Steps between log rows.")
    out_dir: str = Field("runs/align", description="Run directory.")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field("auto", description="Compute device.")


class GRPOConfig(BaseConfig):
    """Group-relative policy optimization on a verifiable reward (spec B5, optional track)."""

    seed: int = Field(1337, description="Seed for rollouts and batching.")
    group_size: PositiveInt = Field(8, description="Completions sampled per prompt.")
    n_prompts: PositiveInt = Field(4, description="Prompts per optimizer step.")
    max_steps: PositiveInt = Field(50, description="Optimizer steps.")
    max_new_tokens: PositiveInt = Field(24, description="Rollout length.")
    temperature: float = Field(1.0, gt=0.0, description="Rollout sampling temperature.")
    lr: float = Field(1e-5, gt=0.0, description="Policy learning rate.")
    kl_coef: NonNegFloat = Field(0.02, description="KL penalty toward the frozen reference policy.")
    clip_eps: float = Field(0.2, gt=0.0, description="PPO-style ratio clip.")
    log_interval: PositiveInt = Field(5, description="Steps between log rows.")
    out_dir: str = Field("runs/grpo", description="Run directory.")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field("auto", description="Compute device.")


class AlignConfig(BaseConfig):
    """Umbrella alignment config so a single YAML can drive the whole post-training stack."""

    sft: SFTConfig = Field(default_factory=SFTConfig)
    preference: PreferenceConfig = Field(default_factory=PreferenceConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    enable_grpo: bool = Field(False, description="The RLVR track is opt-in (spec B5).")


# --------------------------------------------------------------------------------------
# Arc 2: distillation / quantization / speculative decoding
# --------------------------------------------------------------------------------------


class DistillConfig(BaseConfig):
    """Knowledge-distillation configuration (spec B6)."""

    method: Literal["forward_kl", "seqkd", "reverse_kl"] = Field(
        "reverse_kl",
        description=(
            "'forward_kl' = Hinton token KD (mode-covering baseline); 'seqkd' = MLE on "
            "teacher samples; 'reverse_kl' = MiniLLM-style on-policy reverse KL (headline)."
        ),
    )
    seed: int = Field(1337, description="Seed for init, batching and rollouts.")
    seq_len: PositiveInt = Field(256, description="Sequence length for off-policy batches.")
    batch_size: PositiveInt = Field(8, description="Sequences per micro-batch.")
    max_steps: PositiveInt = Field(200, description="Optimizer steps.")
    warmup_steps: int = Field(
        0,
        ge=0,
        description=(
            "Plain-MLE warm-start steps before the configured objective takes over. "
            "MiniLLM prescribes this: a randomly-initialised student generates noise, so "
            "on-policy rollouts carry no usable signal until it can produce something. "
            "Applied identically to every objective so comparisons stay controlled."
        ),
    )
    lr: float = Field(3e-4, gt=0.0, description="Student learning rate.")
    temperature: float = Field(2.0, gt=0.0, description="Hinton temperature tau (forward KL).")
    alpha_ce: UnitFloat = Field(
        0.5, description="Weight on the hard cross-entropy term; (1-alpha) weights the KD term."
    )
    max_new_tokens: PositiveInt = Field(32, description="Rollout length for on-policy methods.")
    top_p: UnitFloat = Field(0.95, description="Nucleus cutoff for on-policy sampling.")
    length_norm: bool = Field(True, description="Length-normalize the policy-gradient objective.")
    single_step_reg: bool = Field(
        True,
        description=(
            "MiniLLM's single-step regularisation term, which reduces reward-hacking on short "
            "sequences by adding the teacher-weighted single-step KL."
        ),
    )
    baseline_ema: UnitFloat = Field(0.9, description="EMA coefficient for the PG baseline.")
    onpolicy_lr_scale: float = Field(
        0.1,
        gt=0.0,
        le=1.0,
        description=(
            "Learning-rate multiplier applied during the on-policy phase of reverse-KL "
            "distillation. A REINFORCE-style estimator is far higher-variance than a "
            "supervised one, so it needs a smaller step at the same nominal LR; MiniLLM "
            "uses a separate, much lower LR for this phase. Only affects 'reverse_kl'."
        ),
    )
    log_interval: PositiveInt = Field(10, description="Steps between log rows.")
    out_dir: str = Field("runs/distill", description="Run directory.")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field("auto", description="Compute device.")


class QuantConfig(BaseConfig):
    """Post-training quantization configuration (spec B7)."""

    method: Literal["rtn", "gptq", "awq"] = Field("gptq", description="Quantization algorithm.")
    bits: Literal[2, 3, 4, 8] = Field(4, description="Weight bit-width.")
    group_size: int = Field(
        64, description="Weights per scale/zero group along the input dim; -1 = per-channel."
    )
    symmetric: bool = Field(False, description="Asymmetric (zero-point) quantization by default.")
    calib_samples: PositiveInt = Field(128, description="Calibration sequences.")
    calib_seq_len: PositiveInt = Field(256, description="Calibration sequence length.")
    # GPTQ specifics
    damp_percent: float = Field(
        0.01, gt=0.0, lt=1.0, description="Hessian dampening as a fraction of mean(diag(H))."
    )
    act_order: bool = Field(
        True, description="Quantize columns in order of decreasing activation salience."
    )
    block_size: PositiveInt = Field(128, description="GPTQ lazy-batch column block size.")
    # AWQ specifics
    awq_grid: PositiveInt = Field(
        20, description="Grid points searched for the AWQ scale exponent."
    )
    # KV cache
    kv_bits: Literal[0, 2, 3, 4, 8] = Field(0, description="0 disables KV-cache quantization.")
    kv_group_size: PositiveInt = Field(32, description="KV quantization group along head_dim.")
    seed: int = Field(1337, description="Seed for calibration sampling.")
    out_dir: str = Field("runs/quant", description="Run directory.")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field("auto", description="Compute device.")

    @model_validator(mode="after")
    def _check_group(self) -> QuantConfig:
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be -1 (per-channel) or a positive integer.")
        return self


class SpecConfig(BaseConfig):
    """Speculative-decoding configuration (spec B8)."""

    method: Literal["draft_target", "medusa"] = Field(
        "draft_target", description="Classic draft-model speculation, or Medusa-style heads."
    )
    gamma: PositiveInt = Field(4, description="Draft tokens proposed per verification step.")
    max_new_tokens: PositiveInt = Field(128, description="Tokens to generate per request.")
    temperature: float = Field(1.0, ge=0.0, description="0 means greedy on the target.")
    top_p: UnitFloat = Field(1.0, description="Nucleus cutoff applied to the target distribution.")
    medusa_heads: PositiveInt = Field(3, description="Number of Medusa heads.")
    medusa_topk: PositiveInt = Field(3, description="Candidates per Medusa head for the tree.")
    tree_max_nodes: PositiveInt = Field(16, description="Cap on tree-attention candidate nodes.")
    seed: int = Field(1337, description="Seed for sampling and the acceptance coin flips.")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field("auto", description="Compute device.")


class GenerateConfig(BaseConfig):
    """Sampling configuration for the generation loop / server (spec Phase 10)."""

    max_new_tokens: PositiveInt = Field(128, description="Tokens to generate.")
    temperature: float = Field(0.8, ge=0.0, description="0 means greedy decoding.")
    top_k: int = Field(0, ge=0, description="0 disables top-k.")
    top_p: UnitFloat = Field(0.95, description="Nucleus cutoff; 1.0 disables top-p.")
    repetition_penalty: float = Field(1.0, ge=1.0, description="1.0 disables the penalty.")
    seed: int = Field(1337, description="Sampling seed; generation is reproducible under it.")
    stop_on_eos: bool = Field(True, description="Stop as soon as <eos> is sampled.")


class BenchConfig(BaseConfig):
    """Benchmark-harness configuration (spec Phase 10)."""

    warmup_iters: PositiveInt = Field(2, description="Unmeasured warmup iterations.")
    measure_iters: PositiveInt = Field(5, description="Measured iterations; median is reported.")
    prompt_len: PositiveInt = Field(32, description="Prompt length in tokens.")
    max_new_tokens: PositiveInt = Field(64, description="Decode length in tokens.")
    batch_size: PositiveInt = Field(
        1, description="Concurrent sequences; single-stream (1) by default."
    )
    seed: int = Field(1337, description="Seed for prompts and sampling.")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field("auto", description="Compute device.")
    out_dir: str = Field("results/bench", description="Where the benchmark table is written.")


# --------------------------------------------------------------------------------------
# Top-level experiment config
# --------------------------------------------------------------------------------------


class ExperimentConfig(BaseConfig):
    """The whole pipeline in one object; a tier preset is exactly one of these."""

    name: str = Field("nano", description="Experiment/tier name.")
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    distill: DistillConfig = Field(default_factory=DistillConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    spec: SpecConfig = Field(default_factory=SpecConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)

    @model_validator(mode="after")
    def _cross_checks(self) -> ExperimentConfig:
        if self.model.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                f"model.vocab_size={self.model.vocab_size} must equal "
                f"tokenizer.vocab_size={self.tokenizer.vocab_size}."
            )
        if self.data.seq_len > self.model.max_seq_len:
            raise ValueError(
                f"data.seq_len={self.data.seq_len} exceeds model.max_seq_len="
                f"{self.model.max_seq_len}."
            )
        return self


ALL_CONFIG_MODELS: tuple[type[BaseConfig], ...] = (
    TokenizerConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    ScheduleConfig,
    TrainConfig,
    SFTConfig,
    PreferenceConfig,
    GRPOConfig,
    AlignConfig,
    DistillConfig,
    QuantConfig,
    SpecConfig,
    GenerateConfig,
    BenchConfig,
    ExperimentConfig,
)
