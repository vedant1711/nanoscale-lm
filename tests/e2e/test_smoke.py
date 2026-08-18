"""The end-to-end smoke test (spec D3).

Runs the **entire pipeline** at `nano` tier on CPU, tokenizer → pretrain → SFT → DPO →
quantize → speculative decode → evaluate, and asserts a sanity metric at every stage.
The spec's requirement is that this completes in under 10 minutes on a CPU with no GPU
and no network.

Why this exists on top of 500 unit tests: every stage is individually correct and the
pipeline can still be broken at the seams. A tokenizer whose vocabulary does not match
the model's, a checkpoint that does not carry its config, an aligned model the quantizer
cannot load, none of those show up in a unit test of the component.

Run it directly with ``make smoke``, or as part of the suite with
``pytest -m slow tests/e2e``.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import pytest
import torch

from nanoscale.align import PreferenceTrainer, SFTTrainer
from nanoscale.config import TokenizerConfig, draft_model_config, load_experiment
from nanoscale.data.toy import generate_corpus
from nanoscale.eval import perplexity, run_tiny_bench
from nanoscale.model import build_model
from nanoscale.quantize import GPTQQuantizer, quantize_rtn
from nanoscale.serve import generate_text
from nanoscale.specdec import SpeculativeSampler, autoregressive_baseline
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import TokenBatcher, Trainer, build_packed_tokens

pytestmark = pytest.mark.slow

#: The spec's budget. Generous relative to the observed ~3 minutes, because CI runners
#: are slower and more variable than a laptop, but tight enough to catch a regression
#: that makes the pipeline unusable on free hardware.
BUDGET_SECONDS = 600.0


def test_full_pipeline_on_cpu(tmp_path: Path) -> None:
    """Tokenizer -> pretrain -> SFT -> DPO -> quantize -> speculate -> evaluate."""
    started = time.perf_counter()
    stages: dict[str, float] = {}

    def mark(name: str) -> None:
        stages[name] = round(time.perf_counter() - started, 1)

    # ---------------------------------------------------------------- 1. tokenizer
    corpus = generate_corpus(seed=1337, n_stories=4000)
    tok = BPETokenizer.train(corpus, TokenizerConfig(vocab_size=1024, max_train_bytes=1_200_000))
    assert tok.n_merges == tok.config.n_merges, "the toy corpus must fill the nano vocabulary"
    assert tok.decode(tok.encode("Lily went to the park.")) == "Lily went to the park."
    assert tok.compression_ratio(corpus[:20_000]) > 3.0
    mark("tokenizer")

    # ---------------------------------------------------------------- 2. pretrain
    cfg = load_experiment(
        tier="nano",
        overrides=[
            "train.device=cpu",
            "train.max_steps=120",
            "train.token_budget=null",
            "train.eval_interval=60",
            "train.log_interval=20",
            "train.ckpt_interval=100000",
        ],
    )
    trainer = Trainer(cfg, tokenizer=tok, out_dir=tmp_path / "pretrain")
    pretrain = trainer.train()

    assert pretrain.history[0]["loss"] == pytest.approx(math.log(cfg.model.vocab_size), abs=0.05)
    assert pretrain.final_val_loss < 2.0, f"loss barely moved: {pretrain.final_val_loss}"
    assert (tmp_path / "pretrain" / "manifest.json").exists()
    base = trainer.model
    mark("pretrain")

    # ---------------------------------------------------------------- 3. SFT
    sft_cfg = cfg.align.sft.merged(max_steps=60, device="cpu", out_dir=str(tmp_path / "sft"))
    sft_trainer = SFTTrainer(
        base, tok, sft_cfg, out_dir=tmp_path / "sft", n_examples=600, experiment_config=cfg
    )
    before_sft = sft_trainer.evaluate()
    sft = sft_trainer.train()
    assert sft.final_loss < before_sft, "SFT did not reduce the completion-masked loss"
    mark("sft")

    # ---------------------------------------------------------------- 4. DPO
    dpo_cfg = cfg.align.preference.merged(
        method="dpo", max_steps=40, log_interval=10, device="cpu", sft_loss_weight=1.0
    )
    dpo_trainer = PreferenceTrainer(
        base, tok, dpo_cfg, out_dir=tmp_path / "dpo", n_pairs=200, experiment_config=cfg
    )
    dpo = dpo_trainer.train()
    margins = [row["reward_margin"] for row in dpo.history if "reward_margin" in row]
    assert margins[-1] > margins[0], f"DPO did not raise the reward margin: {margins}"
    assert dpo.final_accuracy >= 0.5
    mark("dpo")

    # ---------------------------------------------------------------- 5. evaluate
    data = build_packed_tokens(cfg.data, tok)
    eval_batches = TokenBatcher(
        data.val, seq_len=cfg.data.seq_len, batch_size=4, shuffle=False
    ).take(8)
    base_ppl = perplexity(base, eval_batches)
    bench = run_tiny_bench(base, tok)
    assert base_ppl.perplexity < 8.0
    assert bench.accuracy >= bench.chance, "the model should beat chance on the tiny benchmark"
    mark("evaluate")

    # ---------------------------------------------------------------- 6. quantize
    rtn_model = copy.deepcopy(base)
    quantize_rtn(rtn_model, bits=4, group_size=64)
    rtn_ppl = perplexity(rtn_model, eval_batches)

    gptq_model = copy.deepcopy(base)
    calib = TokenBatcher(data.train, seq_len=cfg.data.seq_len, batch_size=4, seed=1337).take(4)
    quantizer = GPTQQuantizer(gptq_model, bits=4, group_size=64, act_order=True)
    quantizer.collect([b.inputs for b in calib])
    quantizer.apply()
    gptq_ppl = perplexity(gptq_model, eval_batches)

    # 4-bit must stay usable, and GPTQ must not be *worse* than RTN.
    assert gptq_ppl.perplexity < base_ppl.perplexity * 1.5, "4-bit destroyed the model"
    assert gptq_ppl.perplexity <= rtn_ppl.perplexity * 1.02, (
        f"GPTQ {gptq_ppl.perplexity:.4f} lost to RTN {rtn_ppl.perplexity:.4f}"
    )

    text = generate_text(gptq_model, tok, "It was a sunny day. Lily went to").text
    assert len(text.strip()) > 0, "the 4-bit model produced nothing"
    mark("quantize")

    # ---------------------------------------------------------------- 7. speculate
    # The draft here is *untrained*, training one would blow the time budget, and the
    # measured acceptance rate is reported by `scripts/specdec_bench.py` against the
    # distilled draft, not here. So the bar is only "the wiring works and at least some
    # proposals survive the accept rule"; the real assertion is losslessness below.
    torch.manual_seed(1337)
    draft = build_model(draft_model_config(cfg.model))
    prompt = torch.tensor([tok.encode("It was a sunny day. Lily went to", add_bos=True)])

    sampler = SpeculativeSampler(base, draft, gamma=4, temperature=1.0)
    spec = sampler.generate(prompt, max_new_tokens=32, generator=torch.Generator().manual_seed(1))
    assert spec.generated == 32
    assert spec.mean_accepted_length > 1.0, (
        f"acceptance length {spec.mean_accepted_length:.2f} <= 1: every proposal was rejected, "
        "which means the draft/target plumbing is broken rather than merely unhelpful"
    )

    # Losslessness: greedy speculation must equal greedy autoregressive, exactly.
    greedy_spec = SpeculativeSampler(base, draft, gamma=4, temperature=0.0).generate(
        prompt, max_new_tokens=24
    )
    greedy_base = autoregressive_baseline(base, prompt, max_new_tokens=24, temperature=0.0)
    assert torch.equal(greedy_spec.tokens, greedy_base.tokens), (
        "greedy speculative decoding diverged from greedy autoregressive decoding"
    )
    mark("speculate")

    # ---------------------------------------------------------------- budget
    total = time.perf_counter() - started
    (tmp_path / "smoke.json").write_text(
        json.dumps(
            {
                "total_s": round(total, 1),
                "stages_cumulative_s": stages,
                "pretrain_val_loss": round(pretrain.final_val_loss, 4),
                "base_perplexity": round(base_ppl.perplexity, 4),
                "rtn4_perplexity": round(rtn_ppl.perplexity, 4),
                "gptq4_perplexity": round(gptq_ppl.perplexity, 4),
                "tiny_bench_accuracy": round(bench.accuracy, 4),
                "mean_accepted_length": round(spec.mean_accepted_length, 4),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert total < BUDGET_SECONDS, (
        f"the smoke pipeline took {total:.0f}s, over the {BUDGET_SECONDS:.0f}s budget. "
        f"Stage timings (cumulative): {stages}"
    )
