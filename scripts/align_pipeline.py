"""Run the full Phase-6 alignment pipeline and write the committed comparison.

Pipeline: pretrained checkpoint → SFT → {DPO, SimPO} → scripted head-to-head evaluation
and the length-exploitation diagnostic (spec E4).

Usage::

    python scripts/align_pipeline.py runs/nano/pretrain/final.pt

Writes ``results/alignment/{summary.json, alignment.md, length_exploitation.png}``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from nanoscale.align import PreferenceTrainer, SFTTrainer
from nanoscale.align.preference import PreferenceResult
from nanoscale.data.instruct import iter_preference_pairs
from nanoscale.eval import head_to_head
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, git_sha, hardware_string, resolve_device
from nanoscale.utils.plotting import COLORS, new_figure, save_figure

log = get_logger("nanoscale.scripts.align")
RESULTS = Path("results/alignment")


def _fresh(model: NanoScaleLM) -> NanoScaleLM:
    """A deep copy, so each arm starts from exactly the same SFT weights."""
    return copy.deepcopy(model)


def plot_length_diagnostic(results: dict[str, PreferenceResult]) -> Path:
    """Plot generated-response length before and after each preference method."""
    fig, ax = new_figure(figsize=(6.4, 4.2))
    labels = list(results)
    x = range(len(labels))
    before = [results[k].generated_len_before for k in labels]
    after = [results[k].generated_len_after for k in labels]

    width = 0.36
    ax.bar([i - width / 2 for i in x], before, width, label="before", color=COLORS[5])
    ax.bar([i + width / 2 for i in x], after, width, label="after", color=COLORS[1])
    for i, (b, a) in enumerate(zip(before, after, strict=True)):
        ax.annotate(
            f"{a - b:+.1f}",
            (i + width / 2, a),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.upper() for k in labels])
    ax.set_ylabel("mean generated length (tokens)")
    ax.set_title("Length exploitation: does the objective make the model longer?")
    ax.legend()
    return save_figure(
        fig,
        RESULTS / "length_exploitation.png",
        script="scripts/align_pipeline.py",
        extra="nano tier · length-matched preference data",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Pretrained checkpoint.")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/nano.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/nano"))
    parser.add_argument("--sft-steps", type=int, default=250)
    parser.add_argument("--pref-steps", type=int, default=150)
    parser.add_argument("--eval-prompts", type=int, default=40)
    args = parser.parse_args()

    cfg = load_config_from_checkpoint(args.checkpoint)
    tok = BPETokenizer.load(args.tokenizer)
    device = resolve_device("cpu")

    base = build_model(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model=base, restore_rng=False, map_location=device)

    # --- SFT ------------------------------------------------------------------
    sft_cfg = cfg.align.sft.merged(max_steps=args.sft_steps, device="cpu")
    sft_model = _fresh(base)
    sft_result = SFTTrainer(
        sft_model,
        tok,
        sft_cfg,
        out_dir=args.runs / "sft",
        n_examples=3000,
        experiment_config=cfg,
    ).train()

    # --- preference optimization ---------------------------------------------
    pref_results: dict[str, PreferenceResult] = {}
    pref_models: dict[str, NanoScaleLM] = {}
    arms = (
        # (name, method, beta, gamma, sft_loss_weight)
        ("dpo", "dpo", 0.1, 0.0, 0.0),
        ("dpo+nll", "dpo", 0.1, 0.0, 1.0),
        ("simpo", "simpo", 2.0, 0.5, 0.0),
    )
    for name, method, beta, gamma, sft_weight in arms:
        pref_cfg = cfg.align.preference.merged(
            method=method,
            max_steps=args.pref_steps,
            beta=beta,
            gamma=gamma,
            sft_loss_weight=sft_weight,
            device="cpu",
        )
        model = _fresh(sft_model)
        run_dir = args.runs / name.replace("+", "_")
        trainer = PreferenceTrainer(
            model, tok, pref_cfg, out_dir=run_dir, n_pairs=1200, experiment_config=cfg
        )
        pref_results[name] = trainer.train()
        pref_models[name] = model

    # --- scripted head-to-head ------------------------------------------------
    pairs = list(iter_preference_pairs(seed=98765, n=args.eval_prompts))
    head_to_heads = {
        method: head_to_head(
            sft_model,
            model,
            tok,
            pairs,
            n_prompts=args.eval_prompts,
            label_a="sft",
            label_b=method,
        ).summary()
        for method, model in pref_models.items()
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "git_sha": git_sha(),
        "hardware": hardware_string(),
        "checkpoint": str(args.checkpoint),
        "sft": sft_result.summary(),
        "preference": {k: v.summary() for k, v in pref_results.items()},
        "head_to_head": head_to_heads,
    }
    (RESULTS / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    figure = plot_length_diagnostic(pref_results)

    dpo, dpo_nll, simpo = (
        pref_results["dpo"],
        pref_results["dpo+nll"],
        pref_results["simpo"],
    )

    def logp_drift(result: PreferenceResult) -> tuple[float, float]:
        """Change in mean per-token chosen/rejected log-probability over the run."""
        rows = [r for r in result.history if "chosen_logp" in r]
        if len(rows) < 2:
            return 0.0, 0.0
        return (
            rows[-1]["chosen_logp"] - rows[0]["chosen_logp"],
            rows[-1]["rejected_logp"] - rows[0]["rejected_logp"],
        )

    lines = [
        "# Alignment, SFT, DPO and SimPO",
        "",
        f"Generated by `scripts/align_pipeline.py` at git `{payload['git_sha']}` from "
        f"`{args.checkpoint}`.",
        "",
        "## Pipeline",
        "",
        f"- **SFT** ({sft_result.steps} steps): completion-masked loss "
        f"{sft_result.final_loss:.4f} (held-out {sft_result.best_val_loss:.4f}).",
        f"- **DPO** ({dpo.steps} steps): reward margin {dpo.final_margin:+.4f}, "
        f"preference accuracy {dpo.final_accuracy:.1%}.",
        f"- **DPO + NLL** ({dpo_nll.steps} steps): reward margin {dpo_nll.final_margin:+.4f}, "
        f"preference accuracy {dpo_nll.final_accuracy:.1%}.",
        f"- **SimPO** ({simpo.steps} steps): reward margin {simpo.final_margin:+.4f}, "
        f"preference accuracy {simpo.final_accuracy:.1%}.",
        "",
        "## The likelihood-collapse diagnostic",
        "",
        "DPO optimises the *difference* of two log-probabilities, so it can reduce its "
        "loss by pushing **both** down, the chosen response merely less far than the "
        "rejected one. A run showing a healthy rising margin can be quietly destroying "
        "the model's absolute likelihood of good responses at the same time. This table "
        "reports the change in mean **per-token** log-probability across the run:",
        "",
        "| method | Δ log p(chosen) | Δ log p(rejected) | Δ margin |",
        "|---|---|---|---|",
    ]
    for name, result in pref_results.items():
        dc, dr = logp_drift(result)
        lines.append(f"| {name.upper()} | {dc:+.4f} | {dr:+.4f} | {dc - dr:+.4f} |")

    lines += [
        "",
        "`DPO + NLL` adds an auxiliary negative-log-likelihood term on the chosen "
        "response (the RPO-style fix, `align.preference.sft_loss_weight`), which anchors "
        "the absolute likelihood so the objective cannot satisfy itself by pushing "
        "everything down. It is off by default so that its effect is measured here "
        "rather than assumed.",
        "",
        "## Head-to-head vs the SFT model",
        "",
        "| aligned model | wins | losses | ties | mean judge score (SFT → aligned) |",
        "|---|---|---|---|---|",
    ]
    for method, h2h in head_to_heads.items():
        lines.append(
            f"| {method.upper()} | {h2h['wins_b']} | {h2h['wins_a']} | {h2h['ties']} | "
            f"{h2h['mean_score_a']:.3f} → {h2h['mean_score_b']:.3f} |"
        )

    lines += [
        "",
        "The judge is programmatic and stated in `src/nanoscale/eval/preference_eval.py`: "
        "on-topic overlap with the prompt, absence of degenerate repetition, and whether "
        "the model emitted `<eot>` rather than running to the token cap. Those are exactly "
        "the properties the preference labels encode, so this measures *did the model "
        "learn the labels*, not *is the model good*. It is deliberately length-insensitive, "
        "so a model that learned to game DPO's length bias gains nothing from it.",
        "",
        "## Length exploitation (spec E4)",
        "",
        f"![length exploitation]({figure.name})",
        "",
        "| method | mean generated length before | after | change |",
        "|---|---|---|---|",
        *[
            f"| {name.upper()} | {r.generated_len_before:.1f} | {r.generated_len_after:.1f} | "
            f"{r.generated_len_after - r.generated_len_before:+.1f} |"
            for name, r in pref_results.items()
        ],
        "",
        "DPO's implicit reward is a **sum** of per-token log-ratios, so a longer response "
        "has more terms to accumulate advantage over and the objective can be reduced by "
        "lengthening rather than improving. SimPO divides by response length, turning the "
        "reward into an average and removing that incentive; it is also reference-free, "
        "so it never allocates the frozen second copy of the model at all.",
        "",
        "The preference data here is **length-matched by construction** (mean chosen and "
        "rejected lengths are within 5%, asserted by a test), so any length drift after "
        "training comes from the objective rather than from the labels.",
        "",
        "## Caveats",
        "",
        "Single seed, ~5M parameters, synthetic instruction data, and a programmatic judge. "
        "These are mechanism demonstrations; the DPO/SimPO losses are implemented from the "
        "papers and unit-tested against hand-computed values: not evidence about how these "
        "methods rank on real preference data at real scale.",
        "",
        "Reproduce with: `python scripts/align_pipeline.py runs/nano/pretrain/final.pt`",
        "",
    ]
    (RESULTS / "alignment.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["head_to_head"], indent=2))
    print(f"wrote {RESULTS}/summary.json, alignment.md, {figure.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
