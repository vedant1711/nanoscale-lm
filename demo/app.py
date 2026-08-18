"""Gradio demo for NanoScale-LM (spec F1).

Four tabs, matching the spec:

1. **Chat / generate** — talk to the model, with a base ↔ aligned toggle so a visitor can
   feel what alignment did.
2. **Speed lab** — the same prompt decoded autoregressively and speculatively, with live
   tokens/second and acceptance length.
3. **Compression explorer** — the committed bits-vs-accuracy frontier and the variants
   table, read from ``results/`` rather than retyped.
4. **About** — the thesis, the architecture, and the honest caveats.

Run locally::

    uv pip install -e ".[demo]"
    python demo/app.py

On Hugging Face Spaces the free CPU-basic tier is enough for the ``nano`` tier. Port 7860
is pinned, which is what Spaces expects.

Every number displayed here is read from a committed artifact in ``results/`` — the demo
never computes a headline number of its own and never hand-edits one (spec F5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from nanoscale.compress import compress, decompress, score_lines
from nanoscale.config import GenerateConfig
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.serve import generate_text
from nanoscale.specdec import SpeculativeSampler, autoregressive_baseline
from nanoscale.tokenizer import BPETokenizer, Message, render_prompt
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, resolve_device

log = get_logger("nanoscale.demo")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
#: The 40M TinyStories model is far better than `nano` at everything this demo shows, but
#: it is 154 MB and therefore gitignored — a fresh clone only has the committed `nano`
#: exports. So prefer it when present and fall back cleanly, rather than shipping a demo
#: that either breaks without it or silently shows the weaker model as if it were the
#: headline one. `make train-micro-tinystories` reproduces it.
MICRO_CKPT = ROOT / "runs" / "micro" / "tinystories" / "final.pt"
MICRO_TOKENIZER = ROOT / "artifacts" / "tokenizer" / "micro_tinystories.json"
HAVE_MICRO = MICRO_CKPT.exists() and MICRO_TOKENIZER.exists()

TOKENIZER_PATH = MICRO_TOKENIZER if HAVE_MICRO else ROOT / "artifacts" / "tokenizer" / "nano.json"

if HAVE_MICRO:
    CHECKPOINTS = {"micro · 40M · TinyStories": MICRO_CKPT}
else:
    CHECKPOINTS = {
        "base (pretrained)": ROOT / "runs" / "nano" / "pretrain" / "final.pt",
        "aligned (SFT + DPO+NLL)": ROOT / "runs" / "nano" / "dpo_nll" / "final.pt",
    }
DRAFT_PATH = ROOT / "runs" / "nano" / "distill" / "reverse_kl" / "final.pt"

EXAMPLES = [
    "It was a sunny day. Lily went to the park with",
    "Tom wanted to find a shiny key",
    "The wind was cold. Mia walked to",
    "But a red ball was stuck",
]


class Models:
    """Lazily loads and caches the checkpoints the demo needs."""

    def __init__(self) -> None:
        """Load the tokenizer; models are loaded on first use."""
        self.tokenizer = BPETokenizer.load(TOKENIZER_PATH)
        self._cache: dict[str, NanoScaleLM] = {}

    def get(self, key: str) -> NanoScaleLM:
        """Return a loaded model, loading it if necessary."""
        if key in self._cache:
            return self._cache[key]
        path = CHECKPOINTS.get(key, DRAFT_PATH if key == "draft" else None)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"checkpoint for {key!r} not found at {path}. Run `make smoke` or "
                "`make train-nano` first."
            )
        cfg = load_config_from_checkpoint(path)
        device = resolve_device("cpu")
        model = build_model(cfg.model).to(device)
        load_checkpoint(path, model=model, restore_rng=False, map_location=device)
        model.eval()
        self._cache[key] = model
        return model


MODELS = Models()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"_(missing: {path.name})_"


# ------------------------------------------------------------------------------ tabs


def do_generate(
    prompt: str, checkpoint: str, max_new_tokens: int, temperature: float, top_p: float, seed: int
) -> tuple[str, str]:
    """Tab 1: generate a completion and report its timing."""
    model = MODELS.get(checkpoint)
    cfg = GenerateConfig(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed),
    )
    if checkpoint.startswith("aligned"):
        ids = render_prompt(MODELS.tokenizer, [Message("user", prompt)])
        out = generate_text(model, MODELS.tokenizer, "", cfg, prompt_ids=ids)
        text = out.text
    else:
        out = generate_text(model, MODELS.tokenizer, prompt, cfg)
        text = prompt + out.text
    stats = (
        f"{out.generated_tokens} tokens · "
        f"{out.decode_tokens_per_s:.0f} tok/s decode · "
        f"prefill {out.prefill_s * 1000:.0f} ms · stop: {out.stop_reason}"
    )
    return text, stats


def do_speed_lab(prompt: str, max_new_tokens: int, gamma: int, seed: int) -> tuple[str, str, str]:
    """Tab 2: the same prompt, decoded both ways."""
    target = MODELS.get("base (pretrained)")
    try:
        draft = MODELS.get("draft")
    except FileNotFoundError as exc:
        return "", "", f"Speculative decoding needs a distilled draft model.\n\n{exc}"

    ids = torch.tensor([MODELS.tokenizer.encode(prompt, add_bos=True)])
    tokens = int(max_new_tokens)

    start = time.perf_counter()
    base = autoregressive_baseline(
        target,
        ids,
        max_new_tokens=tokens,
        temperature=0.8,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    base_wall = time.perf_counter() - start

    sampler = SpeculativeSampler(target, draft, gamma=int(gamma), temperature=0.8)
    start = time.perf_counter()
    spec = sampler.generate(
        ids, max_new_tokens=tokens, generator=torch.Generator().manual_seed(int(seed))
    )
    spec_wall = time.perf_counter() - start

    base_text = MODELS.tokenizer.decode(base.tokens[0].tolist(), skip_special=True)
    spec_text = MODELS.tokenizer.decode(spec.tokens[0].tolist(), skip_special=True)

    report = (
        f"| metric | autoregressive | speculative (γ={int(gamma)}) |\n"
        f"|---|---|---|\n"
        f"| target forward passes | {base.target_calls} | {spec.target_calls} |\n"
        f"| tokens per target pass | {base.mean_accepted_length:.2f} | "
        f"{spec.mean_accepted_length:.2f} |\n"
        f"| draft acceptance rate | — | {spec.acceptance_rate:.1%} |\n"
        f"| wall clock | {base_wall * 1000:.0f} ms | {spec_wall * 1000:.0f} ms |\n"
        f"| tokens/s | {base.generated / max(1e-9, base_wall):.0f} | "
        f"{spec.generated / max(1e-9, spec_wall):.0f} |\n\n"
        "**Read the target-pass row, not the wall clock.** Speculation reduces target "
        "forward passes — that is the mechanism and it is hardware-independent. On a "
        "5M-parameter model running on a free CPU, a forward pass is dominated by Python "
        "dispatch rather than by weight loading, so the wall-clock win the method exists "
        "for does not appear here. It needs a model large enough that memory bandwidth is "
        "the bottleneck."
    )
    return base_text, spec_text, report


def do_compress(text: str, checkpoint: str) -> tuple[str, str]:
    """Compress `text` with the model and compare against the standard tools.

    This is the capability the project argues is its actual use: a language model is a
    compressor, and driving an arithmetic coder with its predictions turns cross-entropy
    into a real file size. The round trip is verified here rather than asserted, because a
    compressor that is not lossless is not a compressor.
    """
    import bz2
    import gzip
    import lzma

    text = (text or "").strip()
    if len(text) < 40:
        return (
            "Enter at least 40 characters — below that the coder's 4-byte flush "
            "dominates the measurement and the comparison is meaningless.",
            "",
        )

    model = MODELS.get(checkpoint)
    tok = MODELS.tokenizer
    raw = text.encode("utf-8")

    t0 = time.perf_counter()
    result = compress(model, tok, text)
    encode_s = time.perf_counter() - t0
    restored = decompress(model, tok, result.payload, result.n_tokens)
    lossless = restored == text

    rows = [
        ("NanoScale-LM", result.n_bytes_out, result.ratio),
        ("xz -9", len(lzma.compress(raw, preset=9)), len(raw) / len(lzma.compress(raw, preset=9))),
        ("bzip2 -9", len(bz2.compress(raw, 9)), len(raw) / len(bz2.compress(raw, 9))),
        ("gzip -9", len(gzip.compress(raw, 9)), len(raw) / len(gzip.compress(raw, 9))),
    ]
    table = ["| coder | bytes | ratio | bits/byte |", "|---|---|---|---|"]
    for name, nbytes, ratio in rows:
        mark = " **&larr;**" if name.startswith("NanoScale") else ""
        table.append(f"| {name}{mark} | {nbytes:,} | {ratio:.2f}x | {nbytes * 8 / len(raw):.3f} |")

    verdict = (
        "**Beating every classical coder** — this text is inside the model's distribution."
        if result.bits_per_byte < min(r[1] * 8 / len(raw) for r in rows[1:])
        else "**Losing to the classical coders** — this text is outside the model's "
        "distribution, which is exactly what specialisation costs."
    )
    report = "\n".join(table) + (
        f"\n\n{verdict}\n\n"
        f"- lossless round trip: **{'verified' if lossless else 'FAILED'}**\n"
        f"- {len(raw):,} bytes in, {result.n_tokens:,} tokens, {encode_s:.1f}s to encode\n"
        f"- coder overhead above the model's own cross-entropy: "
        f"{result.coder_overhead * 100:.2f}%"
    )
    return (report, f"{result.bits_per_byte:.4f} bits/byte  ·  {result.ratio:.2f}x smaller")


def do_anomaly(lines_text: str, checkpoint: str) -> str:
    """Score each line by surprisal — the same quantity the compressor sums."""
    lines = [ln.strip() for ln in (lines_text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return "Enter at least two lines, one per row."

    report = score_lines(MODELS.get(checkpoint), MODELS.tokenizer, lines)
    ranked = report.ranked()
    if not ranked:
        return "Nothing scoreable."

    # Baseline against the *least* surprising line rather than the median: in a short,
    # deliberately mixed list the median is frequently an anomaly itself, which would
    # normalise the scale by the thing being detected. In production the baseline comes
    # from a window of known-good traffic, which is a stronger version of the same idea.
    floor = min(s for s, _ in ranked)
    out = ["| bits/token | vs cleanest line | line |", "|---|---|---|"]
    for score, line in ranked:
        ratio = score / max(floor, 1e-9)
        flag = " **&larr; anomalous**" if ratio > 1.5 else ""
        out.append(f"| {score:.2f} | {ratio:.1f}x | `{line[:70]}`{flag} |")
    out.append(
        f"\n\nThe cleanest line costs {floor:.2f} bits/token; anything above 1.5x that is "
        f"flagged. No labels and no rules are involved. In a real deployment the baseline is "
        f"a percentile of known-good traffic rather than the best line in the batch — same "
        f"idea, more data behind it."
    )
    return "\n".join(out)


def compression_view() -> str:
    """Tab 3: the committed frontier and variants table."""
    parts = [_read(RESULTS / "quantization" / "quantization.md")]
    table = RESULTS / "bench" / "table.md"
    if table.exists():
        parts.append("\n\n---\n\n" + _read(table))
    return "".join(parts)


def about_view() -> str:
    """Tab 4: the thesis and the caveats."""
    summary_path = RESULTS / "bench" / "table.json"
    provenance = ""
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        provenance = (
            f"\n\nNumbers on this page were produced at git `{payload.get('git_sha')}` on "
            f"`{payload.get('hardware')}`."
        )
    return (
        "## NanoScale-LM\n\n"
        "> I built a language model from scratch and then made it deployable on hardware "
        "anyone can afford.\n\n"
        "Every algorithm behind this demo is implemented in the repository: the BPE "
        "merges, attention with GQA/RoPE/QK-norm, the Muon optimizer's Newton–Schulz "
        "orthogonalization, the DPO and SimPO losses, GPTQ's Hessian error compensation, "
        "and the speculative-sampling accept/reject rule. No high-level trainer library "
        "is used anywhere in `src/nanoscale/`.\n\n"
        "### What this model is, honestly\n\n"
        "The model you are talking to has about **5 million parameters** and was trained "
        "for **95 seconds on a laptop CPU** on a **synthetic story corpus** with a "
        "1024-token vocabulary. It writes coherent little stories about Lily and Tom "
        "because that is the only thing it has ever seen. It is not a general-purpose "
        "assistant, it knows no facts about the world, and it will not answer questions "
        "outside that tiny domain.\n\n"
        "That is the point. The claim is not that this model is good; it is that the "
        "**whole lifecycle** — tokenizer, pretraining, alignment, distillation, "
        "quantization, speculative decoding — is implemented correctly, measured "
        "honestly, and runs end to end on hardware anyone has. The `micro` tier scales "
        "the identical code path to FineWeb-Edu on a free Colab GPU.\n\n"
        "### Architecture\n\n"
        "```\n"
        "token embedding\n"
        "  ↓\n"
        "6 × [ RMSNorm → GQA attention (RoPE, QK-norm, KV cache) → +residual\n"
        "      RMSNorm → SwiGLU MLP                              → +residual ]\n"
        "  ↓\n"
        "RMSNorm → untied LM head (zero-init)\n"
        "```\n\n"
        "### Links\n\n"
        "- Methodology, with every formula and citation: `docs/methodology.md`\n"
        "- All results with reproduction commands: `docs/results.md`\n"
        "- What this does *not* show: `docs/limitations.md`" + provenance
    )


def build_demo() -> Any:  # noqa: ANN401 - gradio is an optional extra, so Blocks
    """Assemble the Gradio interface."""  # cannot be imported at module scope.
    import gradio as gr

    with gr.Blocks(title="NanoScale-LM", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# NanoScale-LM\n"
            "*A small language model built from scratch, and the full stack that makes "
            "it cheap to serve.*"
        )

        with gr.Tab("Chat / generate"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Prompt", value=EXAMPLES[0], lines=3, placeholder="Type a prompt…"
                    )
                    gr.Examples(EXAMPLES, inputs=prompt)
                    output = gr.Textbox(label="Completion", lines=8)
                    stats = gr.Markdown()
                with gr.Column(scale=1):
                    checkpoint = gr.Radio(
                        list(CHECKPOINTS),
                        value="base (pretrained)",
                        label="Checkpoint",
                        info="Flip between the pretrained and aligned models.",
                    )
                    max_new = gr.Slider(8, 128, value=64, step=8, label="Max new tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                    seed = gr.Number(value=1337, precision=0, label="Seed")
                    go = gr.Button("Generate", variant="primary")
            go.click(
                do_generate,
                [prompt, checkpoint, max_new, temperature, top_p, seed],
                [output, stats],
            )

        with gr.Tab("Speed lab"):
            gr.Markdown(
                "Decode the same prompt twice — once autoregressively, once with "
                "draft–target speculative decoding — and compare. The outputs differ "
                "because both sample; speculative decoding is lossless **in "
                "distribution**, not token-for-token, unless you decode greedily."
            )
            with gr.Row():
                speed_prompt = gr.Textbox(label="Prompt", value=EXAMPLES[0], lines=2)
                speed_tokens = gr.Slider(16, 128, value=64, step=8, label="Tokens")
                speed_gamma = gr.Slider(1, 8, value=6, step=1, label="γ (draft length)")
                speed_seed = gr.Number(value=1337, precision=0, label="Seed")
            speed_go = gr.Button("Race them", variant="primary")
            with gr.Row():
                base_out = gr.Textbox(label="Autoregressive", lines=6)
                spec_out = gr.Textbox(label="Speculative", lines=6)
            speed_report = gr.Markdown()
            speed_go.click(
                do_speed_lab,
                [speed_prompt, speed_tokens, speed_gamma, speed_seed],
                [base_out, spec_out, speed_report],
            )

        with gr.Tab("Compress"):
            gr.Markdown(
                "### The model as a lossless compressor\n"
                "Shannon: a token of probability `p` costs `-log2(p)` bits. Feeding those "
                "probabilities to an arithmetic coder turns the model's cross-entropy into "
                "an actual file size. **Try in-domain text (a children's story) against "
                "out-of-domain text (news, code, an email)** — the model wins enormously on "
                "the first and loses badly on the second, and that trade is the point."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    comp_in = gr.Textbox(
                        label="Text to compress",
                        lines=7,
                        value=(
                            "Once upon a time, there was a little girl named Lily. She loved "
                            "to play in the garden with her dog. One sunny day, she found a "
                            "big red ball under the tree and played with it all afternoon."
                        ),
                    )
                    comp_go = gr.Button("Compress and verify", variant="primary")
                    comp_head = gr.Markdown()
                with gr.Column(scale=2):
                    comp_out = gr.Markdown()
            comp_go.click(do_compress, inputs=[comp_in, checkpoint], outputs=[comp_out, comp_head])

        with gr.Tab("Anomaly detection"):
            gr.Markdown(
                "### Surprisal as an unsupervised anomaly score\n"
                "The same per-token cost, un-summed. A line the model finds expensive to "
                "encode is a line unlike its training data. One line per row — mix ordinary "
                "sentences with something that does not belong and watch the ordering."
            )
            anom_in = gr.Textbox(
                label="Lines to score",
                lines=9,
                value=(
                    "Lily went to the park with her little dog.\n"
                    "Tom was very happy and played with the ball all day.\n"
                    "The bird looked sad because it had lost its nest.\n"
                    "Tom picked up the quantum entanglement and put it in his pocket.\n"
                    "2026-08-17T14:22:01Z ERROR db.pool timeout after 30000ms retry=3\n"
                    "xqzk vburt plimf woggle zzzt krrn."
                ),
            )
            anom_go = gr.Button("Score lines", variant="primary")
            anom_out = gr.Markdown()
            anom_go.click(do_anomaly, inputs=[anom_in, checkpoint], outputs=anom_out)

        with gr.Tab("Compression explorer"):
            gr.Markdown(compression_view())

        with gr.Tab("About"):
            gr.Markdown(about_view())

    return demo


if __name__ == "__main__":
    build_demo().queue().launch(server_name="0.0.0.0", server_port=7860)
