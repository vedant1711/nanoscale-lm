r"""Precompute real model outputs for the static browser demo.

The demo has to run on GitHub Pages, which serves static files and cannot execute a
PyTorch model. The honest way to build an interactive demo under that constraint is to
run the *real* checkpoint here, exhaustively, over every input the page can offer, and
ship the outputs as JSON. Nothing on the page is invented or interpolated: every token,
probability, byte count and surprisal score in `demo/demo_data.json` came out of
`runs/micro/tinystories/final.pt`.

What that buys, and what it costs:

* The visitor sees genuine model behaviour, generations at several temperatures and
  seeds, real arithmetic-coding byte counts, real per-token surprisal, with no server,
  no cold start and no cost.
* They cannot type a free-form prompt. The page says so plainly rather than pretending
  otherwise; `demo/app.py` remains the live version for anyone who clones the repo.

Usage::

    python scripts/build_demo_data.py runs/micro/tinystories/final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nanoscale.compress import compress, score_lines, token_surprisal
from nanoscale.config import GenerateConfig
from nanoscale.model import build_model
from nanoscale.serve import generate_text
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, git_sha, hardware_string, resolve_device

log = get_logger("nanoscale.demo-data")
ROOT = _ROOT

PROMPTS = [
    "Once upon a time there was a little girl named Mia who",
    "Tom found a shiny red box in the garden. When he opened it,",
    "The sun was setting when Ben walked home. On the way he saw",
    "Lily and her brother wanted to build something. They decided to",
    "It was raining hard. The cat sat by the window and",
]
TEMPERATURES = [0.2, 0.6, 0.8, 1.0]
SEEDS = [1, 2, 3]

#: Lines the visitor can score for anomalousness. Mixed on purpose: the point is that the
#: ordering is meaningful, not that obvious gibberish scores badly.
PROBE_LINES = [
    ("in-domain", "Lily went to the park with her little dog."),
    ("in-domain", "Tom was very happy and played with the ball all day."),
    ("in-domain", "The bird looked sad because it had lost its nest."),
    ("in-domain", "She opened the box and found a shiny red key inside."),
    ("plausible", "The dog sat quietly under the old wooden table."),
    ("odd but grammatical", "Tom picked up the quantum entanglement and put it in his pocket."),
    ("odd but grammatical", "Lily ate the concept of Tuesday for breakfast."),
    ("wrong register", "Pursuant to section 4(b), the party of the first part shall indemnify."),
    ("wrong domain", "The mitochondrion generates ATP via oxidative phosphorylation."),
    ("code", "for (int i = 0; i < n; i++) { buf[i] = malloc(sizeof(node)); }"),
    ("log line", "2026-08-17T14:22:01Z ERROR db.pool timeout after 30000ms retry=3"),
    ("gibberish", "xqzk vburt plimf woggle zzzt krrn."),
]

COMPRESS_SAMPLES = [
    (
        "A TinyStories passage (in domain)",
        "Once upon a time, there was a little girl named Lily. She loved to play in the "
        "garden with her dog. One sunny day, she found a big red ball under the tree. "
        "Lily was very happy and played with the ball all afternoon.",
    ),
    (
        "English prose the model never saw",
        "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic "
        "organisms. Mitochondria generate most of the cell's supply of adenosine "
        "triphosphate, subsequently used throughout the cell as chemical energy.",
    ),
    (
        "Source code",
        "def quicksort(items):\n    if len(items) <= 1:\n        return items\n"
        "    pivot = items[len(items) // 2]\n    left = [x for x in items if x < pivot]\n"
        "    return quicksort(left) + [pivot]",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, nargs="?", default=ROOT / "runs/micro/tinystories/final.pt"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "artifacts/tokenizer/micro_tinystories.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "demo" / "demo_data.json")
    args = parser.parse_args()

    device = resolve_device("cpu")
    cfg = load_config_from_checkpoint(args.checkpoint)
    model = build_model(cfg.model)
    state, _ = load_checkpoint(args.checkpoint, model=model, restore_rng=False)
    model.to(device).eval()
    tok = BPETokenizer.load(args.tokenizer)
    log.info("loaded %s params, step %s", f"{model.num_parameters():,}", state.step)

    payload: dict[str, Any] = {
        "git_sha": git_sha(),
        "hardware": hardware_string(),
        "params": model.num_parameters(),
        "vocab_size": cfg.model.vocab_size,
        "train_tokens": state.tokens,
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
    }

    # ---------------------------------------------------------------- generation
    gens: list[dict[str, Any]] = []
    with torch.no_grad():
        for prompt in PROMPTS:
            for temp in TEMPERATURES:
                for seed in SEEDS:
                    out = generate_text(
                        model,
                        tok,
                        prompt,
                        GenerateConfig(max_new_tokens=90, temperature=temp, top_p=0.95, seed=seed),
                    )
                    gens.append(
                        {
                            "prompt": prompt,
                            "temperature": temp,
                            "seed": seed,
                            "completion": out.text.strip(),
                        }
                    )
        log.info("generated %d completions", len(gens))
        payload["generations"] = gens

        # ------------------------------------------------------------ surprisal
        report = score_lines(model, tok, [ln for _, ln in PROBE_LINES], device=device)
        payload["anomaly"] = {
            "lines": [
                {"kind": kind, "line": line, "bits_per_token": round(score, 3)}
                for (kind, line), score in zip(PROBE_LINES, report.scores, strict=True)
            ],
        }

        # Per-token surprisal for one sentence, so the page can colour each token.
        payload["token_surprisal"] = [
            {
                "sentence": s,
                "tokens": [
                    {"t": t, "bits": round(b, 3)} for t, b in token_surprisal(model, tok, s)
                ],
            }
            for s in [
                "Lily went to the park with her little dog.",
                "Tom picked up the quantum entanglement and put it in his pocket.",
            ]
        ]

        # ------------------------------------------------------------ compression
        import bz2
        import gzip
        import lzma

        comp: list[dict[str, Any]] = []
        for label, text in COMPRESS_SAMPLES:
            raw = text.encode("utf-8")
            r = compress(model, tok, text, device=device)
            comp.append(
                {
                    "label": label,
                    "text": text,
                    "bytes_in": r.n_bytes_in,
                    "nanoscale": {
                        "bytes": r.n_bytes_out,
                        "bits_per_byte": round(r.bits_per_byte, 4),
                        "ratio": round(r.ratio, 2),
                    },
                    "gzip": {
                        "bytes": len(gzip.compress(raw, 9)),
                        "ratio": round(len(raw) / len(gzip.compress(raw, 9)), 2),
                    },
                    "bzip2": {
                        "bytes": len(bz2.compress(raw, 9)),
                        "ratio": round(len(raw) / len(bz2.compress(raw, 9)), 2),
                    },
                    "xz": {
                        "bytes": len(lzma.compress(raw, preset=9)),
                        "ratio": round(len(raw) / len(lzma.compress(raw, preset=9)), 2),
                    },
                }
            )
            log.info("compressed %r: %.3f bpb", label, r.bits_per_byte)
        payload["compression"] = comp

        # ------------------------------------------------------------ next-token
        # The distribution at each step of a short continuation, so the page can show
        # what the model is actually choosing between.
        steps: list[dict[str, Any]] = []
        ids = tok.encode("Once upon a time there was a little", add_bos=True)
        for _ in range(12):
            logits = model(torch.tensor([ids], device=device)).logits[0, -1]
            probs = torch.softmax(logits.float(), dim=-1)
            top = torch.topk(probs, 6)
            steps.append(
                {
                    "context": tok.decode(ids[1:]),
                    "top": [
                        {"token": tok.decode([int(i)]), "p": round(float(p), 4)}
                        for p, i in zip(top.values, top.indices, strict=True)
                    ],
                }
            )
            ids.append(int(top.indices[0]))
        payload["next_token"] = steps

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    size = args.out.stat().st_size
    print(f"wrote {args.out.relative_to(ROOT)} ({size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
