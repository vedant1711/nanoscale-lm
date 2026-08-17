# `site-src/` — the long-form explainer

`explainer.html` is a single self-contained page that walks through the whole project:
every algorithm with its motivation and mathematics, the architecture diagrams, the design
decisions, all measured results including the negative ones, and the limitations.

It is written for a reader whose deep-learning knowledge runs
ANN → CNN → RNN → LSTM → encoder–decoder → attention → Transformer and stops there;
everything past that point is developed from first principles.

**Provenance.** Every number in the page comes from a committed artifact under `results/`
or from a run manifest under `runs/`. The page hard-codes them rather than reading them at
render time, so it is a *snapshot*: when `results/` changes, the page must be updated to
match. `docs/results.md`, by contrast, is generated and CI-checked against `results/` —
that is the always-current surface, and this page is the narrative one.

The inline SVG charts (loss curves, quantization frontier) are drawn from the committed
`runs/*/metrics.jsonl` and `results/quantization/frontier.json` rather than embedded as
PNGs, so that they follow the reader's light/dark theme.
