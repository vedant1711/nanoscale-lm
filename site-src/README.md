# `site-src/` — the single-page documentation

`explainer.html` is the whole project in one self-contained page: every algorithm with its
motivation and mathematics, the architecture diagrams, the design decisions, all measured
results including the negative ones, the statistics behind the ablations, and the
limitations.

It is written for a reader whose deep-learning knowledge runs
ANN → CNN → RNN → LSTM → encoder–decoder → attention → Transformer and stops there;
everything past that point is developed from first principles as a delta from the 2017
decoder block.

## Design notes

- **Self-contained.** No external fonts, scripts or images. Every chart is inline SVG
  drawn with `currentColor`, so figures follow the reader's theme instead of being baked
  for one background.
- **Three theme states.** Light, dark, and follow-the-system, with an explicit toggle in
  the sidebar whose choice persists in `localStorage`. The stylesheet defines the full
  light palette on bare `:root`, redefines only tokens under `prefers-color-scheme: dark`
  (guarded so an explicit light choice wins), and again under `[data-theme="dark"]`.
- **The charts are generated**, not hand-drawn: `make_charts*.py` read
  `runs/*/metrics.jsonl` and `results/**/*.json` and emit the SVG that is pasted into the
  page.

## Provenance

Every number in the page comes from a committed artifact under `results/` or a run
manifest under `runs/`. The page hard-codes them rather than reading them at render time,
so it is a **snapshot**: when `results/` changes, the page must be regenerated to match.

`docs/results.md` is the complementary surface — it is generated from `results/` on every
build and CI fails if it has drifted. That one cannot go stale; this one is the narrative
version and can, so it is regenerated alongside any result change.
