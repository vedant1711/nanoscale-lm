"""Assemble ``docs/results.md`` from the committed artifacts in ``results/``.

Spec F5 requires that every number in the docs matches a committed artifact and that the
demo may not hand-edit numbers. The strongest way to enforce that is to make the results
page **generated**: it is stitched together from the markdown each measurement script
already wrote, so there is no second copy of a number to drift.

``--check`` regenerates into memory and fails if the committed page differs, which is what
CI runs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

#: ``(title, slug, fragment, asset_subdir)``. The slug is a **stable anchor** other pages
#: link to; it must not be derived from the fragment's own heading, because that heading
#: is prose written inside a measurement script and is free to change.
SECTIONS = (
    ("Tokenizer", "tokenizer", RESULTS / "tokenizer" / "nano.md", None),
    (
        "Full evaluation",
        "full-evaluation",
        RESULTS / "evaluation" / "micro-tinystories.md",
        "evaluation",
    ),
    (
        "External baseline (bits per byte)",
        "external-baseline",
        RESULTS / "baseline" / "baseline.md",
        "baseline",
    ),
    (
        "Capability emergence",
        "capability-emergence",
        RESULTS / "emergence" / "emergence.md",
        "emergence",
    ),
    (
        "Compression and anomaly detection",
        "compression",
        RESULTS / "compression" / "compression.md",
        "compression",
    ),
    (
        "Optimizer ablation",
        "optimizer-ablation",
        RESULTS / "ablations" / "optimizer.md",
        "ablations",
    ),
    (
        "Architecture ablation",
        "architecture-ablation",
        RESULTS / "ablations" / "architecture.md",
        "ablations",
    ),
    (
        "Optimizer ablation, 5 seeds",
        "optimizer-multiseed",
        RESULTS / "ablations" / "optimizer_multiseed.md",
        "ablations",
    ),
    (
        "Architecture ablation, 5 seeds",
        "architecture-multiseed",
        RESULTS / "ablations" / "architecture_multiseed.md",
        "ablations",
    ),
    ("Alignment", "alignment", RESULTS / "alignment" / "alignment.md", "alignment"),
    ("Distillation", "distillation", RESULTS / "distillation" / "distillation.md", "distillation"),
    ("Quantization", "quantization", RESULTS / "quantization" / "quantization.md", "quantization"),
    (
        "Speculative decoding",
        "speculative-decoding",
        RESULTS / "speculative" / "speculative.md",
        "speculative",
    ),
    ("Unified results table", "unified-results-table", RESULTS / "bench" / "table.md", "bench"),
)

HEADER = """# Results

Every figure and number on this page is copied verbatim from a committed artifact under
`results/`, which was in turn produced by a committed script and stamped with the git SHA
and hardware that produced it. This page is **generated** by
`scripts/build_docs_results.py`: editing it by hand will be reverted by CI.

!!! warning "Read the limitations first"
    Two models produced everything here. The **`micro`** tier is 40.4M parameters trained
    for 3.2 hours on TinyStories to 4% of a Chinchilla-optimal budget, and is the source of
    the evaluation, baseline, emergence and compression results. The **`nano`** tier is 5M
    parameters trained for 95 seconds on a synthetic corpus, and is the source of the
    ablations and the Arc 2 compression/speculation numbers, because it is cheap enough to
    run many controlled arms against.

    Neither is a frontier-scale claim. See [Limitations](limitations.md) for what these
    numbers do and do not support. Several results here are *negative*; a predicted effect
    that did not appear, or appeared backwards, and they are reported rather than omitted.

## On this page

<!--TOC-->

---

## The loss curve

![nano loss curve](curves/nano_loss.png)

Reproduce: `make train-nano`

---
"""


def _demote(body: str, *, title: str, slug: str) -> str:
    """Shift every heading down one level and give the section a stable anchor.

    The fragments are standalone documents with their own ``#`` title, so pasting them
    verbatim would produce a page with nine H1s. Demoting also keeps their sub-headings
    out of the page's top-level table of contents, where eight repetitions of
    "What the numbers say" would be useless.
    """
    lines = body.splitlines()
    out: list[str] = []
    in_code = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("#"):
            if i == 0:
                out.append(f"## {title} {{#{slug}}}")
                continue
            line = "#" + line
        out.append(line)
    return "\n".join(out).rstrip()


def build() -> str:
    """Stitch the page together from the committed section files."""
    toc = "\n".join(f"- [{title}](#{slug})" for title, slug, _p, _a in SECTIONS)
    parts = [HEADER.replace("<!--TOC-->", toc)]
    for title, slug, path, _assets in SECTIONS:
        if not path.exists():
            parts.append(
                f"\n## {title} {{#{slug}}}\n\n_Not yet generated: `{path.relative_to(ROOT)}`._\n"
            )
            continue
        body = _demote(path.read_text(encoding="utf-8"), title=title, slug=slug)
        parts.append("\n" + body + "\n\n---\n")
    return "".join(parts).rstrip() + "\n"


def copy_assets() -> list[Path]:
    """Copy the referenced figures into ``docs/`` so mkdocs can resolve them."""
    copied: list[Path] = []
    curves = DOCS / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    for png in (RESULTS / "curves").glob("*.png"):
        shutil.copy2(png, curves / png.name)
        copied.append(curves / png.name)
    for _title, _slug, path, subdir in SECTIONS:
        if subdir is None or not path.exists():
            continue
        for png in path.parent.glob("*.png"):
            shutil.copy2(png, DOCS / png.name)
            copied.append(DOCS / png.name)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed page is stale.")
    args = parser.parse_args()

    page = build()
    target = DOCS / "results.md"

    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != page:
            print(
                "docs/results.md is stale: it does not match the committed results/ "
                "artifacts. Run `python scripts/build_docs_results.py`.",
                file=sys.stderr,
            )
            return 1
        print("docs/results.md is up to date with results/.")
        return 0

    DOCS.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding="utf-8")
    assets = copy_assets()
    print(f"wrote {target} and {len(assets)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
