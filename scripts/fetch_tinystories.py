"""Download a slice of TinyStories to `data/tinystories/` for the `micro` tier.

Why this exists rather than `datasets.load_dataset(..., streaming=True)`: the streaming
path in `nanoscale.train.data.iter_hf_documents` works, but on a cold cache it has to
resolve and pull multi-gigabyte parquet shards before it yields a single document, which
made it unusable on the machine this project was developed on (17 MB/s of real bandwidth,
yet minutes elapsed with no records). A plain HTTP range request over the published
`.txt` files fetches exactly the bytes wanted, is resumable, and is trivially auditable.
Both paths remain supported; this one is the default for reproducing the reported run.

**Why TinyStories rather than FineWeb-Edu.** FineWeb-Edu is general web text, and a
40M-parameter model trained on 50M of its tokens produces fluent-looking nonsense; the
distribution is far wider than the capacity. TinyStories (Eldan & Li, 2023,
arXiv:2305.07759) was constructed for exactly this regime: a vocabulary and grammar a
small child would use, which is what makes coherent generation reachable below 100M
parameters. Choosing it is a capability-matching decision, not a convenience one, and the
`micro` preset keeps FineWeb-Edu available for anyone with the compute to use it.

Usage::

    python scripts/fetch_tinystories.py                # ~200 MB train + full valid
    python scripts/fetch_tinystories.py --mb 500       # more data
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tinystories"

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN = f"{BASE}/TinyStoriesV2-GPT4-train.txt"
VALID = f"{BASE}/TinyStoriesV2-GPT4-valid.txt"

#: TinyStories separates documents with this line; the packer treats each as a document.
SEPARATOR = "<|endoftext|>"


def fetch(url: str, dest: Path, *, max_bytes: int | None) -> int:
    """Download ``url`` to ``dest``, optionally stopping after ``max_bytes``."""
    if dest.exists() and (max_bytes is None or dest.stat().st_size >= max_bytes * 0.99):
        print(f"  {dest.name}: already present ({dest.stat().st_size / 2**20:.0f} MB)")
        return dest.stat().st_size

    request = urllib.request.Request(url)
    if max_bytes is not None:
        request.add_header("Range", f"bytes=0-{max_bytes - 1}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
            written += len(chunk)
            print(f"\r  {dest.name}: {written / 2**20:7.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)

    # A range request will almost certainly cut mid-story. Truncate back to the last
    # complete document so the corpus never ends in a fragment the model would learn from.
    if max_bytes is not None:
        text = dest.read_text(encoding="utf-8", errors="ignore")
        cut = text.rfind(SEPARATOR)
        if cut > 0:
            dest.write_text(text[: cut + len(SEPARATOR)], encoding="utf-8")
    return dest.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mb", type=int, default=200, help="Megabytes of training text.")
    parser.add_argument("--valid-mb", type=int, default=20, help="Megabytes of validation text.")
    args = parser.parse_args()

    print(f"Fetching TinyStories into {OUT.relative_to(ROOT)}/")
    train = fetch(TRAIN, OUT / "train.txt", max_bytes=args.mb * 2**20)
    valid = fetch(VALID, OUT / "valid.txt", max_bytes=args.valid_mb * 2**20)

    total = train + valid
    print(
        f"\n{total / 2**20:.0f} MB total (~{total // 4:,} tokens at ~4 bytes/token).\n"
        f"Train with: nanoscale train pretrain --config configs/micro_tinystories.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
