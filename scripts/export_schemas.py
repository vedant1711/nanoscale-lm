"""Export the JSON Schema of every NanoScale config model into ``configs/schema/``.

Run with ``make schemas``. The exported schemas are committed so that config changes
show up as a reviewable diff and so editors can validate the YAML files under
``configs/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanoscale.config import export_json_schemas

DEFAULT_OUT = Path("configs/schema")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory.")
    args = parser.parse_args()
    written = export_json_schemas(args.out)
    for path in written:
        print(f"wrote {path}")
    print(f"{len(written)} schema files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
