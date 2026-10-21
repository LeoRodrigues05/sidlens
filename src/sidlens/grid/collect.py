"""Concatenate per-cell `estimates.csv` files of grid cells into one table.

For experiments whose cells already carry their run's identity and bootstrap
intervals (the saved-list analyses, `grid.saved_lists`), a summary is the
union of the cells' estimates, with the cell folder beside each row.

    python -m sidlens.grid.collect --cells <cell> [...] --out <new dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    frames, skipped = [], []
    for c in args.cells:
        f = c / "estimates.csv"
        if not f.exists():
            skipped.append(str(c))
            continue
        frames.append(pd.read_csv(f).assign(cell=str(c)))
    est = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    est.to_csv(args.out / "estimates.csv", index=False)
    (args.out / "inputs.json").write_text(json.dumps({"cells": [str(c) for c in args.cells],
                                                      "skipped": skipped, "rows": len(est)}, indent=1))
    print(f"wrote {args.out}: {len(est)} rows from {len(frames)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
