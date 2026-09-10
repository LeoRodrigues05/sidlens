#!/usr/bin/env python
"""Run the retrospective Exp1 reveal-policy audit from frozen aggregates."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis import default_frozen_root, default_repo_root, run_audit


def main(argv: list[str] | None = None) -> int:
    repo_root = default_repo_root()
    default_out = (
        default_frozen_root().parent
        / "derived/retrospective/exp1_reveal_order"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo_root / "manifests/registry.diffusion.json",
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=default_frozen_root(),
    )
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args(argv)

    result = run_audit(
        repo_root=repo_root,
        registry_path=args.registry,
        frozen_root=args.frozen_root,
        out_dir=args.out,
        recompute_hashes=True,
    )
    primary = result["results"]["balanced18"]["metric_summaries"]
    print(f"[exp1] validation: {result['validation']['status']}")
    print(
        "[exp1] balanced18 delta: "
        f"NDCG@10={primary['ndcg@10']['mean_delta']:.6f}, "
        f"Recall@10={primary['recall@10']['mean_delta']:.6f}")
    print(f"[exp1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
