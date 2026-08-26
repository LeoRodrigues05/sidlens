"""Regenerate every figure in diagrams/ from the workbook.

    .conda/bin/python scripts/diagrams/make_all.py

Each figure is also runnable on its own; they share style.py and read the tidy
CSVs that parse_workbook.py writes.
"""

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIGURES = [
    "fig01_collision_landscape",
    "fig02_collision_vs_performance",
    "fig03_within_tokenizer",
    "fig04_regression",
    "fig05_adjusted_ranking",
    "fig06_paradigm_gap",
    "fig07_backbone_sweep",
    "fig08_reference_grid",
]


def main():
    print("== parsing workbook ==")
    importlib.import_module("parse_workbook").main()

    for name in FIGURES:
        print(f"\n== {name} ==")
        importlib.import_module(name).main()

    print("\nAll figures written to diagrams/ (png + pdf).")


if __name__ == "__main__":
    main()
