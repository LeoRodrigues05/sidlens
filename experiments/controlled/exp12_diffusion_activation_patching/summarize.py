#!/usr/bin/env python
"""Summarize controlled/exp12 over cells: restored shares, block profiles, edge shares.

Reads `<run>/cell-XX/` (only cells whose validation.json exists), writes
`estimates.csv` (one row per cell x estimand x site x digit), `profiles.csv`
(the block profiles the figures draw), `report.md` and `figures/` into a new
directory. Ratios are ratios of pooled sums, bootstrapped over users with
`UserBootstrap` (2,000 draws, seed 20261014); intervals exclude training-seed
uncertainty.

    python summarize.py --run <run dir> --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402

SEED = 20261014
N_LAYERS = 4


def markdown(df: pd.DataFrame) -> str:
    """A pipe table without the optional `tabulate` dependency."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, rule] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def cell_info(ckpt: str) -> dict:
    from sidlens import grid
    if grid.is_grid(ckpt):
        return grid.cell_info(ckpt)
    q, depth, width = re.match(r"diff-next1-(\w+)-(\d)cb-(\d+)", ckpt).groups()
    return {"quantizer": q, "depth": int(depth), "width": int(width)}


def load_cell(path: Path) -> dict | None:
    if not (path / "validation.json").exists():
        return None
    inputs = json.loads((path / "inputs.json").read_text())
    out = {"inputs": inputs, **cell_info(inputs["checkpoint"])}
    for part in "HS":
        z = np.load(path / f"sites_{part}.npz")
        out[part] = {"meta": pd.read_csv(path / f"pairs_{part}.csv"), "sites": list(z["sites"]),
                     "knockouts": list(z["knockouts"]), "kl": z["kl"], "gold": z["gold"], "copy": z["copy"],
                     "ko_gold": z["ko_gold"]}
    return out


def restored(boot, user, kl_site, kl_corr, direction):
    """Denoising: 1 - sum KL(patched) / sum KL(corrupted); noising: sum KL(noised) / sum KL(corrupted)."""
    r = boot.ratio(user, kl_site, kl_corr)
    if direction == "denoise":
        r = {**r, "est": 1 - r["est"], "lo": 1 - r["hi"], "hi": 1 - r["lo"]}
    return r


def summarize_cell(cell: dict, boot_users) -> tuple[list[dict], list[dict]]:
    rows, prof = [], []
    base = {"checkpoint": cell["inputs"]["checkpoint"], "quantizer": cell["quantizer"], "depth": cell["depth"],
            "width": cell["width"]}
    boot = UserBootstrap(boot_users, draws=2000, seed=SEED)
    H, S = cell["H"], cell["S"]
    users = cell["inputs"].get("n_users")

    def col(part, site):
        return part["sites"].index(site)

    # ---- Part H -------------------------------------------------------------------
    m = H["meta"]
    groups = {"H0": (m.d == 0).values, "Hd_match": ((m.d >= 1) & m.match).values,
              "Hd_nonmatch": ((m.d >= 1) & ~m.match).values}
    for d in sorted(m.d.unique()):
        groups[f"H{d}" + ("" if d == 0 else "_match")] = ((m.d == d) & (m.match | (m.d == 0))).values
    for g, sel in groups.items():
        if sel.sum() == 0:
            continue
        u, klc = m.user_index.values[sel], m.kl_corr.values[sel]
        rows.append({**base, "estimand": "delta_gold", "group": g, "site": "-",
                     **boot.mean(u, (m.gold_clean - m.gold_corr).values[sel])})
        rows.append({**base, "estimand": "kl_corrupted", "group": g, "site": "-", **boot.mean(u, klc)})
        for direction in ("denoise", "noise"):
            for site in ("E:r1", "E:others", "E:r2"):
                r = restored(boot, u, H["kl"][sel, col(H, f"{direction}|{site}")], klc, direction)
                rows.append({**base, "estimand": f"R_{direction}", "group": g, "site": site, **r})
            diff = boot.ratio(u, H["kl"][sel, col(H, f"{direction}|E:r1")] - H["kl"][sel, col(H, f"{direction}|E:others")], klc)
            if direction == "noise":
                diff = {**diff, "est": -diff["est"], "lo": -diff["hi"], "hi": -diff["lo"]}
            rows.append({**base, "estimand": f"R_{direction}_others_minus_r1", "group": g, "site": "-", **diff})
        ds = m.d.values[sel]
        n = cell["depth"]
        for kind in ("X", "D"):
            for direction in ("denoise", "noise"):
                for l in range(N_LAYERS):
                    query = np.stack([H["kl"][sel, col(H, f"{direction}|{kind}:L{l}:p{p}")] for p in range(n)], 1)
                    q_kl = query[np.arange(len(ds)), ds]
                    other = (query.sum(1) - q_kl) / max(n - 1, 1)
                    for pos, val in (("query", q_kl), ("other_mean", other)):
                        r = restored(boot, u, val, klc, direction)
                        rows.append({**base, "estimand": f"R_{direction}", "group": g, "site": f"{kind}:L{l}:{pos}", **r})
                        prof.append({**base, "group": g, "kind": kind, "direction": direction, "layer": l, "pos": pos,
                                     "est": r["est"], "lo": r["lo"], "hi": r["hi"]})

    # ---- Part S -------------------------------------------------------------------
    m = S["meta"]
    groups = {"S_all": np.ones(len(m), bool), **{f"S{d}": (m.d == d).values for d in sorted(m.d.unique())}}
    for g, sel in groups.items():
        if sel.sum() == 0:
            continue
        u, klc = m.user_index.values[sel], m.kl_corr.values[sel]
        dg = (m.gold_clean - m.gold_corr).values[sel]
        rows.append({**base, "estimand": "delta_gold", "group": g, "site": "-", **boot.mean(u, dg)})
        rows.append({**base, "estimand": "kl_corrupted", "group": g, "site": "-", **boot.mean(u, klc)})
        ds = m.d.values[sel]
        n = cell["depth"]
        for kind in ("D", "X"):
            for direction in ("denoise", "noise"):
                for l in range(N_LAYERS):
                    per = np.stack([S["kl"][sel, col(S, f"{direction}|{kind}:L{l}:p{p}")] for p in range(n)], 1)
                    idx = np.arange(len(ds))
                    for pos, val in (("changed", per[idx, ds - 1]), ("query", per[idx, ds]),
                                     ("other_mean", (per.sum(1) - per[idx, ds - 1] - per[idx, ds]) / max(n - 2, 1))):
                        if pos == "other_mean" and n < 3:
                            continue
                        r = restored(boot, u, val, klc, direction)
                        rows.append({**base, "estimand": f"R_{direction}", "group": g, "site": f"{kind}:L{l}:{pos}", **r})
                        prof.append({**base, "group": g, "kind": kind, "direction": direction, "layer": l, "pos": pos,
                                     "est": r["est"], "lo": r["lo"], "hi": r["hi"]})
        for ki, kname in enumerate(S["knockouts"]):
            dko = S["ko_gold"][sel, ki, 0] - S["ko_gold"][sel, ki, 1]
            r = boot.ratio(u, dko, dg)
            rows.append({**base, "estimand": "edge_share", "group": g, "site": kname,
                         **{**r, "est": 1 - r["est"], "lo": 1 - r["hi"], "hi": 1 - r["lo"]}})
    return rows, prof


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, nargs="*", default=[], help="run dirs holding cell-*/")
    ap.add_argument("--cells", type=Path, nargs="*", default=[], help="cell dirs (grid layout)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, prof, skipped, validations = [], [], [], {}
    for path in [p for run in args.run for p in sorted(run.glob("cell-*"))] + list(args.cells):
        cell = load_cell(path)
        if cell is None:
            skipped.append(str(path))
            continue
        validations[cell["inputs"]["checkpoint"]] = json.loads((path / "validation.json").read_text())
        users = np.unique(np.concatenate([cell["H"]["meta"].user_index.values, cell["S"]["meta"].user_index.values]))
        r, p = summarize_cell(cell, users)
        rows += r
        prof += p
        print(f"{path.name} {cell['inputs']['checkpoint']}: {len(r)} estimates", flush=True)
    est, prof = pd.DataFrame(rows), pd.DataFrame(prof)
    est.to_csv(args.out / "estimates.csv", index=False)
    prof.to_csv(args.out / "profiles.csv", index=False)
    gates = {k: all(bool(v.get(g, False)) for g in v if g.startswith(("V2", "V3", "V4"))) for k, v in validations.items()}
    (args.out / "inputs.json").write_text(json.dumps({"runs": [str(r) for r in args.run], "skipped": skipped,
                                                      "gates_passed": gates, "seed": SEED}, indent=1))

    def pick(estimand, group, site):
        sel = est[(est.estimand == estimand) & (est.group == group) & (est.site == site)]
        return sel.set_index("checkpoint")[["est", "lo", "hi"]]

    lines = ["# controlled/exp12 summary", "", f"Cells: {len(validations)}; skipped: {len(skipped)}. "
             "Intervals: user bootstrap, 2,000 draws, seed 20261014; they exclude training-seed uncertainty.", ""]
    table = []
    for ck in validations:
        info = cell_info(ck)

        def f(e, g, s):
            t = pick(e, g, s)
            return "" if ck not in t.index else f"{t.loc[ck, 'est']:.2f} [{t.loc[ck, 'lo']:.2f}, {t.loc[ck, 'hi']:.2f}]"
        table.append({"cell": f"{info['quantizer']} {info['depth']}x{info['width']}",
                      "H0 R(E:r1)": f("R_denoise", "H0", "E:r1"), "H0 R(E:others)": f("R_denoise", "H0", "E:others"),
                      "Hd match R(E:r1)": f("R_denoise", "Hd_match", "E:r1"),
                      "Hd match R(E:others)": f("R_denoise", "Hd_match", "E:others"),
                      "S edge share K:all": f("edge_share", "S_all", "K:all")})
    lines += [markdown(pd.DataFrame(table)), ""]
    (args.out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
