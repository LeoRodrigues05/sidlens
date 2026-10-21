"""Review days and target classes for grid rows, both catalogs, both paradigms.

Every grid row (AR CSV row, DiffGRM cohort row, saved prediction) names the
user and the zero-based position of each history and target event in that
user's `inter.json` list (OneDiffRec `horizon_split`). Days therefore join by
(user, position), the join `sidlens.data.timestamps` was built for, with no
window search.

Traps, each with its guard
--------------------------
1. **Another catalog's or root's events.** The days are aligned on the run's
   own data root (`<data_root>/Amazon18/<category>/`), the files the run was
   prepared from, and every row's items are checked against `inter.json` at
   their positions (`timestamps.position_row_times`).
2. **AR and DiffGRM rows joined by number.** Both cohorts of a grid cell share
   `example_id`; tables are keyed by it, never by row index.
3. **An unsettled duplicate day.** Office has one such position; rows touching
   it carry `time_ambiguous` and time analyses exclude them.
4. **Near-duplicates by item index from another item map.** The pairs come
   from structure exp4 (`derived/grid_inputs/near_duplicates/`, shipped with
   the `grid` profile); they are used only if the run's `item2id` file has the
   hash those indices were made with.

Target classes (retrospective exp7's, in its precedence): `same_item` (the
target item is in the history), `sid_partner` (another item with the target's
SID is), `nd_variant` (a near-duplicate of the target with a different SID
is), `new`.
"""

from __future__ import annotations

import json
from pathlib import Path

from sidlens import paths
from sidlens.data import timestamps as T
from sidlens.grid import inputs as GI
from sidlens.provenance.hashing import sha256_file

CLASSES = ("same_item", "sid_partner", "nd_variant", "new")
ND_DIR = "derived/grid_inputs/near_duplicates"


def event_times(e: dict) -> T.EventTimes:
    """Days of every event of the run's catalog, aligned on the run's data root."""
    root = GI.category_dir(e)
    return T.load_event_times(e["category"], str(root), "flag")


def cohort_rows(e: dict, split: str = "test") -> list[dict]:
    """The prepared cohort's rows (identical for the AR and DiffGRM runs of a cell)."""
    f = Path(e["prepared_dir"]) / f"{GI.SPLIT_FILES[split]}-cohort.json"
    GI._check_prepared(e, f)
    return json.loads(f.read_text())["examples"]


def row_times(e: dict, split: str = "test", example_ids=None, ev: T.EventTimes | None = None):
    """(rows, hist) day tables for the cohort's rows, in `example_ids` order if given.

    Columns as `timestamps.position_row_times`; `row` is the cohort index.
    """
    ev = ev or event_times(e)
    items, users = GI.item2id(e), GI.user2id(e)
    rows = cohort_rows(e, split)
    index = {r["example_id"]: k for k, r in enumerate(rows)}
    order = [index[x] for x in example_ids] if example_ids is not None else range(len(rows))
    specs = []
    for k in order:
        r = rows[k]
        specs.append({"example_id": r["example_id"], "row": k, "user_id": r["user_id"],
                      "user": str(users[r["user_id"]]),
                      "history_positions": r["history_event_positions"],
                      "target_positions": r["target_event_positions"],
                      "history_item_ids": [items[a] for a in r["history_item_ids"]],
                      "target_item_ids": [items[a] for a in r["target_item_ids"]]})
    return T.position_row_times(specs, ev)


def near_duplicates(e: dict) -> dict[int, set[int]]:
    """Item index -> its near-duplicates, for the run's catalog (trap 4)."""
    d = paths.WORK / ND_DIR
    src = json.loads((d / "source.json").read_text())["catalogs"][e["dataset"]]
    f = d / src["file"]
    if sha256_file(f) != src["sha256"]:
        raise RuntimeError(f"{f} does not have its recorded sha256")
    mine = sha256_file(GI.category_dir(e) / f"{e['category']}.item2id")
    if mine != src["item2id_sha256"]:
        raise RuntimeError(f"{e['key']}: item2id differs from the map the near-duplicate pairs were indexed by")
    import pandas as pd
    nd = pd.read_csv(f)
    adj: dict[int, set[int]] = {}
    for a, b in zip(nd.i, nd.j):
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))
    return adj


def target_classes(e: dict, split: str = "test", example_ids=None, nd_adj=None):
    """One row per cohort row: the class of each target slot (`cls`, and `cls2` for
    next-two), the target's SID bucket size, and the SIDs of its near-duplicate
    history items (exp7's `variant_sids`)."""
    import pandas as pd

    nd_adj = near_duplicates(e) if nd_adj is None else nd_adj
    items, table = GI.item2id(e), GI.sid_table(e)
    code = {items[a]: c for a, c in table.asin2codes.items()}
    rows = cohort_rows(e, split)
    index = {r["example_id"]: k for k, r in enumerate(rows)}
    order = [index[x] for x in example_ids] if example_ids is not None else range(len(rows))
    out = []
    for k in order:
        r = rows[k]
        hist = [items[a] for a in r["history_item_ids"]]
        hcodes = {code[h] for h in hist}
        rec = {"example_id": r["example_id"], "row": k, "user_id": r["user_id"]}
        for slot, a in enumerate(r["target_item_ids"]):
            ti = items[a]
            partners = [h for h in hist if h in nd_adj.get(ti, ())]
            cls = ("same_item" if ti in hist else "sid_partner" if code[ti] in hcodes
                   else "nd_variant" if partners else "new")
            sfx = "" if slot == 0 else str(slot + 1)
            rec.update({f"cls{sfx}": cls, f"target_item{sfx}": ti, f"target_codes{sfx}": code[ti],
                        f"target_bucket{sfx}": len(table.cb2items[code[ti]]),
                        f"variant_codes{sfx}": sorted({code[h] for h in partners})})
        out.append(rec)
    return pd.DataFrame(out)
