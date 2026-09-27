"""Named subsets of the substrate: what a cluster needs, and what `verify` checks.

The frozen snapshot is 14.75 GB, and 12.4 GB of that is four `model.safetensors`
files. A cluster that trains its own AR models, or only needs the SID tables and
cohorts, should not have to copy them, and it must still be able to run the
verify gate. Before profiles existed, the gate knew only "everything", so a
partial copy either failed every job or had to skip the gate. Skipping the
gate is how the upstream run directories drifted in the first place.

Why the unit is a file glob, not a manifest section
---------------------------------------------------
The AR tokenizer (`added_tokens.json`, which every prompt and every CPU test
needs) sits in the same frozen section as the 3 GB weights it belongs to. A
section-level profile can only keep both or drop both. So a profile keeps every
section it names and then excludes individual files by glob. `verify` reports
those files as *excluded by profile*, not as missing, and counts them, so a
partial copy is never mistaken for a complete one.

Beyond frozen/
--------------
`extra` names WORK-relative trees outside the frozen substrate:
`external/` (labels acquired after training) and `derived/` (SidLens's own
results). The provenance manifest does not hash them, so a bundle hashes them
when it is created. Its bundle manifest is the acceptance record for these
trees on the receiving side.

This module is plain data with no Hub or torch import. `verify` imports it on
every job start.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

# The four inference/initialisation weight files. Everything else in frozen/ is
# under 2.4 GB in total.
AR_WEIGHTS = "ckpt/ar/*/model.safetensors"
BASE_WEIGHTS = "base_models/*/model.safetensors"

# Post-training evidence and SidLens results. `external/onediffrec_data` is
# deliberately absent: no SidLens code reads it, and five of its Industrial
# RQ-KMeans tables predate the 2026-08-20 repair (see CLAUDE.md).
DEFAULT_EXTRA = ("external/amazon2018", "external/labels", "derived")


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    # Manifest section keys (e.g. "sids/index_json"). None = every section.
    frozen_sections: tuple[str, ...] | None
    # Globs over "<section>/<relative path>", matched with fnmatch.
    frozen_exclude: tuple[str, ...] = ()
    extra: tuple[str, ...] = DEFAULT_EXTRA

    def selects_section(self, section: str) -> bool:
        return self.frozen_sections is None or section in self.frozen_sections

    def excludes(self, section: str, rel: str) -> bool:
        key = f"{section}/{rel}"
        return any(fnmatch.fnmatchcase(key, g) for g in self.frozen_exclude)


PROFILES: dict[str, Profile] = {
    "core": Profile(
        "core",
        "Every frozen section except the four model weight files (~2.4 GB), "
        "plus labels and derived results. Enough for all CPU analysis and "
        "tests, AR prompts/tokenizers, and the diffusion checkpoints.",
        None, (AR_WEIGHTS, BASE_WEIGHTS)),
    "ar": Profile(
        "ar",
        "core + the three surviving AR weight sets (~11.7 GB). What a cluster "
        "needs to capture activations from, or intervene on, the frozen AR "
        "models.",
        None, (BASE_WEIGHTS,)),
    "full": Profile(
        "full",
        "The whole frozen snapshot, including the Qwen2.5-1.5B-Instruct base "
        "weights needed for AR retraining (~14.8 GB), plus labels and derived.",
        None, ()),
    "results": Profile(
        "results",
        "No frozen files. Only the trees passed with --extra, e.g. a new "
        "derived/controlled/<exp>/<run> directory being sent back.",
        (), (), ()),
}


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; have {sorted(PROFILES)}") from None
