"""Interventions: code that CHANGES what a model computes, then measures the change.

`sidlens.hooks` observes and must never alter an activation; everything that
does alter one lives here, so that "does this change numerics?" has one answer
per package. Each module names the trap its guards close:

    residual   substitute a block's output at chosen (row, position) pairs --
               activation patching. The patch must fire exactly once per
               forward or the call raises: a hook on a module the forward never
               reaches (DiffGRM's decoder under `forward()`) is a silent no-op
               that would read as "patching had no effect".
    history    replace one history item with a prefix-matched catalogue control,
               rebuilt through `ar_prompts.encode`, and prove the new ids differ
               from the clean ids only where intended.
    scoring    per-digit target scores at `predict_pos`, over the digit's codes
               (the model's actual decision set), its legal continuations, and
               the full vocabulary.

Every intervention output must record the clean input, the intervened input,
the site, the patch source, and per-digit log-prob and rank at predict_pos
(docs/clusters/AR_CLUSTER_PROMPT.md, contract 6).
"""
