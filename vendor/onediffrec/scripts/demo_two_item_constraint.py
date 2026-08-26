#!/usr/bin/env python3
"""Show that the two-item second pass generated its item without a constraint.

No GPU, no checkpoint, no model weights. The constrained-decoding trie is a pure
function of the tokenizer and the info file, so the exact lookup that
ConstrainedLogitsProcessor performs can be replayed here and inspected.

The trie is keyed by "what has been generated since the prompt ended". Its root
key is the tokens of "### Response:\n", because generation always starts there.
The processor reproduces that root on its first step by reading the last
`prefix_index` (=3) tokens off the sequence:

    if self.count == 0:
        hash_key = sent[-self.prefix_index:]   # assumes we start at the prompt end
    else:
        hash_key = sent[-self.count:]

Pass 1 satisfies that assumption. Pass 2 does not: it resumes after the first
predicted item, so those last three tokens are that item's finished SID. That
string is itself a valid key in the same trie - the one meaning "this item is
complete" - so the lookup succeeds and returns a single token: the newline that
ends an item. The constraint never errors, never returns empty, never warns.

Prints the allowed-token set at each step for pass 1, for pass 2 as it was, and
for pass 2 as fixed, so the three can be compared directly.

Usage:
    python scripts/demo_two_item_constraint.py [--codebooks 3] [--size 256]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.environ.get("DATA_ROOT", "/l/users/leo.rodrigues/onediffrec/data")
MODEL_DIR = os.environ.get(
    "MODEL_DIR", "/l/users/leo.rodrigues/onediffrec/models/Qwen2.5-1.5B-Instruct")


def get_hash(x):
    """Verbatim from evaluate_two_item.py."""
    return "-".join(str(_) for _ in x)


def build_trie(prefix_ids, prefix_index, eos_id):
    """Verbatim from evaluate_two_item.py lines 99-113."""
    hash_dict = {}
    for ID in prefix_ids:
        ID = list(ID) + [eos_id]
        for i in range(prefix_index, len(ID)):
            key = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            hash_dict.setdefault(key, set()).add(ID[i])
    return {k: sorted(v) for k, v in hash_dict.items()}


def show(tokenizer, label, allowed, limit=6):
    if not allowed:
        print(f"      {label:<24} -> []  (EMPTY: the processor skips this beam,")
        print(f"      {'':<24}     so NO mask is applied - generation is free)")
        return
    shown = ", ".join(repr(tokenizer.decode([t])) for t in allowed[:limit])
    more = f", ... (+{len(allowed) - limit} more)" if len(allowed) > limit else ""
    print(f"      {label:<24} -> {len(allowed):>4} token(s): {shown}{more}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codebooks", type=int, default=3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--method", default="rqvae")
    parser.add_argument("--category", default="Industrial_and_Scientific")
    args = parser.parse_args()

    import variants as V
    from transformers import AutoTokenizer

    paths = V.resolve(DATA_ROOT, V.Variant(
        "two-item", args.category, args.method, args.codebooks, args.size))

    # Reproduce the checkpoint's tokenizer: base vocabulary + the SID tokens that
    # sft.py adds from the index file. Identical to what evaluate.py loads.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    index = json.load(open(paths["index"]))
    new_tokens = sorted({t for sids in index.values() for t in sids})
    tokenizer.add_tokens(new_tokens)

    with open(paths["info"]) as fh:
        info = fh.readlines()
    semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info]
    info_semantic = [f"### Response:\n{s}" for s in semantic_ids]

    prefix_ids = [tokenizer(s).input_ids for s in info_semantic]
    prefix_index = 3                      # Qwen path in evaluate_two_item.py
    hash_dict = build_trie(prefix_ids, prefix_index, tokenizer.eos_token_id)

    def allowed_for(key_tokens):
        return hash_dict.get(get_hash(list(key_tokens)), [])

    response_prefix = prefix_ids[0][:prefix_index]

    print("=" * 78)
    print(f"  Two-item constrained decoding - {args.method} "
          f"{args.codebooks}codebook_{args.size}, {args.category}")
    print("=" * 78)
    print(f"\n  info file    : {os.path.basename(paths['info'])}  ({len(info)} items)")
    print(f"  SID tokens   : {len(new_tokens)} added to the tokenizer")
    print(f"  trie keys    : {len(hash_dict)}")
    print(f"  prefix_index : {prefix_index}")
    print(f"  '### Response:' + newline -> {response_prefix} = "
          f"{[tokenizer.decode([t]) for t in response_prefix]}")

    item1 = semantic_ids[0].strip()
    item1_tokens = tokenizer(item1, add_special_tokens=False).input_ids
    print(f"\n  a first predicted item : {item1}  -> {len(item1_tokens)} tokens")

    print("\n" + "-" * 78)
    print("  PASS 1  - generation starts immediately after the response prefix")
    print("-" * 78)
    print("    step 0, count == 0 : hash_key = sent[-3:] = the response prefix")
    show(tokenizer, "allowed at step 0", allowed_for(response_prefix))
    print("    -> works: only real first-digits of catalogue SIDs are permitted.")

    print("\n" + "-" * 78)
    print("  PASS 2 (as written)  - resumes after the first item")
    print("-" * 78)
    print(f"    sequence now ends with : '{item1}'   (the finished first SID)")
    print("    step 0, count == 0 : hash_key = sent[-3:] = that SID's own tokens")
    key0 = item1_tokens[-prefix_index:]
    a0 = allowed_for(key0)
    show(tokenizer, "allowed at step 0", a0)
    if len(a0) == 1:
        print(f"    -> ONE token allowed: {tokenizer.decode(a0)!r} - the end-of-item")
        print("       marker. The model is forced to close the item before emitting")
        print("       a single digit of the second one.")

    print("\n    step 1, count == 1 : hash_key = sent[-1:] = that forced token")
    a1 = allowed_for(a0[:1]) if a0 else []
    show(tokenizer, "allowed at step 1", a1)
    print("    -> the mask is never applied again. The rest of item 2 is generated")
    print("       completely unconstrained, free to emit non-existent SIDs.")

    print("\n" + "-" * 78)
    print("  PASS 2 (fixed)  - prefix_key names the trie root explicitly")
    print("-" * 78)
    print("    step 0, count == 0 : hash_key = prefix_key = the response prefix")
    show(tokenizer, "allowed at step 0", allowed_for(response_prefix))
    print("    -> identical to pass 1, which is the point.")

    import pandas as pd
    train_target = str(pd.read_csv(paths["train"])["item_sid"].iloc[0])
    sep = tokenizer.encode(" ||| ", add_special_tokens=False)
    print("\n" + "=" * 78)
    print("  Separately: the separator the model was trained on")
    print("=" * 78)
    print(f"\n    a training target looks like : {train_target}")
    print("    pass 2 as written appended   : input_ids + tokens(item1)")
    print("    pass 2 fixed appends         : input_ids + tokens(item1) + tokens(' ||| ')")
    print(f"\n    ' ||| ' -> {sep} = {[tokenizer.decode([t]) for t in sep]}")
    print("    Without it the model never sees the format it was trained on.")

    ok = len(a0) == 1 and not a1
    print("\n" + "=" * 78)
    if ok:
        print("  CONFIRMED: pass 2 allowed exactly one token at step 0 (end-of-item),")
        print("  then lost the constraint entirely. The second item was never")
        print("  restricted to the catalogue.")
    else:
        print("  NOT REPRODUCED on this variant - inspect the output above.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
