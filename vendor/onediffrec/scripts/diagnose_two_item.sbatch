#!/bin/bash
#SBATCH --job-name=onediffrec-diag2i
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=gpu-debug-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Why does evaluate_two_item.py produce an empty second item in ~99% of pairs?
# Reproduces pass 1 and pass 2 exactly, shows what the model generates when
# unconstrained, and tests a candidate fix, on a handful of test rows.

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda
CKPT=${WORK}/sweep/two-item/runs/twoitem__Industrial_and_Scientific__MQ__3cb__512/final_checkpoint
DATA=${WORK}/data/two-item
TEST=${DATA}/test/Industrial_and_Scientific_MQ_3codebook_512.csv
INFO=${DATA}/info/Industrial_and_Scientific_MQ_3codebook_512.txt

export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=${WORK}/cache/huggingface

cd "${REPO}"
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"

srun --ntasks=1 --kill-on-bad-exit=1 "${ENV_PREFIX}/bin/python" - "${CKPT}" "${TEST}" "${INFO}" <<'PY'
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList

sys.path.insert(0, "/home/leo.rodrigues/onediffrec/OneDiffRec")
from data import EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor

ckpt, test_path, info_path = sys.argv[1:4]
device = "cuda"

def get_hash(x):
    return '-'.join(str(_) for _ in x)

tok = AutoTokenizer.from_pretrained(ckpt)
tok.pad_token = tok.eos_token
tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).eval().to(device)
model.config.pad_token_id = model.config.eos_token_id = tok.eos_token_id

# --- rebuild the constraint tree exactly as evaluate_two_item.py does -------
info = open(info_path).readlines()
semantic_ids = [l.split('\t')[0].strip() + "\n" for l in info]
info_semantic = [f'### Response:\n{s}' for s in semantic_ids]
prefixID = [tok(s).input_ids for s in info_semantic]
PREFIX_INDEX = 3

hash_dict = {}
for ID in prefixID:
    ID.append(tok.eos_token_id)
    for i in range(PREFIX_INDEX, len(ID)):
        key = get_hash(ID[:i]) if i == PREFIX_INDEX else get_hash(ID[PREFIX_INDEX:i])
        hash_dict.setdefault(key, set()).add(ID[i])
hash_dict = {k: list(v) for k, v in hash_dict.items()}

def allowed(batch_id, ids):
    return hash_dict.get(get_hash(ids), [])


class FixedConstrainedLogitsProcessor(ConstrainedLogitsProcessor):
    """Candidate fix: key the first constrained step off an explicit prefix.

    The stock processor assumes generation starts immediately after
    "### Response:\n" and looks at sent[-prefix_index:] on its first call. In
    pass 2 those tokens are the tail of pred1, so nothing matches and the
    constraint silently disappears. Supplying the real response-prefix key
    restores it.
    """

    def __init__(self, *args, prefix_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_key = prefix_key

    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, -1000000)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    key = list(self.prefix_key) if self.prefix_key is not None \
                        else sent[-self.prefix_index:].tolist()
                else:
                    key = sent[-self.count:].tolist()
                toks = self._prefix_allowed_tokens_fn(batch_id, key)
                if len(toks) == 0:
                    continue
                mask[batch_id * self._num_beams + beam_id, toks] = 0
        self.count += 1
        return scores + mask

RESPONSE_PREFIX = prefixID[0][:PREFIX_INDEX]
print(f"response-prefix tokens = {RESPONSE_PREFIX} -> {tok.decode(RESPONSE_PREFIX)!r}")
print(f"hash_dict entries      = {len(hash_dict)}")
print(f"valid SIDs             = {len(semantic_ids)}")

ds = EvalSidDataset(train_file=test_path, tokenizer=tok, max_len=2560,
                    category="industrial and scientific items", test=True, seed=42)
# get_all() is avoided on purpose: get_history() writes the eval'd list back into
# self.data when the frame is a single object-dtype block, so a second pass over
# the same rows raises. Ground truth is read straight from the CSV instead.
import pandas as pd
gt_raw = pd.read_csv(test_path)["item_sid"].tolist()
rows = [{"output": g} for g in gt_raw]

def gen(input_ids, num_beams, prefix_key=None, constrained=True, max_new=32):
    cfg = GenerationConfig(num_beams=num_beams, num_return_sequences=num_beams,
                           length_penalty=0.0, max_new_tokens=max_new,
                           pad_token_id=model.config.pad_token_id,
                           eos_token_id=model.config.eos_token_id)
    procs = None
    if constrained:
        if prefix_key is not None:
            clp = FixedConstrainedLogitsProcessor(prefix_allowed_tokens_fn=allowed,
                                                  num_beams=num_beams, base_model=ckpt,
                                                  prefix_key=prefix_key)
        else:
            clp = ConstrainedLogitsProcessor(prefix_allowed_tokens_fn=allowed,
                                             num_beams=num_beams, base_model=ckpt)
        procs = LogitsProcessorList([clp])
    with torch.no_grad():
        out = model.generate(torch.tensor([input_ids]).to(device),
                             generation_config=cfg, return_dict_in_generate=True,
                             logits_processor=procs)
    n = len(input_ids)
    return [tok.decode(s[n:], skip_special_tokens=True) for s in out.sequences]

for idx in range(3):
    enc = ds[idx]
    ids = enc["input_ids"]
    print("\n" + "=" * 78)
    print(f"SAMPLE {idx}  gt = {rows[idx]['output'].strip()!r}")
    print(f"  prompt ends with: {tok.decode(ids[-12:])!r}")

    pass1 = gen(ids, num_beams=4)
    p1 = pass1[0].split("Response:\n")[-1].strip()
    print(f"  PASS1 (constrained, top-4): {[p.split('Response:')[-1].strip() for p in pass1]}")

    p1_tokens = tok.encode(p1, add_special_tokens=False)

    cur = gen(ids + p1_tokens, num_beams=1)
    print(f"  PASS2 as-implemented      : {cur[0]!r}   <-- expect empty")

    unc = gen(ids + p1_tokens, num_beams=1, constrained=False)
    print(f"  PASS2 UNCONSTRAINED       : {unc[0]!r}   <-- what the model wants to emit")

    sep = tok.encode(" ||| ", add_special_tokens=False)
    unc2 = gen(ids + p1_tokens + sep, num_beams=1, constrained=False)
    print(f"  after appending ' ||| '   : {unc2[0]!r}")

    fixed = gen(ids + p1_tokens + sep, num_beams=1, prefix_key=RESPONSE_PREFIX)
    print(f"  CANDIDATE FIX (constrained w/ prefix_key): {fixed[0]!r}")

print("\ndiagnosis complete")
PY

echo "[$(date --iso-8601=seconds)] done"
