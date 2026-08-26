#!/bin/bash
# Score one trained checkpoint with the repo's standard offline eval pipeline:
#   split.py -> N x evaluate.py (constrained beam search) -> merge.py -> calc.py
#
# Generation settings match evaluate.sh so numbers stay comparable with the
# results already recorded in the project notes.
#
# Usage:
#   scripts/eval_variant.sh <tree> <category> <method> <codebooks> <size> \
#                           <checkpoint_dir> <work_dir> <result_json>
#
# Writes the merged predictions to <result_json> and prints calc.py's metrics
# block to stdout.

set -euo pipefail

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
ENV_PREFIX=${REPO}/.conda
DATA_ROOT=${DATA_ROOT:-/l/users/leo.rodrigues/onediffrec/data}

TREE=$1
CATEGORY=$2
METHOD=$3
CODEBOOKS=$4
SIZE=$5
CHECKPOINT=$6
WORKDIR=$7
RESULT_JSON=$8

NUM_BEAMS=${NUM_BEAMS:-50}
EVAL_BATCH=${EVAL_BATCH:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
LENGTH_PENALTY=${LENGTH_PENALTY:-0.0}
CUDA_LIST=${CUDA_LIST:-0,1,2,3}

# The two-item setting needs the two-pass generator and its own scorer.
if [[ "${TREE}" == "two-item" ]]; then
    EVAL_SCRIPT=${REPO}/evaluate_two_item.py
    CALC_SCRIPT=${REPO}/calc_two_item.py
else
    EVAL_SCRIPT=${REPO}/evaluate.py
    CALC_SCRIPT=${REPO}/calc.py
fi

read -r TEST_FILE INFO_FILE < <(
    "${ENV_PREFIX}/bin/python" - "$TREE" "$CATEGORY" "$METHOD" "$CODEBOOKS" "$SIZE" "$DATA_ROOT" <<'PY'
import sys
sys.path.insert(0, "/home/leo.rodrigues/onediffrec/OneDiffRec/scripts")
import variants as V

tree, category, method, codebooks, size, data_root = sys.argv[1:7]
paths = V.resolve(data_root, V.Variant(tree, category, method, int(codebooks), int(size)))
print(paths["test"], paths["info"])
PY
)

[[ -f "${TEST_FILE}" ]] || { echo "missing test file: ${TEST_FILE}" >&2; exit 1; }
[[ -f "${INFO_FILE}" ]] || { echo "missing info file: ${INFO_FILE}" >&2; exit 1; }
[[ -d "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }

TEMP_DIR=${WORKDIR}/eval_shards
rm -rf "${TEMP_DIR}"
mkdir -p "${TEMP_DIR}" "$(dirname "${RESULT_JSON}")"

echo "eval: tree=${TREE} ${CATEGORY} ${METHOD} ${CODEBOOKS}cb/${SIZE} beams=${NUM_BEAMS}"
echo "  test=${TEST_FILE}"
echo "  info=${INFO_FILE}"
echo "  ckpt=${CHECKPOINT}"

"${ENV_PREFIX}/bin/python" "${REPO}/split.py" \
    --input_path "${TEST_FILE}" \
    --output_path "${TEMP_DIR}" \
    --cuda_list "${CUDA_LIST}"

shards=${CUDA_LIST//,/ }
pids=()
for i in ${shards}; do
    [[ -f "${TEMP_DIR}/${i}.csv" ]] || { echo "missing shard ${i}, skipping"; continue; }
    CUDA_VISIBLE_DEVICES=${i} "${ENV_PREFIX}/bin/python" -u "${EVAL_SCRIPT}" \
        --base_model "${CHECKPOINT}" \
        --info_file "${INFO_FILE}" \
        --category "${CATEGORY}" \
        --test_data_path "${TEMP_DIR}/${i}.csv" \
        --result_json_data "${TEMP_DIR}/${i}.json" \
        --batch_size "${EVAL_BATCH}" \
        --num_beams "${NUM_BEAMS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --length_penalty "${LENGTH_PENALTY}" \
        > "${TEMP_DIR}/${i}.log" 2>&1 &
    pids+=($!)
done

# Fail loudly if any shard died - a partial merge would silently understate metrics.
if (( ${#pids[@]} == 0 )); then
    echo "no eval shards were launched" >&2
    exit 1
fi
failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done
if (( failed )); then
    echo "at least one eval shard failed; tail of each shard log:" >&2
    for i in ${shards}; do
        [[ -f "${TEMP_DIR}/${i}.log" ]] && { echo "--- shard ${i} ---" >&2; tail -20 "${TEMP_DIR}/${i}.log" >&2; }
    done
    exit 1
fi

produced=$(ls "${TEMP_DIR}"/*.json 2>/dev/null | wc -l)
expected=$(wc -w <<< "${shards}")
if (( produced != expected )); then
    echo "expected ${expected} shard results, got ${produced}" >&2
    exit 1
fi

actual_list=$(ls "${TEMP_DIR}"/*.json | sed 's#.*/##; s#\.json$##' | paste -sd, -)
"${ENV_PREFIX}/bin/python" "${REPO}/merge.py" \
    --input_path "${TEMP_DIR}" \
    --output_path "${RESULT_JSON}" \
    --cuda_list "${actual_list}"

[[ -f "${RESULT_JSON}" ]] || { echo "merge produced no output" >&2; exit 1; }

"${ENV_PREFIX}/bin/python" "${CALC_SCRIPT}" \
    --path "${RESULT_JSON}" \
    --item_path "${INFO_FILE}"

rm -rf "${TEMP_DIR}"
echo "eval complete: ${RESULT_JSON}"
