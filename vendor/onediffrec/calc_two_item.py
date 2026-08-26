import fire
import json
import math
import numpy as np
from tqdm import tqdm


def normalize_sid(s):
    return s.replace(" ", "").strip().strip('\n"')


def compute_pair_relevance(pred1, pred2, gt1, gt2):
    """Both hit = 1.0, one hit = 0.5, none = 0.0."""
    hit1 = normalize_sid(pred1) == normalize_sid(gt1)
    hit2 = normalize_sid(pred2) == normalize_sid(gt2)
    if hit1 and hit2:
        return 1.0
    elif hit1 or hit2:
        return 0.5
    else:
        return 0.0


def compute_ndcg(relevances, k):
    """NDCG@k: find best relevance in top-k and discount by its position.
    IDCG is always 1.0 (theoretical ideal: relevance=1.0 at position 0).
    """
    best_rel = 0.0
    best_pos = 0
    for i in range(min(k, len(relevances))):
        if relevances[i] > best_rel:
            best_rel = relevances[i]
            best_pos = i
    if best_rel == 0:
        return 0.0
    return best_rel / math.log2(best_pos + 2)


def compute_hr(relevances, k):
    return max(relevances[:k]) if k <= len(relevances) else max(relevances)


def gao(path, item_path):
    if isinstance(path, str):
        path = [path]

    if item_path.endswith(".txt"):
        item_path = item_path[:-4]

    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    f.close()
    # info lines are "sid \t title \t item_id". Stripping only the last field
    # left the key as "sid\ttitle", so no prediction ever matched and the
    # legality counter below flagged every beam (and every ground truth).
    # Metrics never consult item_dict, so only the diagnostic was affected.
    item_names = [_.split('\t')[0].strip() for _ in items]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [i]
        else:
            item_dict[item_names[i]].append(i)

    topk_list = [3, 5, 10, 20]

    for p in path:
        with open(p, 'r') as f:
            samples = json.load(f)

        print(f"\nFile: {p}")
        print(f"Total samples: {len(samples)}")

        hr_sum = {k: 0.0 for k in topk_list}
        ndcg_sum = {k: 0.0 for k in topk_list}
        total = 0
        CC = 0

        for sample in tqdm(samples):
            gt1 = sample["gt1"]
            gt2 = sample["gt2"]
            top_pairs = sample["top_pairs"]

            for pair in top_pairs:
                p1 = normalize_sid(pair["pred1"])
                p2 = normalize_sid(pair["pred2"])
                if p1 not in item_dict:
                    CC += 1
                    print(f"Invalid pred1: {pair['pred1']}")
                if p2 not in item_dict:
                    CC += 1
                    print(f"Invalid pred2: {pair['pred2']}")

            relevances = [
                compute_pair_relevance(pair["pred1"], pair["pred2"], gt1, gt2)
                for pair in top_pairs
            ]

            if len(relevances) == 0:
                # No valid pairs (e.g. all candidates illegal), treat as all-miss
                for k in topk_list:
                    hr_sum[k] += 0.0
                    ndcg_sum[k] += 0.0
            else:
                for k in topk_list:
                    hr_sum[k] += compute_hr(relevances, k)
                    ndcg_sum[k] += compute_ndcg(relevances, k)
            total += 1

        print(f"{'=' * 50}")
        print(f"Two-pass Results ({total} samples)")
        print(f"{'=' * 50}")
        for k in topk_list:
            hr_avg = hr_sum[k] / total if total > 0 else 0
            ndcg_avg = ndcg_sum[k] / total if total > 0 else 0
            print(f"  HR@{k}: {hr_avg:.4f}    NDCG@{k}: {ndcg_avg:.4f}")
        print(f"{'=' * 50}")
        print(f"Invalid predictions: {CC}")


if __name__ == '__main__':
    fire.Fire(gao)
