"""
Convert OneDiffRec interaction data to DiffGRM format (all_item_seqs.json + id_mapping.json).

DiffGRM format:
- all_item_seqs.json: {user_raw_id: [item_asin1, item_asin2, ...], ...}
- id_mapping.json: {
    "user2id": {"[PAD]": 0, "user_raw1": 1, ...},
    "item2id": {"[PAD]": 0, "asin1": 1, ...},
    "id2user": ["[PAD]", "user_raw1", ...],
    "id2item": ["[PAD]", "asin1", ...]
  }
"""

import json
import argparse


def convert(inter_file, user2id_file, item2id_file, output_dir):
    # Load user2id (raw_user -> numeric_id)
    uid2raw = {}
    with open(user2id_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            raw_user, uid = parts[0], parts[1]
            uid2raw[uid] = raw_user

    # Load item2id (asin -> numeric_id)
    iid2raw = {}
    with open(item2id_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            asin, iid = parts[0], parts[1]
            iid2raw[iid] = asin

    # Load inter.json (numeric_user_id -> [numeric_item_ids])
    with open(inter_file, 'r') as f:
        inter_data = json.load(f)

    # Build all_item_seqs: raw_user -> [asin1, asin2, ...]
    all_item_seqs = {}
    for uid, item_ids in inter_data.items():
        raw_user = uid2raw[uid]
        asins = [iid2raw[str(iid)] for iid in item_ids]
        all_item_seqs[raw_user] = asins

    # Build id_mapping (DiffGRM format, ID starts from 1, 0 reserved for [PAD])
    id2user = ['[PAD]']
    id2item = ['[PAD]']
    user2id = {'[PAD]': 0}
    item2id = {'[PAD]': 0}

    # Add users in order
    for uid in sorted(inter_data.keys(), key=int):
        raw_user = uid2raw[uid]
        user2id[raw_user] = len(id2user)
        id2user.append(raw_user)

    # Add items in order
    for iid in sorted(iid2raw.keys(), key=int):
        asin = iid2raw[iid]
        item2id[asin] = len(id2item)
        id2item.append(asin)

    id_mapping = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }

    # Save
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'all_item_seqs.json'), 'w') as f:
        json.dump(all_item_seqs, f)

    with open(os.path.join(output_dir, 'id_mapping.json'), 'w') as f:
        json.dump(id_mapping, f)

    print(f'Users: {len(all_item_seqs)}, Items: {len(item2id) - 1}')
    print(f'id2user length: {len(id2user)}, id2item length: {len(id2item)}')
    sample_user = list(all_item_seqs.keys())[0]
    print(f'Sample: user={sample_user}, items={all_item_seqs[sample_user][:3]}...')
    print(f'Saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inter_file', type=str, required=True)
    parser.add_argument('--user2id_file', type=str, required=True)
    parser.add_argument('--item2id_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    convert(args.inter_file, args.user2id_file, args.item2id_file, args.output_dir)
