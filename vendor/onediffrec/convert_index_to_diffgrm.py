"""
Convert OneDiffRec index files to DiffGRM .sem_ids format.

OneDiffRec format: {"0": ["<a_107>", "<b_229>", "<c_29>"], ...}  (key=item_id, value=prefixed tokens)
DiffGRM format:    {"B0002YTLFE": [107, 229, 29], ...}           (key=ASIN, value=plain integers)
"""

import json
import re
import argparse


def convert(index_file, item2id_file, output_file):
    # Load item2id mapping (ASIN -> item_id), build reverse mapping (item_id -> ASIN)
    id2item = {}
    with open(item2id_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            asin, item_id = parts[0], parts[1]
            id2item[item_id] = asin

    # Load OneDiffRec index
    with open(index_file, 'r') as f:
        index_data = json.load(f)

    # Convert
    sem_ids = {}
    for item_id, tokens in index_data.items():
        if item_id not in id2item:
            print(f"Warning: item_id {item_id} not found in item2id mapping, skipping")
            continue
        asin = id2item[item_id]
        # Extract numbers from tokens like "<a_107>" -> 107
        codes = []
        for token in tokens:
            match = re.search(r'(\d+)', token)
            if match:
                codes.append(int(match.group(1)))
        sem_ids[asin] = codes

    with open(output_file, 'w') as f:
        json.dump(sem_ids, f)

    print(f"Converted {len(sem_ids)} items")
    print(f"Sample: {list(sem_ids.items())[:3]}")
    print(f"Saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str, required=True)
    parser.add_argument('--item2id_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    convert(args.index_file, args.item2id_file, args.output_file)
