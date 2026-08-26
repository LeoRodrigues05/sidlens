"""
Convert single-target CSV to two-target CSV.
Move the last history item to become the first target item.

Input CSV columns:
  user_id, history_item_title, item_title, history_item_id, item_id, history_item_sid, item_sid

Output CSV columns:
  user_id, history_item_title, item_title, history_item_id, item_id, history_item_sid, item_sid

Changes:
  - history_item_title: remove last element
  - history_item_id: remove last element
  - history_item_sid: remove last element
  - item_title: "moved_title ||| original_title"
  - item_id: "moved_id ||| original_id"
  - item_sid: "moved_sid ||| original_sid"
  - Samples with history length == 1 are dropped (no history left after moving)
"""

import argparse
import ast
import csv
import glob
import os
import sys


def convert_csv(input_path, output_path):
    total = 0
    kept = 0
    dropped = 0

    with open(input_path, 'r') as fin, open(output_path, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            total += 1
            user_id = row[0]
            history_titles = ast.literal_eval(row[1])
            item_title = row[2]
            history_ids = ast.literal_eval(row[3])
            item_id = row[4]
            history_sids = ast.literal_eval(row[5])
            item_sid = row[6]

            if len(history_ids) < 2:
                dropped += 1
                continue

            # Move last history item to first target
            moved_title = history_titles[-1]
            moved_id = history_ids[-1]
            moved_sid = history_sids[-1]

            new_history_titles = history_titles[:-1]
            new_history_ids = history_ids[:-1]
            new_history_sids = history_sids[:-1]

            # Two targets separated by |||
            new_item_title = f"{moved_title} ||| {item_title}"
            new_item_id = f"{moved_id} ||| {item_id}"
            new_item_sid = f"{moved_sid} ||| {item_sid}"

            writer.writerow([
                user_id,
                str(new_history_titles),
                new_item_title,
                str(new_history_ids),
                new_item_id,
                str(new_history_sids),
                new_item_sid,
            ])
            kept += 1

    print(f"  {os.path.basename(input_path)}: total={total}, kept={kept}, dropped={dropped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data1/dnian/OneDiffRec/data/Amazon18")
    parser.add_argument("--output_dir", type=str, default="/data1/dnian/OneDiffRec/data/two-item")
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        in_split_dir = os.path.join(args.input_dir, split)
        out_split_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)

        csv_files = sorted(glob.glob(os.path.join(in_split_dir, "*.csv")))
        if not csv_files:
            print(f"No CSV files found in {in_split_dir}")
            continue

        print(f"\n=== {split} ===")
        for csv_file in csv_files:
            output_file = os.path.join(out_split_dir, os.path.basename(csv_file))
            convert_csv(csv_file, output_file)

    # Copy info files as-is
    in_info_dir = os.path.join(args.input_dir, "info")
    out_info_dir = os.path.join(args.output_dir, "info")
    os.makedirs(out_info_dir, exist_ok=True)
    info_files = glob.glob(os.path.join(in_info_dir, "*.txt"))
    for f in info_files:
        import shutil
        shutil.copy2(f, os.path.join(out_info_dir, os.path.basename(f)))
    print(f"\nCopied {len(info_files)} info files")


if __name__ == "__main__":
    main()
