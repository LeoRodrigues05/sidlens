#!/usr/bin/env python3
"""
Convert dataset (Sports/Industrial_and_Scientific) to ReRe format with semantic IDs
"""

import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any
import argparse
import sys

def load_dataset(data_dir: str, dataset_name: str, index_file: str = None) -> Dict[str, Any]:
    """Load all dataset files"""
    data = {}
    
    # Load item metadata (id -> {title, description, ...})
    with open(os.path.join(data_dir, f'{dataset_name}.item.json'), 'r') as f:
        data['items'] = json.load(f)
    
    # Load item_id to semantic tokens mapping from index.json
    with open(os.path.join(data_dir, index_file), 'r') as f:
        data['item_to_semantic'] = json.load(f)
    
    # Load train/valid/test splits
    splits = {}
    for split in ['train', 'valid', 'test']:
        split_file = os.path.join(data_dir, f'{dataset_name}.{split}.inter')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                splits[split] = [line.strip().split('\t') for line in lines if line.strip()]
    
    data['splits'] = splits
    return data

def semantic_tokens_to_id(tokens: List[str]) -> str:
    """Convert semantic tokens list to concatenated string with brackets preserved"""
    # Keep brackets and concatenate directly (no spaces)
    return ''.join(tokens)

def create_item_info_file(items: Dict[str, Dict], item_to_semantic: Dict[str, List], output_path: str):
    """Create item info file (sid -> title -> item_id mapping)"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item_id, item_data in items.items():
            # Get semantic ID from index mapping
            if item_id in item_to_semantic:
                semantic_tokens = item_to_semantic[item_id]
                semantic_id = semantic_tokens_to_id(semantic_tokens)
                # Get item title, fallback to Item_id if not available
                item_title = item_data.get('title', f'Item_{item_id}')
                f.write(f"{semantic_id}\t{item_title}\t{item_id}\n")

def create_item_info_file_t(items: Dict[str, Dict], item_to_semantic: Dict[str, List], output_path: str):
    """Create item info file (sid -> title -> item_id mapping)"""
    refer_list = np.load('./data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.item2new_item_interval.npy', allow_pickle=True)
    new_id_to_old_id = {}
    for row in refer_list:
        old_item_id = int(row[0])
        new_item_id = int(row[2])
        # 注意：如果一个新ID对应多个旧ID，这里只会保留最后一个
        # 如果需要保留时间信息，可以用 (new_id, timestamp) 作为键
        new_id_to_old_id[new_item_id] = old_item_id

    with open(output_path, 'w', encoding='utf-8') as f:
        for new_item_id, old_item_id in new_id_to_old_id.items():
            if old_item_id == -1:
                continue
            if str(new_item_id) in item_to_semantic:
                semantic_tokens = item_to_semantic[str(new_item_id)]
                semantic_id = semantic_tokens_to_id(semantic_tokens)
                # Get item title, fallback to Item_id if not available
                item_data = items[str(old_item_id)]
                item_title = item_data.get('title', f'Item_{old_item_id}')
                f.write(f"{semantic_id}\t{item_title}\t{old_item_id}\n")


def convert_interactions_to_csv(splits: Dict[str, List], items: Dict[str, Dict],
                               item_to_semantic: Dict[str, List], output_dir: str, category: str = "Sports",
                               max_valid_samples: int = None, max_test_samples: int = None, seed: int = 42,
                               keep_longest_only: bool = True, tag: str = "rqkmeans_3codebook_256"):
    """Convert interaction data to ReRe CSV format using semantic IDs"""
    
    import random
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        rows = []
        user_to_longest = {}  # For train data: keep only longest sequence per user
        
        for line in split_data:
            if len(line) != 3:
                continue
                
            user_id, item_sequence, target_item = line
            
            # Parse item sequence - these are item_ids from the interaction data
            if item_sequence.strip():
                history_item_ids = [int(x) for x in item_sequence.split()]
            else:
                history_item_ids = []
            
            target_item_id = int(target_item)
            
            # Convert item_ids to semantic_ids using index mapping
            history_semantic_ids = []
            for item_id in history_item_ids:
                if str(item_id) in item_to_semantic:
                    semantic_tokens = item_to_semantic[str(item_id)]
                    semantic_id = semantic_tokens_to_id(semantic_tokens)
                    history_semantic_ids.append(semantic_id)
            
            # Get target semantic_id
            target_semantic_id = None
            if str(target_item_id) in item_to_semantic:
                semantic_tokens = item_to_semantic[str(target_item_id)]
                target_semantic_id = semantic_tokens_to_id(semantic_tokens)
            
            if target_semantic_id is None:
                continue  # Skip if no semantic_id found for target
            
            # Get item titles using item_ids (not semantic_ids)
            history_item_titles = []
            for item_id in history_item_ids:
                if str(item_id) in items:
                    title = items[str(item_id)].get('title', f'Item_{item_id}')
                    history_item_titles.append(title)
            
            # Target item title
            target_title = items.get(str(target_item_id), {}).get('title', f'Item_{target_item_id}')
            
            # Create row with required fields
            row = {
                'user_id': f'A{user_id}',  # Format like A2013JDMPUV6D9
                'history_item_title': history_item_titles,
                'item_title': target_title,
                'history_item_id': history_item_ids,  # Original item_ids
                'item_id': target_item_id,  # Original item_id
                'history_item_sid': history_semantic_ids,  # Semantic IDs (with brackets)
                'item_sid': target_semantic_id  # Target semantic ID (with brackets)
            }
            
            # For train data: keep only longest sequence per user (if enabled)
            if split_name == 'train' and keep_longest_only:
                sequence_length = len(history_item_ids)
                if user_id not in user_to_longest or sequence_length > len(user_to_longest[user_id]['history_item_id']):
                    user_to_longest[user_id] = row
            else:
                rows.append(row)
        
        # For train data, add only the longest sequences (if enabled)
        if split_name == 'train' and keep_longest_only:
            rows = list(user_to_longest.values())
        
        # Apply sample limits for valid/test sets
        original_count = len(rows)
        if split_name == 'valid' and max_valid_samples is not None and len(rows) > max_valid_samples:
            rows = random.sample(rows, max_valid_samples)
            print(f"  Sampled {max_valid_samples} from {original_count} validation samples")
        elif split_name == 'test' and max_test_samples is not None and len(rows) > max_test_samples:
            rows = random.sample(rows, max_test_samples)
            print(f"  Sampled {max_test_samples} from {original_count} test samples")
        
        # Save to CSV
        if rows:
            df = pd.DataFrame(rows)
            output_file = os.path.join(output_dir, f'{category}_{tag}.csv')
            df.to_csv(output_file, index=False)
            print(f"Created {split_name} file: {output_file} with {len(rows)} rows")
            if split_name == 'train' and keep_longest_only:
                print(f"  Kept longest sequences for {len(rows)} unique users")
            elif split_name == 'train':
                print(f"  Kept all sequences for train data")
            if rows:
                print(f"  Sample row:")
                print(f"    user_id: {rows[0]['user_id']}")
                print(f"    history_item_id: {rows[0]['history_item_id']}")
                print(f"    item_id: {rows[0]['item_id']}")
                print(f"    history_item_sid: {rows[0]['history_item_sid']}")
                print(f"    item_sid: {rows[0]['item_sid']}")
                print(f"    item_title: {rows[0]['item_title'][:50]}...")


def convert_interactions_to_csv_t(splits: Dict[str, List], items: Dict[str, Dict], 
                               item_to_semantic: Dict[str, List], output_dir: str, category: str = "Sports",
                               max_valid_samples: int = None, max_test_samples: int = None, seed: int = 42,
                               keep_longest_only: bool = True):
    """Convert interaction data to ReRe CSV format using semantic IDs"""
    
    import random
    random.seed(seed)

    # 添加这段：创建新ID到旧ID的映射
    refer_list = np.load('./data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.item2new_item_interval.npy', allow_pickle=True)
    new_id_to_old_id = {}
    for row in refer_list:
        old_item_id = int(row[0])
        new_item_id = int(row[2])
        # 注意：如果一个新ID对应多个旧ID，这里只会保留最后一个
        # 如果需要保留时间信息，可以用 (new_id, timestamp) 作为键
        new_id_to_old_id[new_item_id] = old_item_id
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        rows = []
        user_to_longest = {}  # For train data: keep only longest sequence per user
        
        for line in split_data:
            if len(line) != 3:
                continue
                
            user_id, item_sequence, target_item = line
            
            # Parse item sequence - these are item_ids from the interaction data
            if item_sequence.strip():
                history_item_ids = [int(x) for x in item_sequence.split()]
            else:
                history_item_ids = []
            
            target_item_id = int(target_item)
            
            # 如果是 -1 就删掉这条交互记录
            if new_id_to_old_id.get(target_item_id, target_item_id) == -1:
                continue
            if any(new_id_to_old_id.get(item_id, item_id) == -1 for item_id in history_item_ids):
                continue
            
            # Convert item_ids to semantic_ids using index mapping
            history_semantic_ids = []
            for item_id in history_item_ids:
                if str(item_id) in item_to_semantic:
                    semantic_tokens = item_to_semantic[str(item_id)]
                    semantic_id = semantic_tokens_to_id(semantic_tokens)
                    history_semantic_ids.append(semantic_id)
            
            # Get target semantic_id
            target_semantic_id = None
            if str(target_item_id) in item_to_semantic:
                semantic_tokens = item_to_semantic[str(target_item_id)]
                target_semantic_id = semantic_tokens_to_id(semantic_tokens)
            
            if target_semantic_id is None:
                continue  # Skip if no semantic_id found for target
            
            # Get item titles using item_ids (not semantic_ids)
            history_item_titles = []
            for item_id in history_item_ids:
                old_item_id = new_id_to_old_id.get(item_id, item_id)  # 转换成旧ID
                if str(old_item_id) in items:
                    title = items[str(old_item_id)].get('title', f'Item_{old_item_id}')
                    history_item_titles.append(title)
            
            # Target item title
            old_target_id = new_id_to_old_id.get(target_item_id, target_item_id)
            target_title = items.get(str(old_target_id), {}).get('title', f'Item_{old_target_id}')
            
            # Create row with required fields
            row = {
                'user_id': f'A{user_id}',  # Format like A2013JDMPUV6D9
                'history_item_title': history_item_titles,
                'item_title': target_title,
                'history_item_id': history_item_ids,  # Original item_ids
                'item_id': target_item_id,  # Original item_id
                'history_item_sid': history_semantic_ids,  # Semantic IDs (with brackets)
                'item_sid': target_semantic_id  # Target semantic ID (with brackets)
            }
            
            # For train data: keep only longest sequence per user (if enabled)
            if split_name == 'train' and keep_longest_only:
                sequence_length = len(history_item_ids)
                if user_id not in user_to_longest or sequence_length > len(user_to_longest[user_id]['history_item_id']):
                    user_to_longest[user_id] = row
            else:
                rows.append(row)
        
        # For train data, add only the longest sequences (if enabled)
        if split_name == 'train' and keep_longest_only:
            rows = list(user_to_longest.values())
        
        # Apply sample limits for valid/test sets
        original_count = len(rows)
        if split_name == 'valid' and max_valid_samples is not None and len(rows) > max_valid_samples:
            rows = random.sample(rows, max_valid_samples)
            print(f"  Sampled {max_valid_samples} from {original_count} validation samples")
        elif split_name == 'test' and max_test_samples is not None and len(rows) > max_test_samples:
            rows = random.sample(rows, max_test_samples)
            print(f"  Sampled {max_test_samples} from {original_count} test samples")
        
        # Save to CSV
        if rows:
            df = pd.DataFrame(rows)
            output_file = os.path.join(output_dir, f'{category}_interval_tag_separate_2013-10-2018-11.csv')
            df.to_csv(output_file, index=False)
            print(f"Created {split_name} file: {output_file} with {len(rows)} rows")
            if split_name == 'train' and keep_longest_only:
                print(f"  Kept longest sequences for {len(rows)} unique users")
            elif split_name == 'train':
                print(f"  Kept all sequences for train data")
            if rows:
                print(f"  Sample row:")
                print(f"    user_id: {rows[0]['user_id']}")
                print(f"    history_item_id: {rows[0]['history_item_id']}")
                print(f"    item_id: {rows[0]['item_id']}")
                print(f"    history_item_sid: {rows[0]['history_item_sid']}")   
                print(f"    item_sid: {rows[0]['item_sid']}")
                print(f"    item_title: {rows[0]['item_title'][:50]}...")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset (Sports/Industrial_and_Scientific) to ReRe format with semantic IDs')
    parser.add_argument('--data_dir', type=str, 
                       default='./data/Amazon18/Industrial_and_Scientific',
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='Industrial_and_Scientific',
                       help='Dataset name (Sports, Industrial_and_Scientific)')
    parser.add_argument('--output_dir', type=str,
                       default='data/Amazon18',
                       help='Output directory for ReRe format data')
    parser.add_argument('--category', type=str, default=None,
                       help='Category name for output files (if None, will use dataset_name)')
    parser.add_argument('--max_valid_samples', type=int, default=None,
                       help='Maximum number of samples to keep in validation set (None for all)')
    parser.add_argument('--max_test_samples', type=int, default=None,
                       help='Maximum number of samples to keep in test set (None for all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--keep_longest_only', action='store_true', default=False,
                       help='Keep only longest sequence per user in train data (default: False)')
    parser.add_argument('--no_keep_longest_only', dest='keep_longest_only', action='store_false',
                       help='Keep all sequences per user in train data')
    parser.add_argument('--index_file', type=str, default=None,
                       help='Index JSON filename (e.g. Industrial_and_Scientific.index_3codebook_256.json)')
    parser.add_argument('--tag', type=str, default='rqkmeans_3codebook_256',
                       help='Tag for output filenames (e.g. rqvae_3codebook_256, MQ_3codebook_256)')
    
    args = parser.parse_args()
    print(args.data_dir, args.dataset_name)
    
    # Default category to dataset name if not specified
    if args.category is None:
        args.category = args.dataset_name
    
    print(f"Loading {args.dataset_name} data from {args.data_dir}")
    dataset_data = load_dataset(args.data_dir, args.dataset_name, args.index_file)
    # dict, len: 3 dict_keys(['items', 'item_to_semantic', 'splits'])
    print(dataset_data['splits']['train'])
    
    
    print(f"Found {len(dataset_data['items'])} items")
    print(f"Found {len(dataset_data['item_to_semantic'])} item-to-semantic mappings")

    # Sample a few semantic ID conversions
    sample_items = list(dataset_data['item_to_semantic'].items())[:3]
    for item_id, tokens in sample_items:
        semantic_id = semantic_tokens_to_id(tokens)
        print(f"  Item {item_id}: {tokens} -> {semantic_id}")

    
    # Create output directories
    for subdir in ['train', 'valid', 'test', 'info']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # Create item info file
    info_file = os.path.join(args.output_dir, 'info', f'{args.category}_{args.tag}.txt')
    create_item_info_file(dataset_data['items'], dataset_data['item_to_semantic'], info_file)
    print(f"Created item info file: {info_file}")
    
    # Convert each split
    for split_name in ['train', 'valid', 'test']:
        if split_name in dataset_data['splits']:
            split_output_dir = os.path.join(args.output_dir, split_name)
            convert_interactions_to_csv(
                {split_name: dataset_data['splits'][split_name]},
                dataset_data['items'],
                dataset_data['item_to_semantic'],
                split_output_dir,
                args.category,
                max_valid_samples=args.max_valid_samples,
                max_test_samples=args.max_test_samples,
                seed=args.seed,
                keep_longest_only=args.keep_longest_only,
                tag=args.tag
            )
    
    print(f"\nConversion completed! Data saved to {args.output_dir}")
    print(f"You can now use these files with ReRe training scripts by setting category='{args.category}'")
    
    # Show sampling information if limits were applied
    if args.max_valid_samples is not None:
        print(f"Validation set was limited to {args.max_valid_samples} samples")
    if args.max_test_samples is not None:
        print(f"Test set was limited to {args.max_test_samples} samples")
    
    print(f"\nTo run ReRe training:")
    print(f"1. Modify rere.sh to use category='{args.category}'")
    print(f"2. Run: bash rere.sh")
    print(f"\nUsage examples:")
    print(f"  # Convert Sports dataset:")
    print(f"  python convert_dataset_semantic.py --dataset_name Sports --data_dir /path/to/LCRec_data/Sports")
    print(f"  # Convert Industrial_and_Scientific dataset:")
    print(f"  python convert_dataset_semantic.py --dataset_name Industrial_and_Scientific --data_dir /path/to/LCRec_data/Industrial_and_Scientific")
    print(f"  # Keep all sequences (not just longest) for train data:")
    print(f"  python convert_dataset_semantic.py --no_keep_longest_only")
    print(f"  # Limit validation to 1000 samples and test to 500 samples:")
    print(f"  python convert_dataset_semantic.py --max_valid_samples 1000 --max_test_samples 500")

if __name__ == '__main__':
    main()