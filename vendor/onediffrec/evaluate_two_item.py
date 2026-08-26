"""
Two-item evaluation: predict 2 target items per sample.
Strategy: two-pass generation.
  Pass 1: generate first item with constrained decoding (beam search, top-K beams)
  Pass 2: for each beam from pass 1, append pred1 to input and generate second item
Output format compatible with calc_two_item.py:
  [{gt1, gt2, top_pairs: [{pred1, pred2}, ...]}, ...]
"""

import pandas as pd
import fire
import torch
import json
import os
import copy
import random
import numpy as np
from tqdm import tqdm
from transformers import (
    GenerationConfig, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
)
from data import EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(
    base_model: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 1,
    seed: int = 42,
    length_penalty: float = 0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
    top_k_pairs: int = 20,
):
    set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    category_name = category_dict.get(category, category)
    print(f"Category: {category_name}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    model = model.to(device)

    # Load info file and build constraint hash_dict
    with open(info_file, 'r') as f:
        info = f.readlines()
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        info_semantic = [f'### Response:\n{_}' for _ in semantic_ids]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3

    # Build hash_dict for constrained decoding
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    # Tokens of "### Response:\n". Pass 2 resumes generation after the first
    # predicted item, so it has to name this prefix explicitly for the
    # constraint to apply.
    response_prefix = prefixID[0][:prefix_index]
    separator_tokens = tokenizer.encode(" ||| ", add_special_tokens=False)

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    # Load test data
    val_dataset = EvalSidDataset(
        train_file=test_data_path, tokenizer=tokenizer, max_len=2560,
        category=category_name, test=True, seed=seed
    )
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    # Parse gt1 and gt2 from test_data
    for td in test_data:
        output_raw = td["output"].strip()
        if '|||' in output_raw:
            parts = [p.strip() for p in output_raw.split('|||')]
            td["gt1"] = parts[0]
            td["gt2"] = parts[1]
        else:
            # Fallback: single target (shouldn't happen for two-item data)
            td["gt1"] = output_raw
            td["gt2"] = ""

    def generate_single_pass(input_ids_list, num_beams, max_new_tokens, length_penalty):
        """Run beam search for a batch of inputs, return top-K decoded outputs per sample."""
        maxLen = max([len(ids) for ids in input_ids_list])

        padded_ids = []
        attention_mask = []
        for ids in input_ids_list:
            L = len(ids)
            padded_ids.append([tokenizer.pad_token_id] * (maxLen - L) + ids)
            attention_mask.append([0] * (maxLen - L) + [1] * L)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
        )

        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor(padded_ids).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )

        completions = generation_output.sequences[:, maxLen:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        decoded = [_.split("Response:\n")[-1].strip() for _ in decoded]

        # Group by sample
        results = []
        for i in range(len(input_ids_list)):
            sample_outputs = decoded[i * num_beams: (i + 1) * num_beams]
            results.append(sample_outputs)
        return results

    def generate_second_item(input_ids, pred1_text):
        """Given original input_ids and pred1 text, generate pred2."""
        # Append pred1 + separator tokens to input
        pred1_tokens = tokenizer.encode(pred1_text, add_special_tokens=False)
        # The model is trained on "item1 ||| item2", so the separator is supplied
        # explicitly and the second item is then generated under the same
        # constraint as the first.
        extended_ids = input_ids + pred1_tokens + separator_tokens

        generation_config = GenerationConfig(
            num_beams=1,
            length_penalty=length_penalty,
            num_return_sequences=1,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
        )

        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=1,
                base_model=base_model,
                prefix_key=response_prefix
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor([extended_ids]).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                logits_processor=logits_processor,
            )

        completion = generation_output.sequences[0, len(extended_ids):]
        decoded = tokenizer.decode(completion, skip_special_tokens=True).strip()
        return decoded

    # Main evaluation loop
    print(f"Total samples: {len(encodings)}")
    print(f"Two-pass evaluation: pass1 beam={num_beams}, top_k_pairs={top_k_pairs}")

    all_results = []
    for idx in tqdm(range(len(encodings))):
        encoding = encodings[idx]
        td = test_data[idx]
        input_ids = encoding["input_ids"]

        # Pass 1: generate first item candidates
        pass1_outputs = generate_single_pass(
            [input_ids], num_beams=num_beams,
            max_new_tokens=max_new_tokens, length_penalty=length_penalty
        )[0]

        # Deduplicate pass1 outputs, keep order
        seen = set()
        unique_pass1 = []
        for pred in pass1_outputs:
            if pred not in seen:
                seen.add(pred)
                unique_pass1.append(pred)
            if len(unique_pass1) >= top_k_pairs:
                break

        # Pass 2: for each pass1 candidate, generate second item
        top_pairs = []
        for pred1 in unique_pass1:
            pred2 = generate_second_item(input_ids, pred1)
            top_pairs.append({"pred1": pred1, "pred2": pred2})

        result = {
            "gt1": td["gt1"],
            "gt2": td["gt2"],
            "top_pairs": top_pairs,
        }
        all_results.append(result)

    # Save results
    with open(result_json_data, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved {len(all_results)} results to {result_json_data}")


if __name__ == '__main__':
    fire.Fire(main)
