# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Dict, List, Any


def block_masking(decoder_input_ids: torch.Tensor, n_digit: int, block_size: int):
    """
    Block-aware masking for Block Diffusion training.

    Randomly sample a starting block t for each sample.
    Blocks [0, t-1] are kept clean (given ground truth).
    Blocks [t, n_blocks-1] are all masked (set to 0).

    Example: n_digit=4, block_size=2, block_t=1
        clean GT: [23, 187,  5, 142]
        masked:   [23, 187,  0,   0]
        mask:     [ F,   F,  T,   T]

    Args:
        decoder_input_ids: [B, n_digit] clean ground truth codebook IDs
        n_digit: total number of digits
        block_size: number of digits per block

    Returns:
        masked_input:  [B, n_digit]
        mask_positions: [B, n_digit] bool, True=masked
    """
    B = decoder_input_ids.shape[0]
    n_blocks = n_digit // block_size

    # block_t=0 → all blocks masked; block_t=n_blocks-1 → only last block masked
    block_t = torch.randint(0, n_blocks, (B,), device=decoder_input_ids.device)

    mask_positions = torch.zeros(B, n_digit, dtype=torch.bool, device=decoder_input_ids.device)
    for b in range(B):
        start = int(block_t[b].item()) * block_size
        mask_positions[b, start:] = True

    masked_input = decoder_input_ids.clone()
    masked_input[mask_positions] = 0
    return masked_input, mask_positions


def stack_to_tensor(seq, dtype=None):
    """通用工具：将所有元素stack成tensor；若传入dtype则做类型转换。"""
    if torch.is_tensor(seq[0]):
        out = torch.stack(seq, dim=0)
        return out.to(dtype) if dtype is not None else out
    return torch.tensor(seq, dtype=(dtype if dtype is not None else torch.long))


def collate_fn_train(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    训练时的collate函数
    
    Args:
        batch: 包含以下字段的字典列表：
            - history_sid: 历史SID序列 [seq_len, n_digit]
            - history_mask: 历史掩码 [seq_len]
            - decoder_input_ids: decoder输入 [n_digit]
            - decoder_labels: decoder标签 [n_digit]
    
    Returns:
        批处理后的字典
    """
    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),                      # [B, S, n_digit]
        'history_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),  # [B, S]
        'decoder_input_ids': stack_to_tensor([b['decoder_input_ids'] for b in batch]),          # [B, n_digit]
        'decoder_labels': stack_to_tensor([b['decoder_labels'] for b in batch]),                # [B, n_digit]
    }


def collate_fn_val(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    验证时的collate函数
    
    Args:
        batch: 包含以下字段的字典列表：
            - history_sid: 历史SID序列 [seq_len, n_digit]
            - history_mask: 历史掩码 [seq_len]
            - labels: 真标签序列 [n_digit]
    
    Returns:
        批处理后的字典
    """
    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),
        'history_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),
        'labels': stack_to_tensor([b['labels'] for b in batch]),
    }


def collate_fn_test(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    测试时的collate函数（与验证相同）
    """
    return collate_fn_val(batch) 