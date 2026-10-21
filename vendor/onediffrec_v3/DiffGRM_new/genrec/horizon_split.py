"""Dependency-free, event-disjoint rolling-origin recommendation splits.

Shared by model data loading and canonical cross-paradigm cohort preparation.
Event positions are zero-based within each user's original chronological list.
"""


def build_horizon_splits(
    all_item_seqs,
    *,
    n_target_items=2,
    min_hist_len=2,
    max_hist_len=50,
    train_sliding=True,
):
    """Return split rows with capped histories and exact event provenance.

    For horizon h and sequence length L, test targets are [L-h, L), validation
    targets are [L-2h, L-h), and training targets end at or before L-2h. Each
    split checks minimum history independently: short users may be test-only.
    Sliding training includes every eligible origin, using the latest
    max_hist_len history events at that origin. Adjacent *training* targets may
    overlap; target events never cross train/validation/test boundaries.

    Earlier targets may be observed history for a later origin. Repeated item
    IDs are allowed: disjointness is defined over (user, event position).
    """
    horizon = int(n_target_items)
    minimum, maximum = int(min_hist_len), int(max_hist_len)
    if horizon < 1 or minimum < 1 or maximum < minimum:
        raise ValueError(
            'disjoint_horizon_v3 requires n_target_items >= 1 and '
            '1 <= min_hist_len <= max_hist_len'
        )
    if not isinstance(train_sliding, bool):
        raise ValueError('train_sliding must be a boolean')
    splits = {name: [] for name in ('train', 'val', 'test')}

    def append(name, user, sequence, target_start):
        if target_start < minimum:
            return
        history_start = max(0, target_start - maximum)
        target_end = target_start + horizon
        splits[name].append({
            'user': user,
            'item_seq': sequence[history_start:target_end],
            'history_event_positions': list(range(history_start, target_start)),
            'target_event_positions': list(range(target_start, target_end)),
        })

    for user, sequence in all_item_seqs.items():
        length = len(sequence)
        append('test', user, sequence, length - horizon)
        append('val', user, sequence, length - 2 * horizon)
        last_train_start = length - 3 * horizon
        starts = (
            range(minimum, last_train_start + 1)
            if train_sliding else [last_train_start]
        )
        for target_start in starts:
            append('train', user, sequence, target_start)
    return splits
