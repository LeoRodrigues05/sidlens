"""Shared event-disjoint AR CSVs and Diff cohorts from identical user sequences."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_bundle import DATASETS, atomic_json, sha256
from prepare_office_next2_data import _read_contiguous_mapping
import variants as V
from assignment_data import validate_bundle

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = 'disjoint_horizon_v3'
PREPARATION_VERSION = 'analysis-v3'


def splitter():
    spec = importlib.util.spec_from_file_location('shared_horizon', REPO / 'DiffGRM_new/genrec/horizon_split.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_horizon_splits


def source_assets(data_root: Path, dataset: str) -> dict:
    category = DATASETS[dataset]
    base = data_root / 'Amazon18' / category
    return {name: base / f'{category}.{suffix}' for name, suffix in {
        'interactions': 'inter.json', 'items': 'item2id', 'users': 'user2id',
        'metadata': 'item.json'}.items()}


def analysis_assets(data_root: Path, dataset: str) -> tuple[dict, dict]:
    """Identify shared post-training inputs without requiring them for SFT.

    Present files become immutable run inputs. Missing optional analysis inputs
    remain explicit, including when setup used an existing training-only root.
    Paths in the availability record are relative so copies on another machine
    have the same analysis identity. Hashing does not validate embedding rows or
    recover the original quantizer; analyses must check their input semantics.
    """
    category = DATASETS[dataset]
    base = Path('Amazon18') / category
    suffixes = {
        'analysis_item_embeddings': 'emb-qwen-td.npy',
        'analysis_reviews': 'review.json',
        'analysis_historical_train': 'train.inter',
        'analysis_historical_validation': 'valid.inter',
        'analysis_historical_test': 'test.inter',
    }
    paths, available = {}, {}
    for role, suffix in suffixes.items():
        relative = base / f'{category}.{suffix}'
        path = data_root / relative
        if path.exists() and not path.is_file():
            raise ValueError(f'analysis asset must be a file: {path}')
        present = path.is_file()
        if present:
            paths[role] = path
        available[role] = {'path': relative.as_posix(), 'available': present}
    return paths, {
        'schema': 'onediffrec.analysis-assets/v1',
        'assets': available,
        'missing': sorted(role for role, asset in available.items() if not asset['available']),
        'archived_item_embeddings': ('present_unvalidated' if 'analysis_item_embeddings' in paths
                                     else 'missing'),
        'native_quantizer_state': 'not_established_by_these_assets',
    }


def index_path(data_root: Path, dataset: str, method: str, digits: int, size: int) -> Path:
    category = DATASETS[dataset]
    filename = V.INDEX_OVERRIDES.get((category, method, digits, size))
    if filename is None:
        filename = V.INDEX_PATTERNS[method].format(category=category, codebooks=digits, size=size)
    return data_root / 'Amazon18' / category / filename


def read_sources(data_root: Path, dataset: str):
    paths = source_assets(data_root, dataset)
    items = _read_contiguous_mapping(paths['items'], 'item')
    users = _read_contiguous_mapping(paths['users'], 'user')
    interactions = json.loads(paths['interactions'].read_text())
    if not isinstance(interactions, dict) or not interactions:
        raise ValueError('interactions must be a nonempty object')
    sequences = {}
    for user, sequence in interactions.items():
        if not isinstance(user, str) or not user.isdecimal() or str(int(user)) != user:
            raise ValueError(f'non-canonical numeric interaction user: {user!r}')
        user_index = int(user)
        if not 0 <= user_index < len(users):
            raise ValueError(f'interaction user out of range: {user}')
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f'interaction sequence must be a nonempty list: {user}')
        if any(type(item) is not int or not 0 <= item < len(items) for item in sequence):
            raise ValueError(f'interaction item ID must be an in-range integer: {user}')
        raw_user = users[user_index]
        if raw_user in sequences:
            raise ValueError(f'duplicate interaction user: {raw_user}')
        sequences[raw_user] = [items[item] for item in sequence]
    metadata = json.loads(paths['metadata'].read_text())
    if set(metadata) != {str(i) for i in range(len(items))}:
        raise ValueError('metadata keys must match numeric embedding/SID item rows')
    return paths, items, users, sequences, metadata


def read_index(path: Path, items: list[str], digits: int, size: int):
    index = json.loads(path.read_text())
    if set(index) != {str(i) for i in range(len(items))}:
        raise ValueError(f'SID item coverage mismatch: {path}')
    semantic = {}
    for i, tokens in index.items():
        if not isinstance(tokens, list) or len(tokens) != digits:
            raise ValueError(f'wrong SID width: {path}: {i}')
        codes = []
        for stage, token in enumerate(tokens):
            match = re.fullmatch(r'<([a-z])_(\d+)>', token)
            if match is None or match[1] != chr(97 + stage) or int(match[2]) >= size:
                raise ValueError(f'invalid SID code: {path}: {token}')
            codes.append(int(match[2]))
        semantic[items[int(i)]] = codes
    return index, semantic


def build_cache(data_root: Path, cache_root: Path, dataset: str,
                selected_indexes: dict[str, Path] | None = None) -> dict:
    paths, items, users, sequences, _ = read_sources(data_root, dataset)
    category = 'Office_Products' if dataset == 'office' else DATASETS[dataset]
    processed = cache_root / 'AmazonReviews2014' / category / 'processed'
    processed.mkdir(parents=True, exist_ok=True)
    values = {
        'all_item_seqs.json': sequences,
        'id_mapping.json': {'user2id': {'[PAD]': 0, **{u:i+1 for i,u in enumerate(users)}},
                           'item2id': {'[PAD]': 0, **{u:i+1 for i,u in enumerate(items)}},
                           'id2user': ['[PAD]', *users], 'id2item': ['[PAD]', *items]},
    }
    for variant in V.enumerate_variants('next-item', categories=[DATASETS[dataset]]):
        name = f'{variant.method}_{variant.codebooks}codebook_{variant.size}'
        if selected_indexes is not None and name not in selected_indexes:
            continue
        path = (selected_indexes[name] if selected_indexes is not None else
                index_path(data_root, dataset, variant.method, variant.codebooks, variant.size))
        _, semantic = read_index(path, items, variant.codebooks, variant.size)
        values[f'{variant.method}_{variant.codebooks}codebook_{variant.size}.sem_ids'] = semantic
    for name, value in values.items():
        path = processed / name
        content = json.dumps(value).encode()
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f'existing cache differs; choose a new cache root: {path}')
        else:
            temporary = path.with_name('.' + path.name + '.tmp')
            temporary.write_bytes(content)
            temporary.replace(path)
    return {'processed': str(processed), 'users': len(users), 'items': len(items), 'files': len(values)}


def canonical_examples(sequences: dict, horizon: int) -> dict:
    rows = splitter()(sequences, n_target_items=horizon, min_hist_len=2, max_hist_len=50,
                      train_sliding=True)
    result = {}
    for split, examples in rows.items():
        result[split] = []
        for row in examples:
            identity = [PROTOCOL, horizon, str(row['user']),
                        row['history_event_positions'], row['target_event_positions']]
            result[split].append({
                'example_id': hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest(),
                'user_id': str(row['user']), 'history_item_ids': row['item_seq'][:-horizon],
                'target_item_ids': row['item_seq'][-horizon:],
                'history_event_positions': row['history_event_positions'],
                'target_event_positions': row['target_event_positions'],
                'split_protocol': PROTOCOL,
            })
    return result


def prepare_cell(data_root: Path, output: Path, dataset: str, predtype: str,
                 method: str, digits: int, size: int, *,
                 sid_assignment: str = 'native', zcr_bundle: Path | None = None) -> dict:
    if predtype not in ('next-item', 'next2item'):
        raise ValueError('unsupported prediction horizon')
    if sid_assignment not in ('native', 'zcr'):
        raise ValueError('sid_assignment must be native or zcr')
    horizon = 1 if predtype == 'next-item' else 2
    paths, items, users, sequences, metadata = read_sources(data_root, dataset)
    shared_analysis_paths, analysis_status = analysis_assets(data_root, dataset)
    paths.update(shared_analysis_paths)
    paths['sid_index'] = index_path(data_root, dataset, method, digits, size)
    assignment_id = 'native'
    if sid_assignment == 'zcr':
        if zcr_bundle is None:
            raise ValueError('a verified ZCR bundle is required')
        assignment = validate_bundle(zcr_bundle, paths['sid_index'], dataset, method, digits, size)
        paths.update(assignment['paths'])
        paths['sid_index'] = assignment['index_path']
        assignment_id = assignment['assignment_id']
    elif zcr_bundle is not None:
        raise ValueError('native preparation cannot use a ZCR bundle')
    paths['splitter'] = REPO / 'DiffGRM_new/genrec/horizon_split.py'
    paths['builder'] = Path(__file__)
    input_hashes = {key: sha256(path) for key, path in paths.items()}
    manifest_path = output / 'data-manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if (prior['input_hashes'] != input_hashes or prior['predtype'] != predtype
                or prior.get('sid_assignment', 'native') != sid_assignment
                or prior.get('preparation_version') != PREPARATION_VERSION
                or prior.get('analysis_assets') != analysis_status):
            raise ValueError(f'prepared data differs: {output}; use a new work/version root')
        for name, digest in prior['output_hashes'].items():
            if sha256(output / name) != digest:
                raise ValueError(f'prepared output changed: {name}')
        return prior
    if output.exists():
        raise ValueError(f'incomplete prepared output; choose a fresh path: {output}')
    index, _ = read_index(paths['sid_index'], items, digits, size)
    raw_to_id = {raw: i for i, raw in enumerate(items)}
    examples = canonical_examples(sequences, horizon)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.' + output.name, dir=output.parent))
    try:
        fields = ['user_id','history_item_title','item_title','history_item_id','item_id',
                  'history_item_sid','item_sid','example_id','split_protocol',
                  'history_event_positions','target_event_positions',
                  'raw_history_item_ids','raw_target_item_ids']
        for split, rows in examples.items():
            with (temp / f'{split}.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for row in rows:
                    hist = [raw_to_id[i] for i in row['history_item_ids']]
                    targets = [raw_to_id[i] for i in row['target_item_ids']]
                    writer.writerow({
                        'user_id': row['user_id'], 'example_id': row['example_id'],
                        'history_item_title': repr([metadata[str(i)]['title'] for i in hist]),
                        'item_title': ' ||| '.join(metadata[str(i)]['title'] for i in targets),
                        'history_item_id': repr(hist), 'item_id': ' ||| '.join(map(str, targets)),
                        'history_item_sid': repr([''.join(index[str(i)]) for i in hist]),
                        'item_sid': ' ||| '.join(''.join(index[str(i)]) for i in targets),
                        'split_protocol': PROTOCOL,
                        'history_event_positions': json.dumps(row['history_event_positions']),
                        'target_event_positions': json.dumps(row['target_event_positions']),
                        'raw_history_item_ids': json.dumps(row['history_item_ids']),
                        'raw_target_item_ids': json.dumps(row['target_item_ids']),
                    })
            atomic_json(temp / f'{split}-cohort.json', {'schema': 'onediffrec.canonical-cohort/v1',
                        'split': split, 'split_protocol': PROTOCOL, 'horizon': horizon,
                        'examples': rows})
        manifest = {'schema': 'onediffrec.prepared-cell/v3', 'dataset': dataset,
                    'predtype': predtype, 'split_protocol': PROTOCOL,
                    'sid_assignment': sid_assignment, 'assignment_id': assignment_id,
                    'preparation_version': PREPARATION_VERSION,
                    'analysis_assets': analysis_status,
                    'variant': f'{method}_{digits}codebook_{size}',
                    'input_paths': {k:str(p.resolve()) for k,p in paths.items()},
                    'input_hashes': input_hashes,
                    'rows': {k:len(v) for k,v in examples.items()},
                    'output_hashes': {p.name:sha256(p) for p in sorted(temp.iterdir())}}
        atomic_json(temp / 'data-manifest.json', manifest)
        temp.rename(output)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    return manifest
