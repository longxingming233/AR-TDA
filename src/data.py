"""Generic BPR dataset: returns (user, pos_item, neg_item) triples.

Equivalent to EaMRec's BPRDataset(neg_mode='random'):
  - interactions keys are coerced to int;
  - negative samples are drawn as list(all_items - bought) + random.choice
    (identical to EaMRec, ensuring reproducibility).
"""
import random
import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    def __init__(self, interactions, n_items, excluded_interactions=None):
        """
        interactions: dict {user_id: [item_ids]} (training positives)
        n_items: total number of items (excluding padding)
        excluded_interactions: all known positive interactions to exclude during negative sampling
        """
        self.interactions = {int(k): set(v) for k, v in interactions.items()}
        source = excluded_interactions if excluded_interactions is not None else interactions
        self.excluded_interactions = {int(k): set(v) for k, v in source.items()}
        self.n_items = n_items
        self.all_items = set(range(1, n_items + 1))
        self.samples = [
            (u, p) for u, items in self.interactions.items()
            if len(self.excluded_interactions.get(u, ())) < n_items for p in items
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, p = self.samples[idx]
        # Identical to the implementation in EaMRec data.py:
        # neg_candidates = list(self.all_items - self.buy_interactions[user])
        # neg_item = random.choice(neg_candidates)
        neg_candidates = list(self.all_items - self.excluded_interactions.get(u, set()))
        n = random.choice(neg_candidates)
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(p, dtype=torch.long),
            torch.tensor(n, dtype=torch.long),
        )


class InteractionUserDataset(Dataset):
    """Official HGIB evaluation protocol: repeat the corresponding user once per test positive interaction."""
    def __init__(self, interactions):
        self.users = [
            int(user) for user, items in interactions.items()
            for _ in set(items)
        ]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long)


class UserDataset(Dataset):
    def __init__(self, users):
        self.users = sorted({int(user) for user in users})

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long)


"""MULE / EaMRec-style multi-behaviour graph data loading.

Expected dataset directory structure:
    data_dir/dataset/
        statistics.json    # contains n_users / n_items / bsg_types / tcb_types / tib_types
        ubg.txt            # user-item bipartite graph edge list (one "u i" per line)
        <behavior>.txt     # edge list for each behaviour
        train.json         # {user: [item_ids]} (target behaviour, usually buy)
        test.json          # {user: [item_ids]}

relation_dict is also built for each behaviour:
    relation_dict          dict[str -> torch.sparse_coo_tensor]   shape=[U+1, I+1]
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

def _convert_edge(edge_list, n_users):
    """Convert the [u, i] edge list into undirected homogeneous graph edges (i is offset by n_users+1 so that users and items share one id space)."""
    edge_list = edge_list.clone()
    edge_list[1] += n_users + 1
    edge_list = torch.cat([edge_list, edge_list.flip(0)], dim=1)
    return edge_list


def _load_raw_edges(data_dir, behavior):
    """Load the raw (u, i) edge list from <behavior>.txt (without id offset) and return an [N, 2] int64 array."""
    path = os.path.join(data_dir, f'{behavior}.txt')
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.loadtxt(path, dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f'Invalid edge file {path}: expected two integer columns')
    return arr


def _inject_random_noise_edges(edges, n_users, n_items, ratio, seed, excluded_edges=None):
    """Randomly add user-item noise edges; ratio is relative to the original number of edges for that behaviour."""
    if edges is None or ratio <= 0:
        return edges, 0
    n_noise = int(round(len(edges) * ratio))
    if n_noise <= 0:
        return edges, 0

    rng = np.random.default_rng(seed)
    existing = set(map(tuple, edges.tolist()))
    if excluded_edges:
        existing.update(excluded_edges)
    noise = []
    max_trials = max(n_noise * 20, 1000)
    trials = 0
    while len(noise) < n_noise and trials < max_trials:
        trials += 1
        u = int(rng.integers(1, n_users + 1))
        i = int(rng.integers(1, n_items + 1))
        pair = (u, i)
        if pair in existing:
            continue
        existing.add(pair)
        noise.append(pair)

    if not noise:
        return edges, 0
    noise_arr = np.asarray(noise, dtype=np.int64)
    return np.vstack([edges, noise_arr]), len(noise)


def _build_sparse_relation(edges, n_users, n_items, device):
    """Build the sparse user-item relation matrix: shape = [n_users+1, n_items+1] (ids start at 1)."""
    if edges is None or len(edges) == 0:
        return torch.sparse_coo_tensor(
            indices=torch.zeros((2, 0), dtype=torch.long),
            values=torch.zeros(0, dtype=torch.float),
            size=(n_users + 1, n_items + 1),
        ).to(device).coalesce()
    idx = torch.from_numpy(edges.T).long()
    vals = torch.ones(idx.shape[1], dtype=torch.float)
    return torch.sparse_coo_tensor(
        indices=idx, values=vals,
        size=(n_users + 1, n_items + 1),
    ).to(device).coalesce()


def _split_train_validation(interactions, ratio, seed):
    """Deterministically hold out a validation set from the training positives per user; single-interaction users stay in the training set only."""
    if not 0.0 < ratio < 1.0:
        raise ValueError('validation_ratio must be between 0 and 1')
    rng = np.random.default_rng(seed)
    train, valid = {}, {}
    for user in sorted(interactions):
        items = list(dict.fromkeys(interactions[user]))
        if len(items) < 2:
            train[user] = items
            continue
        n_valid = min(max(1, int(round(len(items) * ratio))), len(items) - 1)
        positions = set(rng.choice(len(items), size=n_valid, replace=False).tolist())
        train[user] = [item for idx, item in enumerate(items) if idx not in positions]
        valid[user] = [item for idx, item in enumerate(items) if idx in positions]
    return train, valid


def _build_relation_dict(data_dir, behaviors, n_users, n_items, device, raw_edge_dict=None):
    """Build the multi-behaviour sparse user-item relation matrices."""
    raw_edge_dict = raw_edge_dict or {}
    relation_dict = {}
    for b in behaviors:
        edges = raw_edge_dict.get(b)
        if edges is None:
            edges = _load_raw_edges(data_dir, b)
        if edges is None:
            continue
        relation_dict[b] = _build_sparse_relation(edges, n_users, n_items, device)
    return relation_dict


def load_multi_behavior_data(data_dir, dataset, device, batch_size, num_workers=4,
                             noise_ratio=0.0, noise_behaviors='view', noise_seed=42,
                             model='mule', eval_batch_size=256,
                             load_relations=None,
                             validation_ratio=0.1, split_seed=42,
                             experiment_protocol='hgib'):
    """Load multi-behaviour data; by default uses the official HGIB pre-built train/test protocol."""
    data_dir = os.path.join(data_dir, dataset)
    with open(os.path.join(data_dir, 'statistics.json'), 'r', encoding='utf-8') as f:
        stats = json.load(f)

    n_users, n_items = stats['n_users'], stats['n_items']
    bsg = stats.get('bsg_types', [])
    # support both field naming schemes: MULE uses tcb/tib, EaMRec uses noisy_aux/confident
    tcb = stats.get('tcb_types') or stats.get('noisy_aux_types', [])
    tib = stats.get('tib_types') or stats.get('confident_behaviors', [])

    noise_ratio = float(noise_ratio or 0.0)
    noise_targets = {b.strip() for b in str(noise_behaviors).split(',') if b.strip()}
    raw_edge_dict = {}
    noise_stats = {}

    behaviors = list(dict.fromkeys(['ubg'] + bsg + tcb + tib))
    if load_relations is None:
        load_relations = False

    with open(os.path.join(data_dir, 'train.json'), 'r', encoding='utf-8') as f:
        full_train_buy = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(data_dir, 'test.json'), 'r', encoding='utf-8') as f:
        test_buy = {int(k): v for k, v in json.load(f).items()}
    heldout_test = {(u, item) for u, items in test_buy.items() for item in items}
    if experiment_protocol == 'hgib':
        train_buy = full_train_buy
        valid_buy = {}
        heldout_edges = set()
    else:
        train_buy, valid_buy = _split_train_validation(
            full_train_buy, float(validation_ratio), int(split_seed),
        )
        heldout_valid = {(u, item) for u, items in valid_buy.items() for item in items}
        heldout_edges = heldout_valid | heldout_test

    # Noise edges must not hit any test positive; the HGIB protocol does not split the training graph, so test.json must still be excluded separately.
    noise_excluded_edges = heldout_edges | heldout_test

    edge_dict = {}
    for b in behaviors:
        edges = _load_raw_edges(data_dir, b)
        if edges is None:
            raise FileNotFoundError(f'Required behavior file is missing: {os.path.join(data_dir, f"{b}.txt")}')
        if heldout_edges:
            edges = np.asarray(
                [edge for edge in edges.tolist() if tuple(edge) not in heldout_edges],
                dtype=np.int64,
            ).reshape(-1, 2)
        if noise_ratio > 0 and b in noise_targets:
            edges, added = _inject_random_noise_edges(
                edges, n_users, n_items, noise_ratio,
                int(noise_seed) + sum((i + 1) * ord(c) for i, c in enumerate(b)),
                excluded_edges=noise_excluded_edges,
            )
            noise_stats[b] = added
        raw_edge_dict[b] = edges
        edges = torch.from_numpy(edges.copy()).to(device).T.long()
        edge_dict[b] = _convert_edge(edges, n_users)

    relation_dict = {}
    if load_relations:
        relation_dict = _build_relation_dict(
            data_dir, all_behaviors, n_users, n_items, device, raw_edge_dict,
        )

    known_positive = None
    if experiment_protocol != 'hgib':
        known_positive = {}
        for user in set(train_buy) | set(valid_buy) | set(test_buy):
            known_positive[user] = list(dict.fromkeys(
                train_buy.get(user, []) + valid_buy.get(user, []) + test_buy.get(user, [])
            ))
    train_dataset = BPRDataset(train_buy, n_items, excluded_interactions=known_positive)
    if len(train_dataset) == 0:
        raise ValueError('No valid BPR training samples remain after negative-sample validation')
    pin_memory = str(device).startswith('cuda')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        UserDataset(valid_buy.keys()),
        batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_dataset = (
        InteractionUserDataset(test_buy)
        if experiment_protocol == 'hgib'
        else UserDataset(test_buy.keys())
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    data = {
        'n_users': n_users,
        'n_items': n_items,
        'bsg_types': bsg,
        'behavior_types': list(behaviors),
        'behavior_data': edge_dict,
        'tcb_types': tcb,
        'tib_types': tib,
        'edge_dict': edge_dict,
        'relation_dict': relation_dict,
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'train_gt': train_buy,
        'valid_gt': valid_buy,
        'test_gt': test_buy,
        'experiment_protocol': experiment_protocol,
        'validation_ratio': 0.0 if experiment_protocol == 'hgib' else float(validation_ratio),
        'split_seed': int(split_seed),
        'noise_ratio': noise_ratio,
        'noise_behaviors': sorted(noise_targets),
        'noise_stats': noise_stats,
    }

    return data
