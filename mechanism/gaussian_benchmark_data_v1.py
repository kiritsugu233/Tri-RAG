"""Versioned VIBE source/roles/scoring; native normalized metric is preserved."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_source(path, expected_size, expected_sha256):
    path = Path(path)
    if path.stat().st_size != expected_size:
        raise ValueError('source size mismatch')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    if h.hexdigest() != expected_sha256:
        raise ValueError('source hash mismatch')
    return h.hexdigest()


def select_roles(n_learn, n_test, count, seed):
    if not 0 < count < n_learn or n_test < 1:
        raise ValueError('invalid role sizes')
    order = sorted(range(n_learn), key=lambda i: hashlib.sha256(f'{seed}:learn:{i}'.encode()).digest())
    return {'development': order[:count], 'reserved_learn': order[count:],
            'official_test': list(range(n_test))}


def stable_id(role, row):
    if role not in ('corpus', 'learn', 'test') or int(row) != row or row < 0:
        raise ValueError('invalid stable ID')
    return f'vibe-hotpotqa:{role}:{int(row)}'


def native_distances(query, vectors):
    return np.clip(1.0 - np.asarray(vectors) @ np.asarray(query), 0.0, 2.0)


def valid_ids(ids, n, minimum=1):
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(ids) < minimum or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('invalid candidate IDs')
    if np.any(ids < 0) or np.any(ids >= n) or len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate or out-of-range candidate IDs')
    return ids.astype(np.int64, copy=False)


def native_recall(returned_distances, kth_distance, k, epsilon=1e-3):
    distances = np.asarray(returned_distances)
    if k < 1 or distances.shape != (k,) or not np.all(np.isfinite(distances)) or not np.isfinite(kth_distance):
        raise ValueError('invalid recall inputs')
    return float(np.count_nonzero(distances <= kth_distance + epsilon) / k)


def strict_retention(candidates, truth, k):
    truth = np.asarray(truth)[:k]
    if k < 1 or len(truth) != k or len(set(map(int, truth))) != k:
        raise ValueError('invalid truth')
    return len(set(map(int, candidates)) & set(map(int, truth))) / k


class BenchmarkData:
    """Test arrays can only be accessed after a source-bound deployment seal."""
    def __init__(self, path, config):
        source = config['source']
        self.source_sha = verify_source(path, source['bytes'], source['sha256'])
        self.file = h5py.File(path, 'r')
        try:
            for name, shape in [('train', (source['n'], source['dimension'])),
                                ('learn', (source['learn'], source['dimension'])),
                                ('test', (source['test'], source['dimension']))]:
                if self.file[name].shape != shape or self.file[name].dtype != np.dtype('float32'):
                    raise ValueError('full source shape/dtype mismatch: ' + name)
            metric = self.file.attrs['distance']
            if isinstance(metric, bytes):
                metric = metric.decode()
            if metric != 'normalized':
                raise ValueError('native metric mismatch')
            self.k = config['k']
            for name, n in [('learn_neighbors', source['learn']), ('neighbors', source['test']), ('distances', source['test'])]:
                if self.file[name].shape[0] != n or self.file[name].shape[1] < self.k:
                    raise ValueError('ground truth shape mismatch')
            self.roles = select_roles(source['learn'], source['test'], config['development_count'], config['seeds']['roles'])
            self.config_hash = fingerprint(config)
        except BaseException:
            self.file.close()
            raise

    def queries(self, role, seal=None):
        if role == 'development':
            rows = self.roles[role]
            # HDF5 requires increasing fancy indices; preserve declared query order.
            q = np.stack([self.file['learn'][i] for i in rows])
            gt = np.stack([self.file['learn_neighbors'][i, :self.k] for i in rows])
            threshold = None
            prefix = 'learn'
        elif role == 'official_test':
            if not seal or seal.get('source_sha256') != self.source_sha or seal.get('config_sha256') != self.config_hash or not seal.get('points'):
                raise ValueError('official test requires source/config-bound nonempty deployment seal')
            rows = self.roles[role]
            q = self.file['test'][:]
            gt = self.file['neighbors'][:, :self.k]
            threshold = self.file['distances'][:, self.k - 1]
            prefix = 'test'
        else:
            raise ValueError('closed/unknown query role')
        if not np.all(np.isfinite(q)):
            raise ValueError('nonfinite queries')
        for ids in gt:
            valid_ids(ids, len(self.file['train']), self.k)
        return q, gt, threshold, [stable_id(prefix, i) for i in rows]

    def close(self):
        self.file.close()
