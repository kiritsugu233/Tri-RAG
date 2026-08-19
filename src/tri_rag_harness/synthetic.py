from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import SyntheticConfig
from .embeddings import EmbeddingTable, make_table


@dataclass(frozen=True)
class SyntheticDataset:
    corpus: EmbeddingTable
    queries: EmbeddingTable
    splits: np.ndarray
    relevant_clusters: np.ndarray

    def split_indices(self) -> Dict[str, np.ndarray]:
        return {
            split: np.flatnonzero(self.splits == split)
            for split in ("query_tune", "query_cert", "query_test")
        }


def generate_synthetic_dataset(config: SyntheticConfig, seed: int) -> SyntheticDataset:
    """Generate independently sampled external queries around corpus clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(config.n_clusters, config.dimension))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    corpus_vectors = []
    corpus_ids = []
    for cluster in range(config.n_clusters):
        noise = rng.normal(
            scale=config.cluster_noise,
            size=(config.docs_per_cluster, config.dimension),
        )
        corpus_vectors.append(centers[cluster] + noise)
        corpus_ids.extend(
            f"doc-{cluster:02d}-{row:04d}" for row in range(config.docs_per_cluster)
        )

    split_counts = (
        ("query_tune", config.query_tune),
        ("query_cert", config.query_cert),
        ("query_test", config.query_test),
    )
    query_vectors = []
    query_ids = []
    split_names = []
    relevant_clusters = []
    global_row = 0
    for split, count in split_counts:
        for local_row in range(count):
            cluster = (global_row * 5) % config.n_clusters
            difficulty = (local_row + 0.5) / count
            noise_scale = config.query_noise_min + difficulty * (
                config.query_noise_max - config.query_noise_min
            )
            query_vectors.append(
                centers[cluster] + rng.normal(scale=noise_scale, size=config.dimension)
            )
            query_ids.append(f"{split}-{local_row:05d}")
            split_names.append(split)
            relevant_clusters.append(cluster)
            global_row += 1

    if set(corpus_ids) & set(query_ids):
        raise AssertionError("synthetic query IDs must be external to corpus IDs")
    return SyntheticDataset(
        corpus=make_table(corpus_ids, np.vstack(corpus_vectors)),
        queries=make_table(query_ids, np.vstack(query_vectors)),
        splits=np.asarray(split_names, dtype=str),
        relevant_clusters=np.asarray(relevant_clusters, dtype=np.int64),
    )
