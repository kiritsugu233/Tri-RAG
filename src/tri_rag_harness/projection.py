from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .utils import fingerprint


def dense_gaussian_projection(m_prime: int, dimension: int, seed: int) -> np.ndarray:
    if isinstance(m_prime, bool) or not isinstance(m_prime, int) or m_prime < 1:
        raise ValueError("m_prime must be a positive integer")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    rng = np.random.default_rng(seed)
    return rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(m_prime),
        size=(m_prime, dimension),
    )


def project_rows(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    projection = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or projection.ndim != 2:
        raise ValueError("vectors and projection matrix must be two-dimensional")
    if values.shape[1] != projection.shape[1]:
        raise ValueError("embedding and projection dimensions do not match")
    projected = values @ projection.T
    if not np.all(np.isfinite(projected)):
        raise ValueError("projection produced nonfinite values")
    return projected


def projection_metadata(
    *,
    dimension: int,
    m_prime: int,
    seed: int,
    normalization: bool,
    embedding_model: str = "unspecified",
    corpus_hash: str = "unspecified",
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "family": "dense_gaussian",
        "dimension": dimension,
        "m_prime": m_prime,
        "seed": seed,
        "entry_mean": 0.0,
        "entry_variance": 1.0 / m_prime,
        "numpy_scale": 1.0 / np.sqrt(m_prime),
        "embedding_model": embedding_model,
        "corpus_hash": corpus_hash,
        "input_l2_normalized": normalization,
        "post_projection_normalized": False,
    }
    metadata["fingerprint"] = fingerprint(metadata)
    return metadata


class CacheMetadataMismatch(ValueError):
    pass


def validate_projection_cache_metadata(
    cached: Dict[str, Any], expected: Dict[str, Any]
) -> None:
    if cached != expected:
        raise CacheMetadataMismatch(
            "projected-embedding cache metadata does not match the frozen run"
        )
