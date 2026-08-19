from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import scipy

from .config import HarnessConfig
from .utils import array_fingerprint, stable_id_hash


def build_manifest(
    config: HarnessConfig,
    *,
    corpus_ids: list[str],
    query_ids_by_split: Dict[str, list[str]],
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    projection_metadata: Dict[str, Any],
    policy_fingerprint: str,
    policy_fingerprints: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "run_name": config.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config.config_fingerprint,
        "config": config.raw,
        "dataset": {
            "name": "deterministic_clustered_external_queries",
            "version": 1,
            "corpus_size": len(corpus_ids),
            "query_size": sum(len(ids) for ids in query_ids_by_split.values()),
            "corpus_id_hash": stable_id_hash(corpus_ids),
            "corpus_embedding_hash": array_fingerprint(corpus_embeddings),
            "query_embedding_hash": array_fingerprint(query_embeddings),
            "queries_are_external": not bool(
                set(corpus_ids).intersection(
                    query_id for ids in query_ids_by_split.values() for query_id in ids
                )
            ),
        },
        "embedding": {
            "model": "synthetic_array_generator",
            "revision": "v1",
            "l2_normalized": True,
            "dtype": str(corpus_embeddings.dtype),
            "dimension": int(corpus_embeddings.shape[1]),
        },
        "projection": projection_metadata,
        "search": {
            "backend": "numpy_exact_squared_l2",
            "distance": "squared_l2",
            "tie_break": "stable_document_id",
            "post_projection_normalized": False,
        },
        "splits": {
            split: {"n": len(ids), "id_hash": stable_id_hash(ids)}
            for split, ids in query_ids_by_split.items()
        },
        "policy_fingerprint": policy_fingerprint,
        "policy_fingerprints": (
            {"monotone_binned_empirical": policy_fingerprint}
            if policy_fingerprints is None
            else policy_fingerprints
        ),
        "seeds": {
            "data": config.seeds.data,
            "projection": config.seeds.projection,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "package": "tri-rag-harness==0.1.0",
        },
    }
