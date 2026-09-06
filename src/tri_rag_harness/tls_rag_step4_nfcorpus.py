"""Prepare a fresh, pinned NFCorpus training-only bundle for the v2 probe.

This is a trusted data-staging process, separate from the offline policy process.
It partitions qrels without selecting queries by their relevance scores. No
metadata, embeddings or labels from existing project runs are reused.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from .tls_rag_step4_probe import (ROOT, ROLES, CODE_FILES, assign_roles, fingerprint,
                                load_protocol, save, save_rows, sha256)

DATA_REVISION = "b5026a0e96e8a7ac4f95f482a596389289d46269"
QRELS_REVISION = "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65"
DATA_BASE = f"https://huggingface.co/datasets/BeIR/nfcorpus/resolve/{DATA_REVISION}"
QRELS_BASE = f"https://huggingface.co/datasets/BeIR/nfcorpus-qrels/resolve/{QRELS_REVISION}"
SOURCES = {
    "corpus.parquet": f"{DATA_BASE}/corpus/corpus-00000-of-00001.parquet",
    "queries.parquet": f"{DATA_BASE}/queries/queries-00000-of-00001.parquet",
    "train.tsv": f"{QRELS_BASE}/train.tsv",
}


def validate_text_rows(rows, corpus=False):
    result = {}
    for row in rows:
        query_id = row.get("_id")
        if not isinstance(query_id, str) or not query_id or query_id in result:
            raise ValueError("data IDs must be nonempty unique strings")
        text = row.get("text")
        title = row.get("title", "") if corpus else ""
        if not isinstance(text, str) or not isinstance(title, str):
            raise ValueError("invalid source text")
        result[query_id] = (title + "\n" + text).strip() if corpus else text.strip()
    return dict(sorted(result.items()))


def read_train_qrels(path):
    """Staging only; query eligibility depends on IDs, never score values."""
    ids, rows, pairs = set(), [], set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("unexpected BEIR qrels header")
        for row in reader:
            query_id, passage_id = row["query-id"], row["corpus-id"]
            if not query_id or not passage_id or (query_id, passage_id) in pairs:
                raise ValueError("empty or duplicate qrel identity")
            try:
                score = int(row["score"])
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid relevance grade") from exc
            if score < 0:
                raise ValueError("negative relevance grade")
            pairs.add((query_id, passage_id))
            ids.add(query_id)
            rows.append((query_id, passage_id, score))
    return ids, rows


def partition_qrels(rows, roles, corpus_ids):
    flattened = [query_id for role in ROLES for query_id in roles[role]]
    if len(set(flattened)) != len(flattened):
        raise ValueError("overlapping roles")
    destination = {query_id: role for role in ROLES for query_id in roles[role]}
    labels = {role: {query_id: [] for query_id in roles[role]} for role in ROLES}
    seen = set()
    for query_id, passage_id, score in rows:
        if passage_id not in corpus_ids:
            raise ValueError("qrel refers to unknown corpus ID")
        if query_id in destination:
            seen.add(query_id)
            if score > 0:
                labels[destination[query_id]][query_id].append(passage_id)
    if seen != set(flattened):
        raise ValueError("selected query has no annotation rows")
    for role_labels in labels.values():
        for query_id, values in role_labels.items():
            role_labels[query_id] = sorted(values)
    return labels


def download(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "TLS-RAG-probe-v2"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("xb") as stream:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            stream.write(chunk)
    return sha256(path)


def prepare(output: Path, device: str):
    protocol = load_protocol()
    # Check optional dependencies and CUDA before any downloads or directory writes.
    try:
        import torch
        import pyarrow.parquet as parquet
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("activate micromamba tri-rag; install requirements-tls-rag-probe.txt first") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; enter the allocated GPU node before preparation")
    if output.exists():
        raise FileExistsError("use a new output directory; existing bundles are never overwritten")
    output.mkdir(parents=True)
    raw = output / "source"
    bundle = output / "bundle"
    raw.mkdir()
    bundle.mkdir()
    source_code = {name: sha256(ROOT / name) for name in CODE_FILES}
    save(output / "pre_data_registration.json", {"protocol": protocol,
        "protocol_fingerprint": fingerprint(protocol), "source_urls": SOURCES,
        "source_code_sha256": source_code,
        "data_staging_paths": [str(raw / name) for name in SOURCES],
        "model_cache_path": str(output / "model_cache"), "bundle_path": str(bundle),
        "official_dev_test_labels": "no download URLs and no reader", "device": device})
    hashes = {}
    for name, url in SOURCES.items():
        print(f"Downloading pinned {name}", flush=True)
        hashes[name] = download(url, raw / name)
    corpus = validate_text_rows(parquet.read_table(raw / "corpus.parquet").to_pylist(), corpus=True)
    queries = validate_text_rows(parquet.read_table(raw / "queries.parquet").to_pylist())
    eligible_ids, qrels = read_train_qrels(raw / "train.tsv")
    roles, role_audit = assign_roles(queries, eligible_ids, protocol["role_counts"], protocol["split_seed"])
    labels = partition_qrels(qrels, roles, set(corpus))
    selected_ids = sorted(query_id for role in ROLES for query_id in roles[role])
    selected = {query_id: queries[query_id] for query_id in selected_ids}
    save(output / "role_assignment_audit.json", role_audit)
    save(bundle / "roles.json", roles)
    save_rows(bundle / "corpus.jsonl", ({"passage_id": key, "text": text} for key, text in corpus.items()))
    save_rows(bundle / "queries.jsonl", ({"query_id": key, "text": text} for key, text in selected.items()))
    for role in ROLES:
        save(bundle / f"{role}.labels.json", labels[role])
    # Staging owns all training qrels. The separate run command loads only one
    # role's labels after closing that role's decision artifacts.
    del labels, qrels
    torch.manual_seed(protocol["embedding_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol["embedding_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SentenceTransformer(protocol["embedding"]["model_id"],
        revision=protocol["embedding"]["revision"], device=device,
        cache_folder=str(output / "model_cache"), trust_remote_code=False)
    model.max_seq_length = protocol["embedding"]["max_seq_length"]
    model.eval()
    for name, texts in (("corpus", list(corpus.values())), ("queries", list(selected.values()))):
        vectors = model.encode(texts, batch_size=32, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.shape != (len(texts), protocol["embedding"]["dimension"]) or not np.all(np.isfinite(vectors)):
            raise ValueError("embedding shape or finite check failed")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("zero text embedding")
        vectors /= norms
        np.save(bundle / f"{name}.npy", vectors, allow_pickle=False)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "torch", "sentence-transformers", "transformers", "pyarrow")}
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        git_head = "unavailable"
    save(bundle / "binding.json", {"schema": "tls_rag_nfcorpus_probe_binding_v2",
        "protocol_fingerprint": fingerprint(protocol), "embedding": protocol["embedding"],
        "source_revisions": {"dataset": DATA_REVISION, "qrels": QRELS_REVISION},
        "source_urls": SOURCES, "source_sha256": hashes, "source_code_sha256": source_code,
        "git_head": git_head, "python": platform.python_version(), "versions": versions,
        "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": device, "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "corpus_count": len(corpus), "selected_query_count": len(selected),
        "role_audit_sha256": sha256(output / "role_assignment_audit.json"),
        "files": {path.name: sha256(path) for path in sorted(bundle.iterdir())}})
    print(f"Prepared bundle: {bundle}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prepare(args.output.resolve(), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
