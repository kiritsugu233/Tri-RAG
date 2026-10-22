"""Local-file-only MS MARCO preparation; never downloads or reads old runs.

Preparation is a separate producer with access to training qrels. The runtime
consumes role-separated vectors and opens the corresponding label file only
after closing its decisions. Sparse judgments do not identify all evidence.
"""
from __future__ import annotations

import argparse
import heapq
import json
import platform
import re
import unicodedata
from pathlib import Path

import numpy as np

from .tls_rag_step5_protocol import ROLES, load_protocol, seal, sha256, source_binding, verify_seal
from .utils import fingerprint, write_json


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_hash(text):
    return fingerprint(normalize_text(text))


def family_key(text):
    # Also collapse punctuation, token-order and number-only lexical variants.
    words = re.findall(r"\w+", normalize_text(text))
    return fingerprint(sorted("#" if word.isdecimal() else word for word in words))


def rank(seed, value):
    return int(fingerprint([seed, value]), 16)


def read_tsv(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2 or not parts[0] or not parts[1].strip():
                raise ValueError("empty or malformed TSV record")
            yield parts


def sample_corpus(path, size, seed):
    heap = []
    count = 0
    for pid, text in read_tsv(path):
        count += 1
        item = (-rank(seed, pid), pid, text)
        if len(heap) < size:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    if len(heap) != size:
        raise ValueError("insufficient corpus")
    rows = sorted((pid, text) for _, pid, text in heap)
    if len({pid for pid, _ in rows}) != size:
        raise ValueError("duplicate corpus IDs")
    return rows, count


def freeze_roles(queries, positives, corpus_texts, protocol, history=None):
    history = history or {"text_hashes": [], "family_hashes": [], "source_ids": [], "complete": False}
    required = {"text_hashes", "family_hashes", "source_ids", "complete"}
    if set(history) != required or any(not isinstance(history[k], list) for k in required - {"complete"}):
        raise ValueError("history manifest must contain only the four documented fields")
    old_text, old_family, old_ids = (set(history[k]) for k in ("text_hashes", "family_hashes", "source_ids"))
    corpus_hashes = {text_hash(text) for text in corpus_texts}
    seen_ids, by_family = set(), {}
    excluded = {"no_positive_in_subcorpus": 0, "corpus_copy": 0, "history": 0, "lexical_family": 0}
    for qid, text in queries:
        if qid in seen_ids:
            raise ValueError("duplicate query ID")
        seen_ids.add(qid)
        if not positives.get(qid):
            excluded["no_positive_in_subcorpus"] += 1
            continue
        th, family = text_hash(text), family_key(text)
        if th in corpus_hashes:
            excluded["corpus_copy"] += 1
            continue
        if th in old_text or family in old_family or "msmarco-passage-v1:" + qid in old_ids:
            excluded["history"] += 1
            continue
        key = (rank(protocol["seeds"]["split"], qid), qid, text)
        if family in by_family:
            excluded["lexical_family"] += 1
            key = min(key, by_family[family])
        by_family[family] = key
    eligible = sorted(by_family.values())
    if len(eligible) < sum(protocol["roles"].values()):
        raise ValueError("insufficient independent lexical families; do not reduce roles")
    roles, offset = {}, 0
    for role in ROLES:
        count = protocol["roles"][role]
        roles[role] = [(qid, text) for _, qid, text in eligible[offset:offset + count]]
        offset += count
    return roles, {"excluded": excluded, "eligible_lexical_families": len(eligible),
                   "history_manifest_supplied_complete": history["complete"] is True,
                   "history_manifest_fingerprint": fingerprint(history),
                   "semantic_family_independence_established": False}


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("embeddings must be a matrix")
    norms = np.linalg.norm(values, axis=1)
    if not np.all(np.isfinite(values)) or np.any(norms == 0):
        raise ValueError("invalid embeddings")
    return values / norms[:, None]


def write_bundle(output, corpus_ids, corpus_vectors, roles, query_vectors, positives, protocol, metadata):
    if np.shape(corpus_vectors) != (protocol["corpus_size"], protocol["dimension"]):
        raise ValueError("embedding model/corpus dimension mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "labels").mkdir()
    write_json(output / "corpus_ids.json", list(corpus_ids))
    np.save(output / "corpus.npy", normalized(corpus_vectors))
    for role in ROLES:
        ids = [qid for qid, _ in roles[role]]
        if len(ids) != protocol["roles"][role] or len(ids) != len(query_vectors[role]):
            raise ValueError("role count mismatch")
        write_json(output / (role + ".ids.json"), ids)
        np.save(output / (role + ".npy"), normalized(query_vectors[role]))
        if role != "latency":
            write_json(output / "labels" / (role + ".json"), {qid: sorted(positives[qid]) for qid in ids})
    write_json(output / "metadata.json", {"protocol": fingerprint(protocol), "metadata": metadata,
        "corpus_size": len(corpus_ids), "role_counts": protocol["roles"], "sources": source_binding(),
        "roles": {role: {"ids": fingerprint([q for q, _ in rows]),
                         "texts": fingerprint([text_hash(t) for _, t in rows]),
                         "families": fingerprint([family_key(t) for _, t in rows])}
                  for role, rows in roles.items()}})
    seal(output)


def load_bundle(path, protocol):
    path = Path(path)
    binding = verify_seal(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["protocol"] != fingerprint(protocol) or metadata["sources"] != source_binding():
        raise ValueError("bundle protocol/source changed")
    ids = tuple(json.loads((path / "corpus_ids.json").read_text()))
    corpus = np.load(path / "corpus.npy", mmap_mode="r", allow_pickle=False)
    if len(ids) != len(corpus) or len(set(ids)) != len(ids) or corpus.shape != (protocol["corpus_size"], protocol["dimension"]):
        raise ValueError("corpus identity/shape mismatch")
    roles, vectors, all_ids = {}, {}, set()
    for role in ROLES:
        qids = json.loads((path / (role + ".ids.json")).read_text())
        values = np.load(path / (role + ".npy"), mmap_mode="r", allow_pickle=False)
        if (len(qids) != protocol["roles"][role] or len(set(qids)) != len(qids)
                or all_ids.intersection(qids) or values.shape != (len(qids), protocol["dimension"])):
            raise ValueError("role overlap or shape mismatch")
        all_ids.update(qids)
        roles[role], vectors[role] = qids, values
    for values in (corpus, *vectors.values()):
        for start in range(0, len(values), 8192):
            block = values[start:start + 8192]
            if not np.all(np.isfinite(block)) or not np.allclose(np.linalg.norm(block, axis=1), 1., rtol=0., atol=1e-12):
                raise ValueError("bundle must contain normalized finite embeddings")
    return binding, ids, corpus, roles, vectors, metadata


def prepare(args):
    protocol = load_protocol()
    corpus, source_count = sample_corpus(args.collection_tsv, protocol["corpus_size"], protocol["seeds"]["corpus"])
    corpus_ids = {pid for pid, _ in corpus}
    positives = {}
    with args.qrels_train.open() as stream:
        for line in stream:
            qid, unused, pid, relevance = line.split()
            if float(relevance) > 0 and pid in corpus_ids:
                positives.setdefault(qid, set()).add(pid)
    history = json.loads(args.history_manifest.read_text()) if args.history_manifest else None
    roles, audit = freeze_roles(read_tsv(args.queries_tsv), positives, (t for _, t in corpus), protocol, history)
    if source_count != 8841823:
        raise ValueError("MS MARCO v1 source row count changed")
    # Local snapshot only. Bind actual files; configured revision is a declared
    # source identity and must be matched to the upstream snapshot by the operator.
    from sentence_transformers import SentenceTransformer
    import torch
    torch.manual_seed(protocol["seeds"]["embedding"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol["seeds"]["embedding"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.model_revision != protocol["model_revision"]:
        raise ValueError("model revision differs from protocol")
    files = sorted(p for p in args.model_dir.rglob("*") if p.is_file() and ".cache" not in p.parts)
    if not files:
        raise ValueError("empty local model snapshot")
    model_hashes = {str(p.relative_to(args.model_dir)): sha256(p) for p in files}
    model = SentenceTransformer(str(args.model_dir.resolve()), device=args.device)
    if model.get_sentence_embedding_dimension() != protocol["dimension"]:
        raise ValueError("local model has the wrong embedding dimension")
    model.max_seq_length = protocol["max_seq_length"]
    def encode(texts):
        return model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                            normalize_embeddings=False, show_progress_bar=True)
    corpus_vectors = encode([text for _, text in corpus])
    query_vectors = {role: encode([text for _, text in rows]) for role, rows in roles.items()}
    metadata = {"kind": "real", "audit": audit, "model_files": model_hashes,
        "model_revision_declared": args.model_revision, "model_revision_independently_verified": False,
        "source_files": {name: sha256(value) for name, value in
                         (("collection", args.collection_tsv), ("queries", args.queries_tsv), ("qrels_train", args.qrels_train))},
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "population": "training lexical-family representatives with a positive in the hash subcorpus",
        "license": "MS MARCO non-commercial research terms; not an open-source data license"}
    write_bundle(args.output, [pid for pid, _ in corpus], corpus_vectors, roles, query_vectors, positives, protocol, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("collection-tsv", "queries-tsv", "qrels-train", "model-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--history-manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
