"""New ANN namespace. No reopening of Step 4 or PDCTP roles.

The ledger reserves each stage before work, closes decisions before labels, and
keeps failed reservations terminal. It is an audit guard, not OS access control.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .certification import empirical_bernstein
from .utils import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/tls_rag_step5_ann_v1.json"
ROLES = ("cal_model_fit", "cal_bound_fit", "tune", "cert", "latency", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path=CONFIG):
    protocol = json.loads(Path(path).read_text())
    if protocol != json.loads(CONFIG.read_text()):
        raise ValueError("only the registered ANN v1 protocol is executable")
    return protocol


def source_binding():
    paths = [CONFIG, *(ROOT / "src/tri_rag_harness").glob("*.py")]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def seal(directory):
    directory = Path(directory)
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "artifact_manifest.json")
    result = {str(p.relative_to(directory)): sha256(p) for p in paths}
    write_json(directory / "artifact_manifest.json", result)
    return fingerprint(result)


def verify_seal(directory):
    root = Path(directory)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p.name != "artifact_manifest.json"}
    if actual != set(manifest):
        raise ValueError("artifact inventory changed")
    for name, expected in manifest.items():
        path = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or any(parent.is_symlink() for parent in path.parents
                       if parent != root and root in parent.parents)):
            raise ValueError("unsafe artifact path")
        if sha256(path) != expected:
            raise ValueError("artifact bytes changed: " + name)
    return fingerprint(manifest)


def save_rows(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def quality_bounds(retention, hit_delta, retention_delta, protocol):
    acceptance = protocol["acceptance"]
    # Half the total alpha is reserved for three independent systems claims.
    alpha = acceptance["alpha_total"] / (acceptance["quality_claims"] + acceptance["systems_claims"])
    def lower(values, paired=False):
        array = np.asarray(values, dtype=np.float64)
        mapped = (array + 1) / 2 if paired else array
        result = empirical_bernstein(mapped, alpha).serialize()
        bound = result["lower_bound_unclipped"] * (2 if paired else 1) - int(paired)
        return {"lower": max(-1. if paired else 0., bound), "bernstein": result}
    bounds = {"retention": lower(retention), "hit_delta": lower(hit_delta, True),
              "retention_delta": lower(retention_delta, True)}
    passed = (bounds["retention"]["lower"] >= acceptance["retention"]
              and bounds["hit_delta"]["lower"] >= -acceptance["hit_drop"]
              and bounds["retention_delta"]["lower"] >= -acceptance["retention_drop"])
    return bounds, bool(passed)


def paired_bootstrap(adaptive, fixed, protocol):
    """Resample query blocks, preserving all method/repeat pairings.

    Returns percentile intervals (empirical uncertainty, not finite-sample
    distribution-free confidence). No deletion, clipping, or outlier trimming.
    """
    a, b = np.asarray(adaptive, dtype=np.float64), np.asarray(fixed, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("positive paired query-by-repeat times required")
    rng = np.random.default_rng(protocol["seeds"]["bootstrap"])
    alpha = protocol["acceptance"]["alpha_total"] / 6
    means, tails = [], []
    for _ in range(protocol["timing"]["bootstrap_replicates"]):
        selected = rng.integers(len(a), size=len(a))
        aa, bb = a[selected], b[selected]
        means.append(float(aa.mean() / bb.mean()))
        tails.append(float(np.percentile(aa, 95) / np.percentile(bb, 95)))
    return {"mean_ratio": float(a.mean() / b.mean()), "p95_ratio": float(np.percentile(a, 95) / np.percentile(b, 95)),
            "mean_ratio_upper": float(np.quantile(means, 1 - alpha, method="higher")),
            "p95_ratio_upper": float(np.quantile(tails, 1 - alpha, method="higher")),
            "alpha_each": alpha, "method": "paired_query_block_percentile_bootstrap"}


class RunLedger:
    def __init__(self, root, binding, protocol):
        self.root = Path(root)
        identity = {"schema": protocol["schema"], "protocol": fingerprint(protocol),
                    "input": binding, "sources": source_binding()}
        self.identity = fingerprint(identity)
        if not self.root.exists():
            self.root.mkdir(parents=True)
            write_json(self.root / "binding.json", identity)
        if json.loads((self.root / "binding.json").read_text()) != identity:
            raise ValueError("run binding changed; never refresh an existing run")

    def result(self, stage):
        if stage not in ROLES:
            raise ValueError("unknown stage")
        directory = self.root / stage
        verify_seal(directory)
        result = json.loads((directory / "result.json").read_text())
        if result["binding"] != self.identity:
            raise ValueError("stage identity changed")
        return result

    def check_ready(self, stage):
        if stage not in ROLES:
            raise ValueError("unknown role")
        index = ROLES.index(stage)
        for ancestor in ROLES[:index]:
            previous = self.result(ancestor)
            if previous["status"] != "passed":
                raise ValueError("previous stage failed or was inconclusive")
        if (self.root / stage).exists():
            raise FileExistsError("role already reserved; preserve the previous attempt")

    def reserve(self, stage):
        self.check_ready(stage)
        directory = self.root / stage
        directory.mkdir()  # exclusive; a crash cannot silently reopen a role
        write_json(directory / "reservation.json", {"binding": self.identity, "stage": stage})
        return directory

    def close_decisions(self, stage):
        if stage not in ROLES:
            raise ValueError("unknown role")
        directory = self.root / stage
        paths = sorted(directory.glob("*.decisions.jsonl"))
        if not paths:
            raise ValueError("decisions must exist before labels")
        receipt = {p.name: sha256(p) for p in paths}
        with (directory / "decisions_closed.json").open("x") as stream:
            json.dump(receipt, stream, sort_keys=True)

    def authorize_labels(self, stage):
        if stage not in ROLES or stage == "latency":
            raise ValueError("latency is label-free")
        directory = self.root / stage
        receipt = json.loads((directory / "decisions_closed.json").read_text())
        for name, expected in receipt.items():
            if sha256(directory / name) != expected:
                raise ValueError("decisions changed after closure")
        fd = os.open(directory / "labels_opened.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump({"stage": stage, "binding": self.identity, "decisions": receipt}, stream)

    def finish(self, stage, status, **values):
        if stage not in ROLES:
            raise ValueError("unknown role")
        if status not in {"passed", "failed", "inconclusive", "no_tune_candidate"}:
            raise ValueError("unknown terminal status")
        directory = self.root / stage
        if status == "passed" and (not (directory / "decisions_closed.json").is_file()
            or (stage != "latency" and not (directory / "labels_opened.json").is_file())):
            raise ValueError("a passed stage requires decision closure and permitted label opening")
        write_json(directory / "result.json", {"binding": self.identity, "status": status, **values})
        return seal(directory)
