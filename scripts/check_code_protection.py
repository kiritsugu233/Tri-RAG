#!/usr/bin/env python3
"""Read-only inventory and L0 byte-integrity check; human approval is external."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/code_protection.json"


def code_path(path: str) -> bool:
    value = Path(path)
    return (
        value.suffix in {".py", ".sh", ".toml"}
        or path.startswith("configs/")
        or (value.name.startswith("requirements") and value.suffix == ".txt")
    )


def inventory(root: Path) -> set[str]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    paths = {p for p in raw.decode().split("\0") if p and code_path(p)}
    # Detect even ignored additions in implementation directories. Do not inspect
    # experiment archives, datasets, runs, dependency environments or .git.
    for directory in ("src", "tests", "scripts", "configs"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if "__pycache__" not in path.parts and path.is_file() and code_path(relative):
                paths.add(relative)
    return paths


def check(root: Path, registry: dict) -> list[str]:
    if registry.get("schema_version") != 1:
        return ["unsupported protection schema"]
    entries = registry.get("files")
    if not isinstance(entries, list):
        return ["missing per-file protection registry"]
    errors = []
    known = set()
    for entry in entries:
        path = entry.get("path", "")
        relative = Path(path)
        if not path or relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe registry path: {path!r}")
            continue
        if path in known:
            errors.append(f"duplicate registry path: {path}")
        known.add(path)
        if entry.get("level") not in {"L0", "L1", "L2"} or not entry.get("reason"):
            errors.append(f"invalid classification: {path}")
        file = root / path
        if file.is_symlink() or any(parent.is_symlink() for parent in file.parents if parent != root):
            errors.append(f"symlink not allowed in registered path: {path}")
            continue
        if not file.is_file():
            errors.append(f"registered file missing: {path}")
            continue
        if entry.get("level") == "L0":
            expected = entry.get("sha256")
            # Self-hashing would require an impossible recursive digest. The
            # registry is protected by mandatory review against Git, not itself.
            if path == "docs/code_protection.json":
                if entry.get("integrity") != "git_review_no_self_hash":
                    errors.append("registry self-integrity rule changed")
            else:
                actual = hashlib.sha256(file.read_bytes()).hexdigest()
                if actual != expected:
                    errors.append(f"L0 content changed (approval/review required): {path}")
    missing = sorted(inventory(root) - known)
    errors.extend(f"unclassified implementation file: {path}" for path in missing)
    return errors


def main() -> int:
    try:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        errors = check(ROOT, registry)
    except (OSError, ValueError, subprocess.SubprocessError, TypeError, AttributeError) as exc:
        print(f"Protection check failed: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("Preserve changes. Do not reset files or refresh hashes without scoped approval.", file=sys.stderr)
        return 1
    counts = {level: sum(row["level"] == level for row in registry["files"])
              for level in ("L0", "L1", "L2")}
    print(f"Protection check passed: {len(registry['files'])} registered files; {counts}.")
    print("This verifies inventory/bytes, not correctness or human consent. Review registry diffs in Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
