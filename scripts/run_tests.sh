#!/bin/sh
set -eu
repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repository_root"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "${TRI_RAG_TEST_PYTHON:-python3}" -m unittest discover -s tests -v
