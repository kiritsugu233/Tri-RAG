#!/bin/sh
# Existing CPU NumPy/SciPy environment only. No install, push or submission.
set -eu
if [ "$#" -ne 1 ]; then
    echo 'usage: sh scripts/run_tls_rag_step4_readiness.sh NEW_OUTPUT_PARENT' >&2
    exit 2
fi
output_parent=$1
# Refuse any existing parent, including an empty directory or dangling symlink.
if [ -e "$output_parent" ] || [ -L "$output_parent" ]; then
    echo 'output parent must be absent; existing contents will not be inspected' >&2
    exit 2
fi
mkdir "$output_parent"
output_parent=$(cd "$output_parent" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
for replay in a b; do
    python3 -m unittest discover -s tests -p 'test_tls_rag_step4.py' -v
    sh scripts/run_tests.sh
    python3 -m tri_rag_harness.tls_rag_step4 \
        --expected-protocol-fingerprint 35a3ae863249d1f1d7aec7f0b2fbaae877871093216523177b67e5c40b0a802e \
        --output "$output_parent/$replay"
done
diff -r "$output_parent/a" "$output_parent/b"
echo 'Step 4 synthetic artifacts are byte-identical; real-data gate remains closed.'
