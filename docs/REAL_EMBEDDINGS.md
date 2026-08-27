# Frozen SciFact text embeddings

## Gate and scope

This stage creates normalized original-space embeddings only. It does not
choose a projection dimension, inspect retrieval quality, fit a policy, or
certify any result. Corpus and all query splits are embedded under one frozen
model contract; later selection code may consume only `query_tune` outcomes.

The embedding request is bound to the independently validated SciFact dataset
manifest:

```text
4a73586d3a29a0567287e501ac3c06c998af661cdc74dbc589e7525a7924f903
```

Any source-text, qrel, split, ID, dataset-config, or manifest change invalidates
the request before model inference begins.

The checked-in embedding-config fingerprint is
`705153fdd5110981e1bb0f37c7007064b851c50af722a34f41e6c2050e077af7`.
Together with the validated dataset artifacts it produces the pre-inference
request fingerprint
`c0b7992d73434f0f421688f14f3a57ecfe92f3d16e44c90c43fe710f798102ca`.
Both can be checked before accepting a GPU artifact; the eventual cache
fingerprint additionally commits to the actual model snapshot, runtime, token
statistics, and output arrays.

## Frozen model contract

- model: `intfloat/e5-base-v2`;
- Hugging Face commit:
  `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`;
- license declared by the model repository: MIT;
- original embedding dimension: 768;
- maximum sequence length: 512 tokens;
- pooling: the model snapshot's mean-token pooling module;
- corpus input: `passage: ` + stripped title + newline + stripped abstract;
- query input: `query: ` + stripped claim;
- model computation and output: float32;
- post-encoding operation: canonical row-wise L2 normalization, then float32;
- deterministic PyTorch algorithms enabled, TF32 disabled, eager attention,
  cuDNN deterministic mode, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- batch size: 128.

The E5 model card explicitly requires the `query: ` and `passage: ` prefixes
for asymmetric retrieval. It also states the 768 dimension, 512-token limit,
mean pooling, and normalization. The exact trailing spaces in both prefixes are
therefore part of the fingerprinted configuration, not presentation whitespace.

The snapshot downloader requests only the ten files declared in
`configs/real_scifact_e5_base_v2_embeddings.json`, including
`model.safetensors`, tokenizer state, SentenceTransformer modules, pooling
configuration, and the model card. Every downloaded file is SHA-256 hashed into
the embedding manifest; ONNX, OpenVINO, and duplicate PyTorch weights are not
downloaded.

## Runtime contract

The cluster environment already supplies PyTorch `2.5.1`, NumPy `1.26.4`, and
SciPy `1.13.0`. `requirements-embedding-e5.txt` pins the added inference stack.
The adapter refuses execution if any package listed in the embedding config has
an unexpected version. The output manifest also records Python, platform,
CUDA/cuDNN, GPU name/memory, requested/resolved device, and all required package
versions.

The model is English-only. Inputs longer than 512 tokens are truncated by the
frozen model. The manifest separately records corpus/query token-count minimum,
maximum, mean, p95, truncated count, and truncated fraction so this loss is
visible rather than implicit.

## Command

Run on an allocated NVIDIA GPU node after the prepared SciFact directory has
passed the dataset gate:

```bash
cd ~/Tri-RAG

eval "$(micromamba shell hook --shell bash)"
micromamba activate tri-rag-faiss
hash -r

python3 -m pip install -r requirements-embedding-e5.txt
python3 -m pip check

export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 -m tri_rag_harness.text_embeddings \
  --config configs/real_scifact_e5_base_v2_embeddings.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --output data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --device cuda:0 \
  --model-cache data/model_cache
```

On a repeated command, the adapter does not load the model if the existing cache
passes request, manifest, file, array, ID, shape, dtype, and normalization
validation. A mismatch is terminal; it never silently overwrites or repairs a
cache.

## Outputs

- `corpus_embeddings.f32.npy`: `5183 x 768`, normalized float32;
- `corpus_ids.json`: row-aligned stable corpus IDs;
- `query_embeddings.f32.npy`: `1107 x 768`, normalized float32;
- `query_ids.json`: row-aligned stable external-query IDs;
- `embedding_manifest.json`: request, dataset, model snapshot, runtime, input
  truncation, array, ID, and cache identities.

The arrays should occupy approximately 15.18 MiB and 3.24 MiB before `.npy`
headers. The model snapshot is larger and remains in ignored `data/model_cache`.

## First A100 inference and quarantine

Slurm job `373564` on `a100-1` successfully executed the strict runtime and
real E5 inference gates against the earlier v1 split: all 65 then-current tests
passed, the exact pinned model snapshot loaded on an A100-SXM4-80GB, and cache
creation plus no-model-load reuse both completed. The returned archive SHA-256
is `3d87f889cc5a5937ea3666b1f8f7657d02bb14467fb23151daa70dc7fcfa6941`.
Its model-snapshot fingerprint is
`000e13bbbec6825eb1c94ddd1f01e47071b45a1f6c8c749cc97306308ad0c874`;
the 5,183 by 768 corpus and 1,109 by 768 query arrays are float32, normalized,
and internally hash-valid. Corpus truncation was 466/5,183 (8.9909%); query
truncation was 0/1,109.

An independent audit then found that the v1 query array's 1,109 rows contain
only 1,105 unique vectors, exactly explained by four duplicated query texts.
Three duplicate groups crossed tune/cert/test. Consequently cache fingerprint
`4c95bbabd03afb82493843bff9856864f0506b1714e7344a490c8f386369b470`
validates the embedding implementation and frozen model runtime, but is
quarantined from all downstream retrieval and certification claims. The v2
command above must create a new cache bound to the repaired dataset; no old
cache directory is overwritten or silently reused.

## Accepted v2 A100 artifact

The repaired run completed on Slurm job `373564`, node `a100-1`, at commit
`d776404`. All 66 tests passed with real FAISS enabled. Dataset regeneration,
embedding creation, and a second no-model-load cache reuse all succeeded. The
returned audit archive has SHA-256
`dddd51c97d04171f253820131ca37feae450e1ba2b620ed83bf2e9de29e0dd63`.

Independent local audit safely extracted the archive, rehashed both manifests
and every declared artifact, verified array/ID row alignment and finite unit
norms, checked every qrel reference, recomputed normalized-text split
disjointness, and regenerated all five dataset artifacts byte for byte from the
archived source ZIP. The accepted embedding-cache fingerprint is
`2ec53ce38e226129ba0feffcd28ba1da1081e0627ad8e54f4a60e430c341e914`.
The corpus/query array SHA-256 values are respectively
`e6c81429c5b126c37c367bef553615fa9750c8791df263045b3b8d285b9686c7`
and
`5c9ee9d68b65b870ae1ec6ed73aefdea00b9fd1fad4a0cd3e8aadda09a3c7497`.
Corpus truncation remains 466/5,183 (8.9909%); query truncation is 0/1,107.

The two remaining same-text query pairs produce exactly equal vectors and stay
within a single split, leaving 1,105 unique vectors among 1,107 query rows.
The old and new corpus arrays are byte-identical. Reordering queries for the
repaired split changed batch padding, so 85 retained query vectors differ from
their quarantined v1 counterparts by at most `1.081e-7`; this is expected
floating-point behavior and is why downstream runs bind the complete v2 cache
fingerprint rather than mixing rows from different embedding requests.

## Interpretation limit

The public E5 model card already reports benchmark results on SciFact. We chose
this frozen off-the-shelf model for its explicit asymmetric-retrieval contract,
English support, moderate 768 dimension, and reproducible revision, not by
comparing locally observed test outcomes. Even so, the published SciFact result
is prior knowledge and must be listed as a threat to a fully blind model-family
selection claim. It does not authorize policy tuning on `query_cert` or
`query_test`.
