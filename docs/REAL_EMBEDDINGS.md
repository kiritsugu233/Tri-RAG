# Frozen SciFact text embeddings

## Gate and scope

This stage creates normalized original-space embeddings only. It does not
choose a projection dimension, inspect retrieval quality, fit a policy, or
certify any result. Corpus and all query splits are embedded under one frozen
model contract; later selection code may consume only `query_tune` outcomes.

The embedding request is bound to the independently validated SciFact dataset
manifest:

```text
6f54d75d95c40569f7382270e833c8602afd317042e2a791118e4a15992038df
```

Any source-text, qrel, split, ID, dataset-config, or manifest change invalidates
the request before model inference begins.

The checked-in embedding-config fingerprint is
`e6cf0c6eb1ffb8fc053b102eab5d3fbaa32dd9c0e770c22df79ef85df216e6f7`.
Together with the validated dataset artifacts it produces the pre-inference
request fingerprint
`bc94e24580b6d0567232f071f1038f979f66cb1d5769e04828df978857928fe6`.
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
- deterministic PyTorch algorithms enabled, TF32 disabled, eager attention;
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

python3 -m tri_rag_harness.text_embeddings \
  --config configs/real_scifact_e5_base_v2_embeddings.json \
  --dataset data/prepared/scifact \
  --output data/embeddings/scifact-e5-base-v2 \
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
- `query_embeddings.f32.npy`: `1109 x 768`, normalized float32;
- `query_ids.json`: row-aligned stable external-query IDs;
- `embedding_manifest.json`: request, dataset, model snapshot, runtime, input
  truncation, array, ID, and cache identities.

The arrays should occupy approximately 15.18 MiB and 3.25 MiB before `.npy`
headers. The model snapshot is larger and remains in ignored `data/model_cache`.

## Interpretation limit

The public E5 model card already reports benchmark results on SciFact. We chose
this frozen off-the-shelf model for its explicit asymmetric-retrieval contract,
English support, moderate 768 dimension, and reproducible revision, not by
comparing locally observed test outcomes. Even so, the published SciFact result
is prior knowledge and must be listed as a threat to a fully blind model-family
selection claim. It does not authorize policy tuning on `query_cert` or
`query_test`.
