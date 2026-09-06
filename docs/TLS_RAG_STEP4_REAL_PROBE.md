# TLS-RAG Step 4: first real retrieval probe (v2)

Implementation entry: `docs/TLS_RAG_STEP4_REAL_PROBE.md`.
Local checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
Branch: `codex/tls-rag-step4-real-probe`, based on `fc38d928e428dc8c10a3ea16e5ec5db5ccfbf72a`.

The user requested starting real algorithm validation after the successful Step 4
cluster readiness replay. This successor provides a real-data preparation command
and an offline evaluation command. The implementation was validated with generated
CPU fixtures. No real dataset, embedding model weights or real result was opened
or produced during this implementation; only public source/model metadata was
checked. A cluster run is still required to learn whether the algorithm works.

The frozen Step 4 v1 readiness protocol and its real-data gate remain unchanged.
This v2 probe has a separate protocol, namespace, data binding and result schema.
It does not implement or claim completion of v1's independent certification,
latency or final-test milestones, and does not start answer generation.

## Question and interpretation

With a fixed model, projection and budget grid, do Rows 2–4 preserve original-space
nearest neighbors and judged-relevant context hits while computing fewer original
query-to-document distances? Do the Row 3 features improve on Row 2, and do Row 4's
simple query-derived features improve on Row 3?

The references are exact original-space search and projected fixed budgets
32, 128 and 512. The adaptive candidates use the unchanged Step 3 ridge fitting,
reachable-bin CP calculations, validity behavior and dual stopping thresholds.
New candidate IDs and component metadata identify this experiment. The internal
bounds are calibration features; their joint population coverage is unproved.

NFCorpus supplies relevance judgments, not complete facet/contradiction evidence.
The new labels are therefore explicit retrieval proxies:

- current proxy: at least one document with positive qrel grade in the context;
- remaining proxy: current context has no such hit and some later frozen-grid
  context has a hit;
- Row 4 deployable plan: one proxy slot, requiring one support; match the longest
  lowercase ASCII alphanumeric token in the query, breaking ties lexicographically.
  No labels, exact top-k, future context, or corpus statistics choose this token.

The bridge into Step 3 uses its existing internal outcome field names. Those names
must not be interpreted as evidence sufficiency on this dataset. Unjudged documents
contribute zero to judged metrics; this does not establish that they are irrelevant.

## Frozen v2 settings

Protocol: `configs/tls_rag_step4_nfcorpus_probe_v2.json`.
Fingerprint: `9f31fc87ec1c0d38716474c951eefb456d3bda8e3637557390e3e577e5833952`.
The CLI rejects an edited protocol. There are no budget, sample-size or threshold
override flags. A subsequent experiment requires a new version and unused queries.

- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; 384 dimensions, maximum 256 tokens.
  Corpus text is title, newline, then text; query text is unchanged.
- Normalize corpus and query vectors before projection; Gaussian variance
  `1/128`, projection seed 24103, `m_prime=128`; no projected normalization.
- Exact squared L2, string-ID tie breaking; `k_gt=10`, `k_ctx=5`, grid `[32,128,512]`.
- Embedding seed 54101; split seed 54109. All other component parameters bind the
  frozen Step 3 config fingerprint; source code is hashed into the input binding.
- Four mutually exclusive roles from official **training query IDs only**:
  model fit 256, bound fit 512, tune 256, held-out descriptive probe 256.
- Group exact normalized query-text duplicates, keep one deterministic representative
  per group, then hash-sort groups and assign counts before fitting or selection.
  Do not select queries by relevance scores. Insufficient groups cause failure.
  This mechanical grouping does not establish independent semantic question families
  or iid sampling. No formal confidence/certification claim is made.
- On tune, keep adaptive candidates with mean retention at least 0.90, context-hit
  rate no more than 0.02 below the exact-original reference, and mean original
  distances strictly below 512. Minimize mean original distances; tie by ID.
  If none qualifies, write `no_tune_candidate` and leave the probe unopened.
- Freeze the selected candidate before probe evaluation. Report all fixed and
  adaptive diagnostics, with Row 3 minus Row 2 and Row 4 minus Row 3 differences.
  Apply the same empirical targets only to the selected candidate for the probe
  status. Never replace it using probe outcomes. A passing descriptive probe is
  preliminary evidence on this sample, not a certified result or a latency result.

## Exact new data scope and artifacts

The preparation command creates a new run root. In the example below it is
`/home/users/u0001611/Tri-RAG-step4-real-probe/step4-nfcorpus-v2-01`.
It refuses an existing run root and never reads old run caches or archives.
The three source URLs are fixed in `src/tri_rag_harness/tls_rag_step4_nfcorpus.py`:

1. `https://huggingface.co/datasets/BeIR/nfcorpus/resolve/b5026a0e96e8a7ac4f95f482a596389289d46269/corpus/corpus-00000-of-00001.parquet`
   → `source/corpus.parquet` beneath the new run root.
2. `https://huggingface.co/datasets/BeIR/nfcorpus/resolve/b5026a0e96e8a7ac4f95f482a596389289d46269/queries/queries-00000-of-00001.parquet`
   → `source/queries.parquet`. Only selected training queries are embedded.
3. `https://huggingface.co/datasets/BeIR/nfcorpus-qrels/resolve/a451b3b26d3ae1358f259c1a3a4dd61fcea35a65/train.tsv`
   → `source/train.tsv`.

The pinned model is downloaded into the new run root's `model_cache/`.
The trusted preparation process partitions training qrels into these exact files:
`bundle/query_cal_model_fit.labels.json`, `bundle/query_cal_bound_fit.labels.json`,
`bundle/query_tune.labels.json`, and `bundle/query_probe.labels.json`.
It does transport/parse training labels for staging; it does not fit or select a
policy. No official `dev.tsv` or `test.tsv` URL or reader is provided.

Label-free payloads are `bundle/corpus.jsonl`, `bundle/queries.jsonl`,
`bundle/corpus.npy`, `bundle/queries.npy`, and `bundle/roles.json`.
`pre_data_registration.json` records settings and exact destinations before any
download; `role_assignment_audit.json` records duplicate handling;
`bundle/binding.json` binds source revisions/hashes, embeddings, code, environment,
seeds via the protocol, role files, host, device and actual Slurm job environment.
The evaluator rejects changed payloads, source code, model metadata or role counts.

The evaluator is a separate process. It writes and hashes each role's label-free
features/decisions before reading that role's label file. Fit models and bound
tables are serialized before the next role. The probe closure also binds selection.
Late label-file modification is rejected. The trusted staging boundary and these
software guards are not an OS-level access-control boundary.

Results go to `evaluation/`: query-level features, decisions, opened annotations,
fit supervision, models, tables, closure receipts, selection, metrics, summaries,
`result.json`, `report.md`, input binding and a complete artifact hash manifest.
All metrics can be reconstructed from query-level rows. Synthetic tests check
byte-identical outputs on the same host. Different GPU/library versions may yield
slightly different embeddings; compare the recorded vector hashes before treating
runs as identical. Replaying the same bundle is not a fresh scientific sample.

The audit prepares every budget prefix offline. Its wall time includes future
prefixes and must not be called selected-policy serving latency. Work records
separate pilot and expansion original distances, rerank candidates, risk pairs,
quadratic candidate-diversity pairs, plan features and controller evaluations.
A full projected scan is still required. Candidate savings alone do not establish
speedup; especially the frozen Python feature computation may be expensive.

## Manual A100 execution using the existing micromamba environment

Use a new checkout, leaving `/home/users/u0001611/Tri-RAG` and its existing job alone.
From the cluster login shell with GitHub access:

```bash
cd /home/users/u0001611
git clone --single-branch --branch codex/tls-rag-step4-real-probe \
  https://github.com/kiritsugu233/Tri-RAG.git Tri-RAG-step4-real-probe
cd /home/users/u0001611/Tri-RAG-step4-real-probe
git rev-parse HEAD
```

Compare that commit to the commit delivered with this milestone. Enter the A100
allocation with the site's existing `srun` command. Then in the compute-node shell:

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate tri-rag
cd /home/users/u0001611/Tri-RAG-step4-real-probe
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python3 -m pip install -r requirements-tls-rag-probe.txt
```

The requirements add the pinned preparation libraries and use an already compatible
CUDA PyTorch. The preparation command reports missing imports or unavailable CUDA
before downloading data. It needs network access to the fixed Hugging Face sources.
Do not force a CUDA/PyTorch replacement without checking the existing cluster setup.

Run the tests and first actual experiment:

```bash
(
set -eu
cd /home/users/u0001611/Tri-RAG-step4-real-probe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sh scripts/run_tests.sh
TRI_RAG_PROBE_RUN="$PWD/step4-nfcorpus-v2-01"
sh scripts/run_tls_rag_step4_probe.sh prepare "$TRI_RAG_PROBE_RUN"
sh scripts/run_tls_rag_step4_probe.sh evaluate "$TRI_RAG_PROBE_RUN"
cat "$TRI_RAG_PROBE_RUN/evaluation/report.md"
)
```

`prepare` uses A100 for embeddings. `evaluate` uses the exact CPU implementation;
it does not require another GPU allocation after preparation. If evaluation fails,
preserve its directory and traceback. Do not delete results and silently rerun a
changed policy on an opened probe. An intentional unchanged replay can use the
Python evaluator with a different absent `--output` directory and the same bundle.

## Implementation validation

- Focused offline probe suite: 14/14 passed (latest focused run 2.207 s); two full pipeline runs produced identical artifact hashes.
- Full CPU regression and preserved v1 synthetic artifact replay: two rounds, each 243 tests (242 passed, one optional FAISS skip), 29.352 s and 29.138 s; v1 focused 26/26 each and all six v1 artifacts byte-identical.
- Optional preparation dependencies are not installed in the local CPU environment;
  real downloads, Parquet reading and CUDA encoding remain cluster validation work.
- Frozen Step 2/3, exact Tri-Law and Step 4 v1 config/source/tests: unchanged.

Primary source descriptions:
[NFCorpus authors](https://webserver.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/),
[BEIR dataset registry](https://github.com/beir-cellar/beir#available-datasets),
[BEIR data files](https://huggingface.co/datasets/BeIR/nfcorpus/tree/b5026a0e96e8a7ac4f95f482a596389289d46269),
[BEIR training judgments](https://huggingface.co/datasets/BeIR/nfcorpus-qrels/tree/a451b3b26d3ae1358f259c1a3a4dd61fcea35a65),
[fixed MiniLM model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf).
