# FiQA query_tune independent audit

Decision: `ACCEPT_QUERY_TUNE_SELECTION_READY_TO_IMPLEMENT_QUERY_CERT`.

The returned allocation-376924 archive passed SHA-256 verification. Independent
reconstruction validated all 16 run artifacts, all 1,967 query records against
the frozen ID order and original FiQA tune qrels, every retention/evidence
curve, the complete `2086 x 1967` candidate-budget matrix, all candidate
evaluations, the six independently selected policies, 11,802 selected-policy
records, the shuffled diagnostic, frozen hypotheses, dense Gaussian matrix,
and the post-tune five-role state. A full local candidate replay reproduced the
returned matrix, outcomes, selection, policies, component registry, and suite
exactly. `query_cert`, `query_latency`, and `query_test` remain closed.

The accepted result is not a positive efficiency result. Tune-selected PDCTP
uses mean `M=1892.763599389934` versus fixed `M=768`, increasing common
coordinate work by `7.31815348183%`, while its retention mean is higher by
`0.009811896289`. It remains quality-eligible under the preregistered tune
constraints, so the selection is frozen and must proceed unchanged to the
independent certification gate. The selected PDCTP emits `M>1984` for 262 tune
queries, including 31 terminal `M=57638` fallbacks; this is incompatible with
the frozen FAISS GPU `M+64<=2048` latency contract and cannot be repaired by
post-tune budget clipping.

The long silent selection interval was redundant computation, not a hung Slurm
step. The runner now shares each deterministic Tri-Predict retention curve
across the five frozen thresholds and reports profile progress. The optimized
implementation reproduced every accepted scientific artifact exactly and does
not change Raw Tri-Predict v1 behavior or identity.
