from __future__ import annotations

from typing import Any, Dict, Optional


def _number(value: float) -> str:
    return f"{value:.4f}"


def generate_report(
    manifest: Dict[str, Any],
    policy: Dict[str, Any],
    certification: Dict[str, Any],
    aggregates: Dict[str, Any],
    timings: Dict[str, Any],
    *,
    tri_predict_policy: Optional[Dict[str, Any]] = None,
    tri_predict_certification: Optional[Dict[str, Any]] = None,
    tri_predict_timings: Optional[Dict[str, Any]] = None,
) -> str:
    config = manifest["config"]
    retrieval = config["retrieval"]
    certificate_status = "PASS" if certification["passed"] else "FAIL"
    sample_warning = (
        "none"
        if certification["sample_size_sufficient"]
        else (
            f"certification has n={certification['n']}, below planned "
            f"n={certification['planned_n']}"
        )
    )
    fixed_baseline = aggregates["certification_comparison"]["smallest_certified_fixed_budget"]
    saving = aggregates["certification_comparison"]["adaptive_candidate_saving"]
    fixed_text = "none" if fixed_baseline is None else str(fixed_baseline)
    saving_text = "not available" if saving is None else _number(saving)
    lines = [
        "# Synthetic Query-Adaptive Tri-RAG Run",
        "",
        "## Frozen setup",
        "",
        f"- Corpus/query sizes: {manifest['dataset']['corpus_size']} / {manifest['dataset']['query_size']}",
        f"- External/disjoint query IDs: {manifest['dataset']['queries_are_external']}",
        f"- Original/projected dimensions: {manifest['embedding']['dimension']} / {retrieval['m_prime']}",
        f"- Projection seed: {manifest['seeds']['projection']}",
        f"- k_gt / k_ctx / M_pilot: {retrieval['k_gt']} / {retrieval['k_ctx']} / {retrieval['m_pilot']}",
        f"- M grid: {retrieval['m_grid']}",
        f"- Post-projection normalization: {manifest['search']['post_projection_normalized']}",
        "",
        "## Frozen empirical adaptive policy",
        "",
        f"- Type: {policy['name']}",
        f"- LID edges: {policy['edges']}",
        f"- Budgets by increasing-LID bin: {policy['budgets']}",
        f"- Fingerprint: `{policy['fingerprint']}`",
        "",
        "## Independent certification",
        "",
        f"- Stored artifact decision: **{certificate_status}**",
        f"- n / mean / lower bound: {certification['n']} / {_number(certification['mean'])} / {_number(certification['lower_bound'])}",
        f"- Target / alpha: {_number(certification['target'])} / {_number(certification['alpha'])}",
        f"- Radius: {_number(certification['radius_total'])}",
        f"- Sample-size warning: {sample_warning}",
        f"- Smallest fixed budget passing the same certificate: {fixed_text}",
        f"- Adaptive candidate saving against that baseline: {saving_text}",
        "",
        "A failed target is retained as a valid result. This certificate covers embedding-neighbor retention only; it does not certify evidence or answer correctness.",
        "",
        "## Test-split summary",
        "",
        f"- Mean embedding retention: {_number(aggregates['query_test']['adaptive_retention']['mean'])}",
        f"- Mean / P95 budget: {_number(aggregates['query_test']['budget']['mean'])} / {_number(aggregates['query_test']['budget']['p95'])}",
        f"- Mean absolute pilot-vs-oracle LID gap: {_number(aggregates['query_test']['lid_diagnostic']['mean_absolute_gap'])}",
        "",
        "## Retrieval timing and work",
        "",
        f"- Mean pilot search ms: {_number(timings['mean_pilot_search_ms'])}",
        f"- Mean expansion search ms: {_number(timings['mean_expansion_search_ms'])}",
        f"- Mean original rerank ms: {_number(timings['mean_rerank_ms'])}",
        f"- Mean pilot original distance count: {_number(timings['mean_pilot_original_distance_count'])}",
        f"- Mean additional original distance count: {_number(timings['mean_additional_original_distance_count'])}",
        f"- Mean projected scan count: {_number(timings['mean_projected_scan_count'])}",
        f"- Mean projected distance count: {_number(timings['mean_projected_distance_count'])}",
        "- Pilot/expansion reuse: one M_max projected scan; expansion slices the cached ranking",
        "",
        "Timing fields are runtime observations and are not expected to be byte-identical across repeated runs. Policy and metric values are deterministic under the saved seeds.",
        "",
    ]
    if (
        tri_predict_policy is not None
        and tri_predict_certification is not None
        and tri_predict_timings is not None
    ):
        tri_status = "PASS" if tri_predict_certification["passed"] else "FAIL"
        tri_aggregates = aggregates["tri_predict"]
        tri_saving = aggregates["tri_predict_certification_comparison"][
            "adaptive_candidate_saving"
        ]
        tri_saving_text = "not available" if tri_saving is None else _number(tri_saving)
        lines.extend(
            [
                "## Query-adaptive Tri-Predict diagnostic",
                "",
                f"- Target predicted retention: {_number(tri_predict_policy['target'])}",
                f"- Safety correction: {_number(tri_predict_policy['safety_correction'])}",
                f"- Rank aggregation: {tri_predict_policy['rank_aggregation']}",
                f"- Fingerprint: `{tri_predict_policy['fingerprint']}`",
                f"- Stored certificate decision: **{tri_status}**",
                f"- Certification n / mean / lower bound: {tri_predict_certification['n']} / {_number(tri_predict_certification['mean'])} / {_number(tri_predict_certification['lower_bound'])}",
                f"- Certification mean M: {_number(tri_aggregates['query_cert']['budget']['mean'])}",
                f"- Certification saturated queries: {tri_aggregates['query_cert']['policy_status']['saturated_n']}",
                f"- Candidate saving against the smallest certified fixed budget: {tri_saving_text}",
                f"- Test mean retention / mean M: {_number(tri_aggregates['query_test']['adaptive_retention']['mean'])} / {_number(tri_aggregates['query_test']['budget']['mean'])}",
                f"- Mean policy computation ms: {_number(tri_predict_timings['mean_policy_compute_ms'])}",
                "",
                "Tri-Predict is not the exact Tri-Law. It additionally uses the LID rank-distance model, the orthogonal specialization, a structural surrogate, conditional independence, and mean-field thresholding.",
                "The two policy certificates are policy-specific; certification results must not be used to choose between policies.",
                "",
            ]
        )
    return "\n".join(lines)
