from __future__ import annotations

from typing import Any, Dict


def _number(value: float) -> str:
    return f"{value:.4f}"


def generate_report(
    manifest: Dict[str, Any],
    policy: Dict[str, Any],
    certification: Dict[str, Any],
    aggregates: Dict[str, Any],
    timings: Dict[str, Any],
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
        "## Frozen adaptive policy",
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
        "",
        "Timing fields are runtime observations and are not expected to be byte-identical across repeated runs. Policy and metric values are deterministic under the saved seeds.",
        "",
    ]
    return "\n".join(lines)
