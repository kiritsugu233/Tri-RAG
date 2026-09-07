#!/usr/bin/env python3
"""Reproduce review R1/R2 without modifying frozen code or expected unit tests.

Exit 1 means a numerical contract is violated. This is deliberately separate
from the historical regression suite, which does not cover these boundaries.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import json
import math
import warnings

import numpy as np
from scipy.stats import chi2, f

from tri_rag_harness.tri_law import (
    tri_law_conditional_orthogonal, tri_law_probability, tri_law_threshold,
)


def reference_threshold(beta: float, rho: float) -> float:
    with localcontext() as context:
        context.prec = 100
        b, r = Decimal.from_float(beta), Decimal.from_float(rho)
        gap = b - 1
        transverse = 1 - r * r
        root = (gap * gap + 4 * b * transverse).sqrt()
        return float((root + gap) ** 2 / (4 * b * transverse))


def main() -> int:
    rows = []
    cases = [
        ("R1_joint_near_tie_collinearity", float(np.nextafter(1., 2.)), float(np.nextafter(1., 0.))),
        ("R2_large_orthogonal_gap", 1e16, 0.),
        ("R2_discriminant_overflow", 1e200, 0.),
    ]
    for name, beta, rho in cases:
        expected = reference_threshold(beta, rho)
        row = {"case": name, "beta": beta, "rho": rho,
               "reference_threshold": expected,
               "reference_probability_m1": float(f.sf(expected, 1, 1))}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                actual = tri_law_threshold(beta, rho)
                probability = tri_law_probability(beta, rho, 1)
                row.update(actual_threshold=actual, actual_probability_m1=probability,
                           passed=math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.))
            except (ValueError, FloatingPointError) as exc:
                row.update(error=f"{type(exc).__name__}: {exc}", passed=False)
            row["warnings"] = [str(item.message) for item in caught]
        rows.append(row)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = tri_law_conditional_orthogonal(1e308, 1e308, 4)
        expected = float(chi2.cdf(4., 4))
        rows.append({"case": "R2_conditional_intermediate_overflow", "y": 1e308,
                     "beta": 1e308, "m_prime": 4, "actual": actual,
                     "reference": expected,
                     "passed": math.isclose(actual, expected, rel_tol=1e-12),
                     "warnings": [str(item.message) for item in caught]})
    print(json.dumps({"schema_version": 1, "cases": rows,
                      "all_passed": all(row["passed"] for row in rows)},
                     indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
