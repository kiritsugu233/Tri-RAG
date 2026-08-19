"""Exact dense-Gaussian single-triplet law, separate from Tri-Predict."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2, f


def _validate_m_prime(m_prime: int) -> None:
    if isinstance(m_prime, bool) or not isinstance(m_prime, (int, np.integer)):
        raise ValueError("m_prime must be an integer")
    if int(m_prime) < 1:
        raise ValueError("m_prime must be positive")


def _scalar_or_array(value: np.ndarray) -> Any:
    if value.ndim == 0:
        return float(value)
    return value


def _broadcast_beta_rho(beta: Any, rho: Any) -> tuple[np.ndarray, np.ndarray]:
    beta_values = np.asarray(beta, dtype=np.float64)
    rho_values = np.asarray(rho, dtype=np.float64)
    try:
        beta_values, rho_values = np.broadcast_arrays(beta_values, rho_values)
    except ValueError as exc:
        raise ValueError("beta and rho must be broadcast-compatible") from exc
    if not np.all(np.isfinite(beta_values)) or not np.all(np.isfinite(rho_values)):
        raise ValueError("beta and rho must be finite")
    if np.any(beta_values <= 1.0):
        raise ValueError("beta must be strictly greater than one")
    tolerance = 8.0 * np.finfo(np.float64).eps
    if np.any(np.abs(rho_values) > 1.0 + tolerance):
        raise ValueError("rho must lie in [-1, 1]")
    return beta_values, np.clip(rho_values, -1.0, 1.0)


def tri_law_threshold(beta: Any, rho: Any) -> Any:
    """Return exact F threshold; scalar inputs return a Python float."""
    beta_values, rho_values = _broadcast_beta_rho(beta, rho)
    absolute_rho = np.abs(rho_values)
    collinear = absolute_rho == 1.0
    discriminant = (1.0 + beta_values) ** 2 - 4.0 * beta_values * rho_values**2
    discriminant = np.maximum(discriminant, 0.0)
    root = np.sqrt(discriminant)
    denominator = root - beta_values + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = (root + beta_values - 1.0) / denominator
        stable = (root + beta_values - 1.0) ** 2 / (
            4.0 * beta_values * (1.0 - rho_values**2)
        )
    near_collinear = (1.0 - rho_values**2) < 1e-8
    result = np.where(near_collinear, stable, direct)
    result = np.where(collinear, np.inf, result)
    if np.any((result <= 0.0) & ~collinear) or np.any(np.isnan(result)):
        raise FloatingPointError("failed to evaluate a positive Tri-Law threshold")
    return _scalar_or_array(np.asarray(result))


def tri_law_probability(beta: Any, rho: Any, m_prime: int) -> Any:
    """Exact dense-Gaussian inversion probability for one ordered triplet."""
    _validate_m_prime(m_prime)
    threshold = np.asarray(tri_law_threshold(beta, rho), dtype=np.float64)
    probability = np.asarray(f.sf(threshold, int(m_prime), int(m_prime)))
    probability = np.clip(probability, 0.0, 1.0)
    return _scalar_or_array(probability)


def tri_law_conditional_orthogonal(y: Any, beta: Any, m_prime: int) -> Any:
    """Exact conditional inversion probability under orthogonal directions."""
    _validate_m_prime(m_prime)
    y_values = np.asarray(y, dtype=np.float64)
    beta_values = np.asarray(beta, dtype=np.float64)
    try:
        y_values, beta_values = np.broadcast_arrays(y_values, beta_values)
    except ValueError as exc:
        raise ValueError("y and beta must be broadcast-compatible") from exc
    if not np.all(np.isfinite(y_values)) or not np.all(np.isfinite(beta_values)):
        raise ValueError("y and beta must be finite")
    if np.any(y_values < 0.0):
        raise ValueError("y must be nonnegative")
    if np.any(beta_values <= 1.0):
        raise ValueError("beta must be strictly greater than one")
    probability = np.asarray(
        chi2.cdf(int(m_prime) * y_values / beta_values, df=int(m_prime))
    )
    return _scalar_or_array(np.clip(probability, 0.0, 1.0))
