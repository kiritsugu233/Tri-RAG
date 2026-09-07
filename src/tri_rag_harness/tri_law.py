"""Exact dense-Gaussian single-triplet law, separate from Tri-Predict."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2, f


TRI_LAW_NUMERICAL_VERSION = 2


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


def _sqrt_threshold(beta: Any, rho: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sqrt(r) without subtracting nearly equal numbers or squaring beta.

    With t=(beta-1)/(2*sqrt(beta)*sqrt(1-rho**2)),
    sqrt(r)=hypot(1,t)+t. Factoring 1-rho**2 retains the near-collinear gap.
    Divide by sqrt(beta) first to keep every non-collinear finite-input t finite.
    """
    beta_values, rho_values = _broadcast_beta_rho(beta, rho)
    absolute_rho = np.abs(rho_values)
    transverse = (1.0 - absolute_rho) * (1.0 + absolute_rho)
    with np.errstate(divide="ignore"):
        t = ((beta_values - 1.0) / np.sqrt(beta_values)) / 2.0 / np.sqrt(transverse)
    root = np.hypot(1.0, t) + t
    root = np.where(absolute_rho == 0.0, np.sqrt(beta_values), root)
    return np.asarray(root), beta_values, absolute_rho


def tri_law_threshold(beta: Any, rho: Any) -> Any:
    """Return r, or +inf for collinearity/float64 overflow of the true threshold."""
    root, beta_values, absolute_rho = _sqrt_threshold(beta, rho)
    with np.errstate(over="ignore"):
        result = root * root
    # Preserve the exact orthogonal identity, including beta=float64.max.
    result = np.where(absolute_rho == 0.0, beta_values, result)
    return _scalar_or_array(np.asarray(result))


def tri_law_probability(beta: Any, rho: Any, m_prime: int) -> Any:
    """Exact dense-Gaussian inversion probability for one ordered triplet."""
    _validate_m_prime(m_prime)
    root, beta_values, absolute_rho = _sqrt_threshold(beta, rho)
    if m_prime == 1:
        # F(1,1) has a Cauchy-square tail. Keep representable probabilities even
        # when r itself exceeds float64.max; sqrt(r) is still representable.
        probability = (2.0 / np.pi) * np.arctan(1.0 / root)
    elif m_prime == 2:
        with np.errstate(under="ignore"):
            reciprocal = (1.0 / root) ** 2
        probability = reciprocal / (1.0 + reciprocal)
    else:
        with np.errstate(over="ignore"):
            threshold = root * root
        threshold = np.where(absolute_rho == 0.0, beta_values, threshold)
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
    with np.errstate(over="ignore", under="ignore"):
        scaled = int(m_prime) * y_values
        argument = scaled / beta_values
        # If m*y overflowed, divide first. In that branch y is large enough
        # that y/beta cannot underflow for a finite representable m.
        argument = np.where(np.isinf(scaled), (y_values / beta_values) * int(m_prime), argument)
    probability = np.asarray(chi2.cdf(argument, df=int(m_prime)))
    return _scalar_or_array(np.clip(probability, 0.0, 1.0))
