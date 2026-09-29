"""Shared numeric checks for model inputs."""

from __future__ import annotations

import math


def require_finite(name: str, value: object) -> float:
    """Return ``value`` as a finite float.

    Booleans are rejected so ``True`` is not accepted as ``1``.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def require_probability(name: str, value: object) -> float:
    """Return a finite probability on the closed unit interval."""
    number = require_finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def require_credit_score(value: object) -> int:
    """Return a whole-number FICO-style score in ``[300, 850]``."""
    if isinstance(value, bool):
        raise TypeError("credit score must be an integer")
    number = require_finite("credit score", value)
    if not number.is_integer():
        raise TypeError("credit score must be an integer")
    score = int(number)
    if not 300 <= score <= 850:
        raise ValueError("credit score must be between 300 and 850")
    return score
