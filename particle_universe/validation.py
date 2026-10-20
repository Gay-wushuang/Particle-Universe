"""Input validation helpers that never mask NaN, infinity, or bad shapes."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import DIMENSIONS


def finite_scalar(value: Any, name: str) -> np.float64:
    try:
        result = np.float64(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def positive_scalar(value: Any, name: str) -> np.float64:
    result = finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def nonnegative_scalar(value: Any, name: str) -> np.float64:
    result = finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def seed_value(value: Any) -> int:
    result = integer(value, "seed", minimum=0)
    if result >= 2**64:
        raise ValueError("seed must be smaller than 2**64")
    return result


def particle_matrix(
    value: Any,
    name: str,
    *,
    minimum_particles: int = 1,
    copy: bool = True,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be convertible to a float64 array") from exc
    if array.ndim != 2 or array.shape[1] != DIMENSIONS:
        raise ValueError(f"{name} must have shape (N, {DIMENSIONS})")
    if array.shape[0] < minimum_particles:
        raise ValueError(f"{name} must contain at least {minimum_particles} particles")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=copy, order="C")


def finite_result(value: np.ndarray | np.floating | float, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} became NaN or infinite")
