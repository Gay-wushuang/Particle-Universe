"""Free and periodic boundaries with an explicit minimum-image convention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .constants import CUTOFF
from .validation import finite_result, positive_scalar


class Boundary(Protocol):
    name: str
    box_length: np.float64 | None

    def displacement(self, displacement: np.ndarray) -> np.ndarray: ...

    def wrap(self, positions: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class FreeBoundary:
    """Unbounded space, restricted to analytic one- and two-body checks."""

    name: str = field(default="free", init=False)
    box_length: None = field(default=None, init=False)

    def displacement(self, displacement: np.ndarray) -> np.ndarray:
        result = np.asarray(displacement, dtype=np.float64)
        finite_result(result, "free-boundary displacement")
        return result

    def wrap(self, positions: np.ndarray) -> np.ndarray:
        result = np.array(positions, dtype=np.float64, copy=True, order="C")
        finite_result(result, "free-boundary positions")
        return result


@dataclass(frozen=True, slots=True)
class PeriodicBoundary:
    """Cubic periodic box satisfying the frozen ``L / 2 > rc`` contract."""

    box_length: np.float64
    name: str = field(default="periodic", init=False)

    def __init__(self, box_length: float) -> None:
        length = positive_scalar(box_length, "box_length")
        if length / 2.0 <= CUTOFF:
            raise ValueError(
                f"minimum-image convention requires box_length / 2 > {CUTOFF}; "
                f"got box_length={length}"
            )
        object.__setattr__(self, "box_length", length)
        object.__setattr__(self, "name", "periodic")

    def displacement(self, displacement: np.ndarray) -> np.ndarray:
        value = np.asarray(displacement, dtype=np.float64)
        finite_result(value, "periodic displacement")
        result = value - np.rint(value / self.box_length) * self.box_length
        finite_result(result, "minimum-image displacement")
        return result

    def wrap(self, positions: np.ndarray) -> np.ndarray:
        value = np.asarray(positions, dtype=np.float64)
        finite_result(value, "periodic positions")
        result = np.mod(value, self.box_length).astype(np.float64, copy=False)
        finite_result(result, "wrapped positions")
        return result
