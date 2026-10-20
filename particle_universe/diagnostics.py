"""Thermodynamic diagnostics for the frozen reduced-unit convention."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .constants import MASS
from .state import ParticleState
from .validation import finite_result, particle_matrix


def kinetic_energy(velocities: np.ndarray) -> np.float64:
    value = particle_matrix(
        velocities, "velocities", minimum_particles=2, copy=False
    )
    result = np.float64(0.5 * MASS * np.sum(value * value, dtype=np.float64))
    finite_result(result, "kinetic energy")
    return result


def total_momentum(velocities: np.ndarray) -> np.ndarray:
    value = particle_matrix(
        velocities, "velocities", minimum_particles=2, copy=False
    )
    result = MASS * np.sum(value, axis=0, dtype=np.float64)
    finite_result(result, "total momentum")
    return np.asarray(result, dtype=np.float64)


def temperature(kinetic: float, particle_count: int) -> np.float64:
    if particle_count <= 1:
        raise ValueError("temperature requires at least two particles")
    degrees_of_freedom = 3 * particle_count - 3
    result = np.float64(2.0 * kinetic / degrees_of_freedom)
    finite_result(result, "temperature")
    return result


@dataclass(frozen=True, slots=True)
class ThermodynamicSample:
    step: int
    time: float
    potential_energy: float
    kinetic_energy: float
    total_energy: float
    temperature: float
    total_momentum: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sample(state: ParticleState, step: int) -> ThermodynamicSample:
    momentum = total_momentum(state.velocities)
    total = np.float64(state.potential_energy + state.kinetic_energy)
    finite_result(total, "total energy")
    return ThermodynamicSample(
        step=step,
        time=float(state.time),
        potential_energy=float(state.potential_energy),
        kinetic_energy=float(state.kinetic_energy),
        total_energy=float(total),
        temperature=float(temperature(state.kinetic_energy, state.particle_count)),
        total_momentum=tuple(float(value) for value in momentum),
    )
