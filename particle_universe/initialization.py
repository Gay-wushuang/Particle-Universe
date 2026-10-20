"""Deterministic FCC, two-body, and Maxwell-Boltzmann initialization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundaries import PeriodicBoundary
from .validation import integer, nonnegative_scalar, positive_scalar, seed_value


@dataclass(frozen=True, slots=True)
class FCCConfiguration:
    positions: np.ndarray
    box_length: np.float64
    density: np.float64


def fcc_positions(cells: int, density: float) -> FCCConfiguration:
    cell_count = integer(cells, "cells", minimum=1)
    reduced_density = positive_scalar(density, "density")
    particle_count = 4 * cell_count**3
    box_length = np.float64((particle_count / reduced_density) ** (1.0 / 3.0))
    PeriodicBoundary(box_length)

    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    lattice_spacing = box_length / cell_count
    positions = np.empty((particle_count, 3), dtype=np.float64)
    cursor = 0
    for x in range(cell_count):
        for y in range(cell_count):
            for z in range(cell_count):
                origin = np.array([x, y, z], dtype=np.float64)
                positions[cursor : cursor + 4] = (origin + basis) * lattice_spacing
                cursor += 4
    return FCCConfiguration(
        positions=positions,
        box_length=box_length,
        density=reduced_density,
    )


def two_particle_positions(
    separation: float, *, box_length: float | None = None
) -> np.ndarray:
    distance = positive_scalar(separation, "separation")
    if box_length is None:
        return np.array(
            [[-distance / 2.0, 0.0, 0.0], [distance / 2.0, 0.0, 0.0]],
            dtype=np.float64,
        )

    boundary = PeriodicBoundary(box_length)
    if distance >= boundary.box_length / 2.0:
        raise ValueError(
            "two-particle separation must be smaller than half the periodic box"
        )
    center = boundary.box_length / 2.0
    positions = np.array(
        [
            [center - distance / 2.0, center, center],
            [center + distance / 2.0, center, center],
        ],
        dtype=np.float64,
    )
    return boundary.wrap(positions)


def maxwell_boltzmann_velocities(
    particle_count: int, temperature: float, *, seed: int
) -> np.ndarray:
    count = integer(particle_count, "particle_count", minimum=2)
    reduced_temperature = nonnegative_scalar(temperature, "temperature")
    deterministic_seed = seed_value(seed)
    generator = np.random.default_rng(deterministic_seed)
    velocities = generator.normal(
        loc=0.0,
        scale=np.sqrt(reduced_temperature),
        size=(count, 3),
    ).astype(np.float64, copy=False)
    velocities -= velocities.mean(axis=0, keepdims=True, dtype=np.float64)
    if not np.all(np.isfinite(velocities)):
        raise FloatingPointError("initialized velocities became NaN or infinite")
    return velocities
