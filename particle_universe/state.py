"""Particle state container for the reduced-unit, equal-mass system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation import finite_scalar, particle_matrix


@dataclass(slots=True)
class ParticleState:
    positions: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    time: np.float64
    potential_energy: np.float64
    kinetic_energy: np.float64

    def __post_init__(self) -> None:
        self.positions = particle_matrix(
            self.positions, "positions", minimum_particles=2, copy=True
        )
        self.velocities = particle_matrix(
            self.velocities, "velocities", minimum_particles=2, copy=True
        )
        self.forces = particle_matrix(
            self.forces, "forces", minimum_particles=2, copy=True
        )
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have the same shape")
        if self.positions.shape != self.forces.shape:
            raise ValueError("positions and forces must have the same shape")
        self.time = finite_scalar(self.time, "time")
        if self.time < 0.0:
            raise ValueError("time must be non-negative")
        self.potential_energy = finite_scalar(
            self.potential_energy, "potential_energy"
        )
        self.kinetic_energy = finite_scalar(self.kinetic_energy, "kinetic_energy")
        if self.kinetic_energy < 0.0:
            raise ValueError("kinetic_energy must be non-negative")

    @property
    def particle_count(self) -> int:
        return self.positions.shape[0]

    def copy(self) -> "ParticleState":
        return ParticleState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            forces=self.forces.copy(),
            time=self.time,
            potential_energy=self.potential_energy,
            kinetic_energy=self.kinetic_energy,
        )
