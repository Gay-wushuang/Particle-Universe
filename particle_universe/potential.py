"""Energy-shifted Lennard-Jones variant A and its analytic pair force."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundaries import Boundary
from .constants import CUTOFF, EPSILON, POTENTIAL_VARIANT, SIGMA
from .validation import finite_result, particle_matrix, positive_scalar


class ParticleOverlapError(ValueError):
    """Raised when two particles occupy the same physical position."""


@dataclass(frozen=True, slots=True)
class ForceResult:
    forces: np.ndarray
    potential_energy: np.float64
    interacting_pairs: int


class ShiftedLennardJones:
    """Frozen PU-001 potential: LJ 12-6, energy shifted at ``rc=2.5``."""

    cutoff = CUTOFF
    sigma = SIGMA
    epsilon = EPSILON
    variant = POTENTIAL_VARIANT

    def __init__(self) -> None:
        inverse_cutoff = SIGMA / CUTOFF
        inverse_cutoff_6 = inverse_cutoff**6
        self._cutoff_energy = np.float64(
            4.0 * EPSILON * (inverse_cutoff_6**2 - inverse_cutoff_6)
        )
        self._cutoff_squared = np.float64(CUTOFF * CUTOFF)

    @property
    def cutoff_energy(self) -> np.float64:
        return self._cutoff_energy

    def pair_energy(self, distance: float) -> np.float64:
        radius = positive_scalar(distance, "distance")
        if radius >= CUTOFF:
            return np.float64(0.0)
        inverse_radius = SIGMA / radius
        try:
            with np.errstate(divide="raise", over="raise", invalid="raise"):
                inverse_radius_6 = inverse_radius**6
                energy = 4.0 * EPSILON * (
                    inverse_radius_6**2 - inverse_radius_6
                ) - self._cutoff_energy
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Lennard-Jones energy is numerically singular at this separation"
            ) from exc
        finite_result(energy, "pair potential energy")
        return np.float64(energy)

    def pair_force(self, displacement: np.ndarray) -> np.ndarray:
        vector = np.asarray(displacement, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError("pair displacement must have shape (3,)")
        finite_result(vector, "pair displacement")
        radius_squared = np.float64(np.dot(vector, vector))
        if radius_squared == 0.0:
            raise ParticleOverlapError("overlapping particles are not allowed")
        if radius_squared >= self._cutoff_squared:
            return np.zeros(3, dtype=np.float64)
        return self._force_inside_cutoff(vector, radius_squared)

    def evaluate(self, positions: np.ndarray, boundary: Boundary) -> ForceResult:
        coordinates = particle_matrix(
            positions, "positions", minimum_particles=2, copy=False
        )
        forces = np.zeros_like(coordinates, dtype=np.float64)
        potential_energy = np.float64(0.0)
        interacting_pairs = 0

        for i in range(coordinates.shape[0] - 1):
            for j in range(i + 1, coordinates.shape[0]):
                displacement = boundary.displacement(coordinates[i] - coordinates[j])
                radius_squared = np.float64(np.dot(displacement, displacement))
                if radius_squared == 0.0:
                    raise ParticleOverlapError(
                        f"particles {i} and {j} overlap under the selected boundary"
                    )
                if radius_squared >= self._cutoff_squared:
                    continue
                pair_force = self._force_inside_cutoff(displacement, radius_squared)
                pair_energy = self._energy_inside_cutoff(radius_squared)
                forces[i] += pair_force
                forces[j] -= pair_force
                potential_energy += pair_energy
                interacting_pairs += 1

        finite_result(forces, "forces")
        finite_result(potential_energy, "potential energy")
        return ForceResult(
            forces=forces,
            potential_energy=np.float64(potential_energy),
            interacting_pairs=interacting_pairs,
        )

    def _energy_inside_cutoff(self, radius_squared: np.float64) -> np.float64:
        try:
            with np.errstate(divide="raise", over="raise", invalid="raise"):
                inverse_radius_2 = np.float64(1.0) / radius_squared
                inverse_radius_6 = inverse_radius_2**3
                energy = 4.0 * EPSILON * (
                    inverse_radius_6**2 - inverse_radius_6
                ) - self._cutoff_energy
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Lennard-Jones energy is numerically singular at this separation"
            ) from exc
        finite_result(energy, "pair potential energy")
        return np.float64(energy)

    def _force_inside_cutoff(
        self, displacement: np.ndarray, radius_squared: np.float64
    ) -> np.ndarray:
        try:
            with np.errstate(divide="raise", over="raise", invalid="raise"):
                inverse_radius_2 = np.float64(1.0) / radius_squared
                inverse_radius_6 = inverse_radius_2**3
                coefficient = (
                    24.0
                    * EPSILON
                    * (2.0 * inverse_radius_6**2 - inverse_radius_6)
                    * inverse_radius_2
                )
                force = coefficient * displacement
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Lennard-Jones force is numerically singular at this separation"
            ) from exc
        finite_result(force, "pair force")
        return np.asarray(force, dtype=np.float64)
