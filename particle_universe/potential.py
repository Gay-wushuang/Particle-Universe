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
        self._all_pair_indices: dict[int, tuple[np.ndarray, np.ndarray]] = {}

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
        first, second = self._pair_indices(coordinates.shape[0])
        displacements = boundary.displacement(
            coordinates[first] - coordinates[second]
        )
        radius_squared = np.einsum(
            "ij,ij->i", displacements, displacements, dtype=np.float64
        )
        overlaps = np.flatnonzero(radius_squared == 0.0)
        if overlaps.size:
            pair = int(overlaps[0])
            raise ParticleOverlapError(
                f"particles {int(first[pair])} and {int(second[pair])} overlap "
                "under the selected boundary"
            )

        active = radius_squared < self._cutoff_squared
        active_first = first[active]
        active_second = second[active]
        active_displacements = displacements[active]
        active_radius_squared = radius_squared[active]
        interacting_pairs = int(np.count_nonzero(active))

        if interacting_pairs:
            pair_forces = self._force_inside_cutoff(
                active_displacements, active_radius_squared
            )
            pair_energies = self._energy_inside_cutoff(active_radius_squared)
            np.add.at(forces, active_first, pair_forces)
            np.add.at(forces, active_second, -pair_forces)
            potential_energy = np.float64(
                np.sum(pair_energies, dtype=np.float64)
            )
        else:
            potential_energy = np.float64(0.0)

        finite_result(forces, "forces")
        finite_result(potential_energy, "potential energy")
        return ForceResult(
            forces=forces,
            potential_energy=np.float64(potential_energy),
            interacting_pairs=interacting_pairs,
        )

    def _pair_indices(self, particle_count: int) -> tuple[np.ndarray, np.ndarray]:
        indices = self._all_pair_indices.get(particle_count)
        if indices is None:
            # This is the complete O(N^2) pair set, not a cutoff neighbor list.
            indices = np.triu_indices(particle_count, k=1)
            self._all_pair_indices[particle_count] = indices
        return indices

    def _energy_inside_cutoff(self, radius_squared: np.ndarray) -> np.ndarray:
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
        return np.asarray(energy, dtype=np.float64)

    def _force_inside_cutoff(
        self, displacement: np.ndarray, radius_squared: np.ndarray | np.float64
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
                force = coefficient[..., np.newaxis] * displacement
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Lennard-Jones force is numerically singular at this separation"
            ) from exc
        finite_result(force, "pair force")
        return np.asarray(force, dtype=np.float64)
