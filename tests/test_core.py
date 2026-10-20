"""Core validation and initialization tests outside the numbered matrix."""

from __future__ import annotations

import json
import unittest

import numpy as np

from particle_universe import (
    FreeBoundary,
    NVESimulation,
    ParticleOverlapError,
    PeriodicBoundary,
    ShiftedLennardJones,
    fcc_positions,
    maxwell_boltzmann_velocities,
    total_momentum,
    two_particle_positions,
)


class InitializationTests(unittest.TestCase):
    def test_fcc_particle_count_dtype_and_box(self) -> None:
        configuration = fcc_positions(3, 0.8)
        self.assertEqual(configuration.positions.shape, (108, 3))
        self.assertEqual(configuration.positions.dtype, np.dtype(np.float64))
        self.assertGreater(configuration.box_length / 2.0, 2.5)
        self.assertTrue(np.all(configuration.positions >= 0.0))
        self.assertTrue(np.all(configuration.positions < configuration.box_length))

    def test_velocity_seed_and_zero_center_of_mass_momentum(self) -> None:
        first = maxwell_boltzmann_velocities(108, 1.0, seed=7)
        second = maxwell_boltzmann_velocities(108, 1.0, seed=7)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(np.abs(total_momentum(first)) < 1.0e-14))

    def test_legal_periodic_two_particle_state(self) -> None:
        positions = two_particle_positions(1.2, box_length=6.0)
        self.assertTrue(np.all(positions >= 0.0))
        self.assertTrue(np.all(positions < 6.0))
        displacement = PeriodicBoundary(6.0).displacement(
            positions[0] - positions[1]
        )
        self.assertAlmostEqual(float(np.linalg.norm(displacement)), 1.2)


class RejectionTests(unittest.TestCase):
    def test_nonfinite_inputs_are_rejected(self) -> None:
        positions = two_particle_positions(1.2)
        velocities = np.zeros((2, 3), dtype=np.float64)
        positions[0, 0] = np.nan
        with self.assertRaises(ValueError):
            NVESimulation(positions, velocities, FreeBoundary())

        positions = two_particle_positions(1.2)
        velocities[0, 0] = np.inf
        with self.assertRaises(ValueError):
            NVESimulation(positions, velocities, FreeBoundary())

    def test_exact_overlap_is_rejected(self) -> None:
        potential = ShiftedLennardJones()
        positions = np.zeros((2, 3), dtype=np.float64)
        with self.assertRaises(ParticleOverlapError):
            potential.evaluate(positions, FreeBoundary())

    def test_periodic_image_overlap_is_rejected(self) -> None:
        potential = ShiftedLennardJones()
        positions = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        with self.assertRaises(ParticleOverlapError):
            potential.evaluate(positions, PeriodicBoundary(6.0))

    def test_numerical_singularity_is_rejected(self) -> None:
        potential = ShiftedLennardJones()
        displacement = np.array([1.0e-200, 0.0, 0.0], dtype=np.float64)
        with self.assertRaises((FloatingPointError, ParticleOverlapError)):
            potential.pair_force(displacement)

    def test_illegal_scalar_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fcc_positions(0, 0.8)
        with self.assertRaises(ValueError):
            fcc_positions(3, np.inf)
        with self.assertRaises(ValueError):
            maxwell_boltzmann_velocities(2, -1.0, seed=1)
        with self.assertRaises(ValueError):
            maxwell_boltzmann_velocities(2, 1.0, seed=-1)


class DiagnosticsTests(unittest.TestCase):
    def test_run_record_is_complete_and_json_safe(self) -> None:
        positions = two_particle_positions(1.3)
        velocities = maxwell_boltzmann_velocities(2, 0.1, seed=11)
        simulation = NVESimulation(positions, velocities, FreeBoundary())
        result = simulation.run(
            steps=2,
            dt=1.0e-4,
            seed=11,
            density=None,
            initial_temperature=0.1,
            solver_commit="unit-test",
        )
        required = {
            "solver_commit",
            "reference_commit",
            "seed",
            "N",
            "dim",
            "rho_star",
            "initial_temperature",
            "temperature_dof",
            "sigma",
            "epsilon",
            "mass",
            "k_B",
            "rc",
            "potential_variant",
            "integrator",
            "dt",
            "steps",
            "t_end",
            "dt_ref",
            "crossing_count",
            "boundary",
            "L",
            "minimum_image_valid",
            "energy_drift",
            "momentum_drift",
            "force_evaluations",
            "timestamps",
            "wall_time_seconds",
            "environment",
        }
        self.assertEqual(set(result.metadata), required)
        self.assertEqual(result.metadata["force_evaluations"], {"initial": 1, "step": 2})
        payload = json.dumps(result.to_dict(), allow_nan=False)
        self.assertNotIn("NaN", payload)
        self.assertNotIn("Infinity", payload)


if __name__ == "__main__":
    unittest.main()
