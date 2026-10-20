"""Equivalence checks against the exact potential implementation at ed4faad."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from types import ModuleType

import numpy as np

from particle_universe import FreeBoundary, PeriodicBoundary, fcc_positions
from particle_universe.potential import ShiftedLennardJones

OLD_COMMIT = "ed4faad4bd571f2a5ba8c7fabe2a45961792d326"
NEW_COMMIT = "b219f95e8b55017f08e073f9981274e3834827e7"
HISTORICAL_PATH = "particle_universe/potential.py"

# Summation order changed from scalar pair accumulation to NumPy reduction.
# These mixed tolerances are deliberately much larger than eps but remain close
# to the O(n * eps) roundoff scale for the tested <= 108-particle systems.
ABSOLUTE_TOLERANCE = 5.0e-12
RELATIVE_TOLERANCE = 5.0e-14


def load_historical_potential() -> ModuleType:
    source = subprocess.run(
        ["git", "show", f"{OLD_COMMIT}:{HISTORICAL_PATH}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    name = "particle_universe._potential_ed4faad"
    spec = importlib.util.spec_from_loader(name, loader=None)
    if spec is None:
        raise RuntimeError("could not construct historical module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, f"{OLD_COMMIT}:{HISTORICAL_PATH}", "exec"), module.__dict__)
    return module


def random_legal_positions(seed: int, count: int, box_length: float) -> np.ndarray:
    generator = np.random.default_rng(seed)
    positions: list[np.ndarray] = []
    boundary = PeriodicBoundary(box_length)
    while len(positions) < count:
        candidate = generator.uniform(0.0, box_length, size=3)
        if all(
            np.linalg.norm(boundary.displacement(candidate - other)) > 0.82
            for other in positions
        ):
            positions.append(candidate)
    return np.asarray(positions, dtype=np.float64)


class PotentialHistoryEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        historical = load_historical_potential()
        cls.old = historical.ShiftedLennardJones()
        cls.old_overlap_error = historical.ParticleOverlapError
        cls.new = ShiftedLennardJones()

    def assert_equivalent(self, positions: np.ndarray, boundary: object) -> None:
        old = self.old.evaluate(positions, boundary)
        new = self.new.evaluate(positions, boundary)
        self.assertEqual(old.interacting_pairs, new.interacting_pairs)
        np.testing.assert_allclose(
            new.potential_energy,
            old.potential_energy,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
        np.testing.assert_allclose(
            new.forces,
            old.forces,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
        np.testing.assert_allclose(
            np.sum(new.forces, axis=0, dtype=np.float64),
            np.zeros(3),
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        )

    def test_two_body_analytic_and_cutoff_regimes(self) -> None:
        cutoff = 2.5
        distances = (
            0.9,
            2.0 ** (1.0 / 6.0),
            1.5,
            2.0,
            2.4,
            np.nextafter(cutoff, 0.0),
            cutoff,
            np.nextafter(cutoff, np.inf),
            2.6,
        )
        for distance in distances:
            with self.subTest(distance=distance):
                positions = np.array(
                    [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float64
                )
                self.assert_equivalent(positions, FreeBoundary())
                displacement = positions[0] - positions[1]
                np.testing.assert_allclose(
                    self.new.pair_force(displacement),
                    self.old.pair_force(displacement),
                    rtol=RELATIVE_TOLERANCE,
                    atol=ABSOLUTE_TOLERANCE,
                )

    def test_asymmetric_random_multibody_multiple_seeds(self) -> None:
        boundary = PeriodicBoundary(8.0)
        for seed in (0, 1, 7, 20261020):
            with self.subTest(seed=seed):
                self.assert_equivalent(
                    random_legal_positions(seed, count=17, box_length=8.0), boundary
                )

    def test_fcc_configuration(self) -> None:
        configuration = fcc_positions(3, 0.8)
        self.assert_equivalent(
            configuration.positions, PeriodicBoundary(configuration.box_length)
        )

    def test_pbc_minimum_image(self) -> None:
        boundary = PeriodicBoundary(6.0)
        positions = np.array(
            [
                [0.05, 0.10, 0.15],
                [5.85, 0.10, 0.15],
                [3.20, 2.90, 5.80],
                [3.10, 2.80, 0.25],
            ],
            dtype=np.float64,
        )
        self.assert_equivalent(positions, boundary)

    def test_overlap_rejection_matches(self) -> None:
        boundary = PeriodicBoundary(6.0)
        direct = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        periodic = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        for positions in (direct, periodic):
            with self.subTest(positions=positions.tolist()):
                with self.assertRaises(self.old_overlap_error):
                    self.old.evaluate(positions, boundary)
                with self.assertRaises(ValueError):
                    self.new.evaluate(positions, boundary)


if __name__ == "__main__":
    unittest.main()
