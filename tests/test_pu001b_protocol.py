"""Fast protocol tests; long PU-001B experiments stay outside CI."""

from __future__ import annotations

import unittest

import numpy as np

from particle_universe import FreeBoundary, PeriodicBoundary
from scripts.run_pu001b_validation import (
    log_log_slope,
    pair_active_mask,
    pbc_position_rms,
    run_forward_euler_series,
    velocity_rms,
)


class PU001BProtocolTests(unittest.TestCase):
    def test_pbc_aware_rms_uses_minimum_image(self) -> None:
        boundary = PeriodicBoundary(6.0)
        actual = np.array([[5.9, 0.0, 0.0], [0.1, 0.0, 0.0]])
        reference = np.array([[0.1, 0.0, 0.0], [5.9, 0.0, 0.0]])
        expected = np.sqrt((0.2**2 + 0.2**2) / 6.0)
        self.assertAlmostEqual(pbc_position_rms(actual, reference, boundary), expected)

    def test_velocity_rms_is_componentwise(self) -> None:
        first = np.ones((2, 3), dtype=np.float64)
        second = np.zeros((2, 3), dtype=np.float64)
        self.assertEqual(velocity_rms(first, second), 1.0)

    def test_cutoff_crossing_mask_is_strict(self) -> None:
        positions = np.array(
            [[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        mask = pair_active_mask(positions, FreeBoundary())
        self.assertTrue(np.array_equal(mask, np.array([True, False, False])))

    def test_log_log_slope_recovers_second_order(self) -> None:
        dt = np.array([0.001, 0.002, 0.004, 0.008])
        error = 3.0 * dt**2
        self.assertAlmostEqual(log_log_slope(dt, error), 2.0, places=13)

    def test_forward_euler_series_archives_energy_components(self) -> None:
        positions = np.array([[2.4, 3.0, 3.0], [3.6, 3.0, 3.0]])
        velocities = np.zeros((2, 3), dtype=np.float64)
        series, failure, _ = run_forward_euler_series(
            positions, velocities, PeriodicBoundary(6.0), dt=0.001, steps=2
        )
        self.assertIsNone(failure)
        self.assertEqual(series["time"].shape, (3,))
        np.testing.assert_array_equal(
            series["total_energy"],
            series["potential_energy"] + series["kinetic_energy"],
        )
        self.assertTrue(np.all(np.isfinite(series["total_energy"])))


if __name__ == "__main__":
    unittest.main()
