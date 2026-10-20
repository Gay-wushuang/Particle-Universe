"""Executable acceptance assertions for the PU-001A-authorized test subset."""

from __future__ import annotations

import unittest

from tests.acceptance_metrics import (
    measure_m1,
    measure_m2,
    measure_m3,
    measure_n3,
    measure_n4,
    measure_r1,
)


class AcceptanceTests(unittest.TestCase):
    def test_m1_force_is_negative_energy_gradient(self) -> None:
        result = measure_m1()
        self.assertTrue(result["passed"], result)

    def test_m2_two_body_analytic_force(self) -> None:
        result = measure_m2()
        self.assertTrue(result["passed"], result)

    def test_m3_shift_is_applied_per_interacting_pair(self) -> None:
        result = measure_m3()
        self.assertTrue(result["passed"], result)

    def test_n3_minimum_image_geometry_is_enforced(self) -> None:
        result = measure_n3()
        self.assertTrue(result["passed"], result)

    def test_n4_force_evaluation_accounting(self) -> None:
        result = measure_n4()
        self.assertTrue(result["passed"], result)

    def test_r1_same_seed_is_bit_identical(self) -> None:
        result = measure_r1()
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
