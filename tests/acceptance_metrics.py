"""Measured PU-001A acceptance checks shared by tests and the record script."""

from __future__ import annotations

from typing import Any

import numpy as np

from particle_universe import (
    CUTOFF,
    FreeBoundary,
    NVESimulation,
    PeriodicBoundary,
    ShiftedLennardJones,
    fcc_positions,
    maxwell_boltzmann_velocities,
    two_particle_positions,
)


def measure_m1() -> dict[str, Any]:
    potential = ShiftedLennardJones()
    boundary = FreeBoundary()
    generator = np.random.default_rng(20261020)
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.2, 1.2, 0.0],
            [0.1, 0.2, 1.3],
        ],
        dtype=np.float64,
    )
    step_sizes = (1.0e-4, 1.0e-5, 1.0e-6)
    errors: dict[str, list[float]] = {f"{step:.0e}": [] for step in step_sizes}

    for _ in range(4):
        positions = base + generator.uniform(-0.025, 0.025, size=base.shape)
        analytic = potential.evaluate(positions, boundary).forces
        denominator = np.linalg.norm(analytic)
        for step in step_sizes:
            numerical = np.empty_like(analytic)
            for particle in range(positions.shape[0]):
                for dimension in range(positions.shape[1]):
                    plus = positions.copy()
                    minus = positions.copy()
                    plus[particle, dimension] += step
                    minus[particle, dimension] -= step
                    energy_plus = potential.evaluate(plus, boundary).potential_energy
                    energy_minus = potential.evaluate(minus, boundary).potential_energy
                    numerical[particle, dimension] = -(
                        energy_plus - energy_minus
                    ) / (2.0 * step)
            relative_error = float(np.linalg.norm(numerical - analytic) / denominator)
            errors[f"{step:.0e}"].append(relative_error)

    maximum = {key: max(values) for key, values in errors.items()}
    passed = bool(
        maximum["1e-05"] < 1.0e-6
        and maximum["1e-06"] < 1.0e-6
        and maximum["1e-05"] < maximum["1e-04"]
    )
    return {
        "id": "M1",
        "threshold": "relative error < 1e-6 at h=1e-5; stable as h decreases",
        "maximum_relative_error_by_h": maximum,
        "configurations": 4,
        "passed": passed,
    }


def measure_m2() -> dict[str, Any]:
    potential = ShiftedLennardJones()
    boundary = FreeBoundary()
    separations = (0.9, 1.12, 1.5, 2.0, 2.4)
    errors: dict[str, float] = {}
    for separation in separations:
        positions = two_particle_positions(separation)
        displacement = positions[0] - positions[1]
        radius_squared = np.dot(displacement, displacement)
        expected = (
            24.0
            * (2.0 * radius_squared ** -7 - radius_squared ** -4)
            * displacement
        )
        actual = potential.evaluate(positions, boundary).forces[0]
        errors[str(separation)] = float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        )
    maximum = max(errors.values())
    return {
        "id": "M2",
        "threshold": "relative error < 1e-12",
        "relative_error_by_separation": errors,
        "maximum_relative_error": maximum,
        "passed": bool(maximum < 1.0e-12),
    }


def measure_m3() -> dict[str, Any]:
    potential = ShiftedLennardJones()
    boundary = FreeBoundary()
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.6, 0.0, 0.0], [5.2, 0.0, 0.0]],
        dtype=np.float64,
    )
    inside_distances: list[float] = []
    raw_energy = 0.0
    for i in range(positions.shape[0] - 1):
        for j in range(i + 1, positions.shape[0]):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance < CUTOFF:
                inside_distances.append(distance)
                raw_energy += 4.0 * (distance**-12 - distance**-6)
    cutoff_energy = 4.0 * (CUTOFF**-12 - CUTOFF**-6)
    expected = raw_energy - len(inside_distances) * cutoff_energy
    incorrect_particle_count_shift = raw_energy - positions.shape[0] * cutoff_energy
    actual = potential.evaluate(positions, boundary)
    absolute_error = abs(float(actual.potential_energy) - float(expected))
    return {
        "id": "M3",
        "threshold": "absolute error < 1e-12 and shift uses N_pairs",
        "interacting_pairs": actual.interacting_pairs,
        "expected_interacting_pairs": len(inside_distances),
        "absolute_error": absolute_error,
        "distance_from_incorrect_N_shift": abs(
            float(actual.potential_energy) - float(incorrect_particle_count_shift)
        ),
        "passed": bool(
            absolute_error < 1.0e-12
            and actual.interacting_pairs == len(inside_distances)
            and abs(float(actual.potential_energy) - incorrect_particle_count_shift)
            > 1.0e-12
        ),
    }


def measure_n3() -> dict[str, Any]:
    legal = PeriodicBoundary(6.0)
    mapped = legal.displacement(np.array([5.8, -5.7, 0.1], dtype=np.float64))
    expected = np.array([-0.2, 0.3, 0.1], dtype=np.float64)
    equality_rejected = False
    below_rejected = False
    try:
        PeriodicBoundary(5.0)
    except ValueError:
        equality_rejected = True
    try:
        PeriodicBoundary(4.9)
    except ValueError:
        below_rejected = True
    mapping_error = float(np.max(np.abs(mapped - expected)))
    return {
        "id": "N3",
        "threshold": "L/2 <= rc rejected; legal minimum image is correct",
        "equality_rejected": equality_rejected,
        "below_rejected": below_rejected,
        "mapping_max_absolute_error": mapping_error,
        "passed": bool(equality_rejected and below_rejected and mapping_error < 1.0e-15),
    }


def measure_n4() -> dict[str, Any]:
    positions = two_particle_positions(1.3)
    velocities = np.array([[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float64)
    simulation = NVESimulation(positions, velocities, FreeBoundary())
    requested_steps = 7
    for _ in range(requested_steps):
        simulation.step(1.0e-4)
    counts = simulation.force_evaluations.to_dict()
    return {
        "id": "N4",
        "threshold": "one initial evaluation and exactly one evaluation per step",
        "steps": requested_steps,
        "initial_force_evaluations": counts["initial"],
        "step_force_evaluations": counts["step"],
        "passed": bool(counts == {"initial": 1, "step": requested_steps}),
    }


def _reproducible_run() -> Any:
    configuration = fcc_positions(3, 0.8)
    velocities = maxwell_boltzmann_velocities(
        configuration.positions.shape[0], 1.0, seed=42
    )
    simulation = NVESimulation(
        configuration.positions,
        velocities,
        PeriodicBoundary(configuration.box_length),
    )
    return simulation.run(
        steps=5,
        dt=0.002,
        seed=42,
        density=0.8,
        initial_temperature=1.0,
        solver_commit="acceptance-test",
    )


def measure_r1() -> dict[str, Any]:
    first = _reproducible_run()
    second = _reproducible_run()
    positions_equal = all(
        np.array_equal(left, right)
        for left, right in zip(first.positions, second.positions, strict=True)
    )
    velocities_equal = all(
        np.array_equal(left, right)
        for left, right in zip(first.velocities, second.velocities, strict=True)
    )
    first_energy = np.asarray(
        [item.total_energy for item in first.samples], dtype=np.float64
    )
    second_energy = np.asarray(
        [item.total_energy for item in second.samples], dtype=np.float64
    )
    energy_equal = np.array_equal(first_energy, second_energy)
    return {
        "id": "R1",
        "threshold": "trajectory and energy sequence bit-identical",
        "positions_bit_identical": positions_equal,
        "velocities_bit_identical": velocities_equal,
        "energy_bit_identical": energy_equal,
        "samples": len(first.samples),
        "passed": bool(positions_equal and velocities_equal and energy_equal),
    }


def measure_all() -> dict[str, dict[str, Any]]:
    return {
        "M1": measure_m1(),
        "M2": measure_m2(),
        "M3": measure_m3(),
        "N3": measure_n3(),
        "N4": measure_n4(),
        "R1": measure_r1(),
    }
