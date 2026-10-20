"""Run the frozen PU-001B numerical validation matrix and archive raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
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
from particle_universe.diagnostics import kinetic_energy, temperature, total_momentum
from particle_universe.state import ParticleState

SEED = 20261020
DT_CANDIDATE = 0.002
MULTIBODY_PARTICLES = 256
MULTIBODY_DENSITY = 0.8
MULTIBODY_TEMPERATURE = 1.0
CHINA_STANDARD_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")


@dataclass(frozen=True, slots=True)
class FinalStateRun:
    positions: np.ndarray
    velocities: np.ndarray
    crossings_per_step: np.ndarray
    duration_seconds: float


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def pair_active_mask(
    positions: np.ndarray,
    boundary: FreeBoundary | PeriodicBoundary,
) -> np.ndarray:
    first, second = np.triu_indices(positions.shape[0], k=1)
    displacement = boundary.displacement(positions[first] - positions[second])
    radius_squared = np.einsum(
        "ij,ij->i", displacement, displacement, dtype=np.float64
    )
    return radius_squared < CUTOFF**2


def pbc_position_rms(
    actual: np.ndarray,
    reference: np.ndarray,
    boundary: FreeBoundary | PeriodicBoundary,
) -> float:
    difference = boundary.displacement(actual - reference)
    return float(np.sqrt(np.mean(difference * difference, dtype=np.float64)))


def velocity_rms(actual: np.ndarray, reference: np.ndarray) -> float:
    difference = actual - reference
    return float(np.sqrt(np.mean(difference * difference, dtype=np.float64)))


def log_log_slope(step_sizes: np.ndarray, errors: np.ndarray) -> float:
    if np.any(step_sizes <= 0.0) or np.any(errors <= 0.0):
        raise ValueError("log-log slope requires strictly positive values")
    return float(np.polyfit(np.log(step_sizes), np.log(errors), 1)[0])


def run_final_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    boundary: FreeBoundary | PeriodicBoundary,
    *,
    dt: float,
    t_end: float,
) -> FinalStateRun:
    steps = round(t_end / dt)
    if not math.isclose(steps * dt, t_end, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("t_end must be an integer multiple of dt")
    simulation = NVESimulation(positions, velocities, boundary)
    previous_active = pair_active_mask(simulation.state.positions, boundary)
    crossings = np.zeros(steps, dtype=np.int64)
    started = perf_counter()
    for index in range(steps):
        simulation.step(dt)
        current_active = pair_active_mask(simulation.state.positions, boundary)
        crossings[index] = np.count_nonzero(current_active != previous_active)
        previous_active = current_active
    duration = perf_counter() - started
    return FinalStateRun(
        positions=simulation.state.positions.copy(),
        velocities=simulation.state.velocities.copy(),
        crossings_per_step=crossings,
        duration_seconds=duration,
    )


def run_n1a(output_directory: Path) -> dict[str, Any]:
    started = perf_counter()
    boundary = FreeBoundary()
    separation = 1.2
    positions = two_particle_positions(separation)
    velocities = np.array(
        [[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64
    )
    t_end = 0.08
    step_sizes = np.array([0.001, 0.002, 0.004, 0.008], dtype=np.float64)
    dt_reference = 1.0e-5
    dt_reference_half = dt_reference / 2.0

    reference = run_final_state(
        positions, velocities, boundary, dt=dt_reference, t_end=t_end
    )
    reference_half = run_final_state(
        positions, velocities, boundary, dt=dt_reference_half, t_end=t_end
    )
    reference_position_difference = pbc_position_rms(
        reference.positions, reference_half.positions, boundary
    )
    reference_velocity_difference = velocity_rms(
        reference.velocities, reference_half.velocities
    )
    reference_converged = bool(
        max(reference_position_difference, reference_velocity_difference) < 1.0e-10
    )

    position_errors = np.empty(step_sizes.size, dtype=np.float64)
    velocity_errors = np.empty(step_sizes.size, dtype=np.float64)
    crossing_counts = np.empty(step_sizes.size, dtype=np.int64)
    durations = np.empty(step_sizes.size, dtype=np.float64)
    final_positions = np.empty((step_sizes.size, 2, 3), dtype=np.float64)
    final_velocities = np.empty((step_sizes.size, 2, 3), dtype=np.float64)
    for index, dt in enumerate(step_sizes):
        run = run_final_state(
            positions, velocities, boundary, dt=float(dt), t_end=t_end
        )
        position_errors[index] = pbc_position_rms(
            run.positions, reference_half.positions, boundary
        )
        velocity_errors[index] = velocity_rms(
            run.velocities, reference_half.velocities
        )
        crossing_counts[index] = np.sum(run.crossings_per_step, dtype=np.int64)
        durations[index] = run.duration_seconds
        final_positions[index] = run.positions
        final_velocities[index] = run.velocities

    position_slope = log_log_slope(step_sizes, position_errors)
    velocity_slope = log_log_slope(step_sizes, velocity_errors)
    position_monotonic = bool(np.all(np.diff(position_errors) > 0.0))
    velocity_monotonic = bool(np.all(np.diff(velocity_errors) > 0.0))
    passed = bool(
        reference_converged
        and np.all(crossing_counts == 0)
        and 1.7 <= position_slope <= 2.3
        and 1.7 <= velocity_slope <= 2.3
        and position_monotonic
        and velocity_monotonic
    )

    raw_path = output_directory / "n1a_raw.npz"
    np.savez_compressed(
        raw_path,
        step_sizes=step_sizes,
        position_errors=position_errors,
        velocity_errors=velocity_errors,
        crossing_counts=crossing_counts,
        durations_seconds=durations,
        initial_positions=positions,
        initial_velocities=velocities,
        final_positions=final_positions,
        final_velocities=final_velocities,
        reference_position=reference.positions,
        reference_velocity=reference.velocities,
        reference_half_position=reference_half.positions,
        reference_half_velocity=reference_half.velocities,
    )
    return {
        "id": "N1a",
        "outcome": "PASS" if passed else "FAIL",
        "threshold": {
            "position_slope": "2.0 +/- 0.3",
            "velocity_slope": "2.0 +/- 0.3",
            "reference_halving_max_rms": "< 1e-10",
            "crossing_count": 0,
            "errors_monotonic_with_dt": True,
        },
        "parameters": {
            "N": 2,
            "boundary": "free",
            "initial_separation": separation,
            "initial_velocities": velocities.tolist(),
            "seed": SEED,
            "t_end": t_end,
            "dt_values": step_sizes.tolist(),
            "dt_ref": dt_reference,
            "dt_ref_half": dt_reference_half,
        },
        "measured": {
            "position_errors": position_errors.tolist(),
            "velocity_errors": velocity_errors.tolist(),
            "position_slope": position_slope,
            "velocity_slope": velocity_slope,
            "position_errors_monotonic": position_monotonic,
            "velocity_errors_monotonic": velocity_monotonic,
            "crossing_counts": crossing_counts.tolist(),
            "reference_position_rms": reference_position_difference,
            "reference_velocity_rms": reference_velocity_difference,
            "reference_converged": reference_converged,
        },
        "duration_seconds": perf_counter() - started,
        "raw_data": raw_path.name,
    }


def multibody_initial_state() -> tuple[np.ndarray, np.ndarray, PeriodicBoundary, float]:
    configuration = fcc_positions(4, MULTIBODY_DENSITY)
    velocities = maxwell_boltzmann_velocities(
        MULTIBODY_PARTICLES, MULTIBODY_TEMPERATURE, seed=SEED
    )
    boundary = PeriodicBoundary(configuration.box_length)
    return configuration.positions, velocities, boundary, float(configuration.box_length)


def run_n1b(output_directory: Path) -> dict[str, Any]:
    started = perf_counter()
    positions, velocities, boundary, box_length = multibody_initial_state()
    t_end = 0.08
    step_sizes = np.array([0.001, 0.002, 0.004, 0.008], dtype=np.float64)
    dt_reference = 1.0e-5
    dt_reference_half = dt_reference / 2.0
    reference = run_final_state(
        positions, velocities, boundary, dt=dt_reference, t_end=t_end
    )
    reference_half = run_final_state(
        positions, velocities, boundary, dt=dt_reference_half, t_end=t_end
    )

    position_errors = np.empty(step_sizes.size, dtype=np.float64)
    velocity_errors = np.empty(step_sizes.size, dtype=np.float64)
    crossing_counts = np.empty(step_sizes.size, dtype=np.int64)
    durations = np.empty(step_sizes.size, dtype=np.float64)
    for index, dt in enumerate(step_sizes):
        run = run_final_state(
            positions, velocities, boundary, dt=float(dt), t_end=t_end
        )
        position_errors[index] = pbc_position_rms(
            run.positions, reference_half.positions, boundary
        )
        velocity_errors[index] = velocity_rms(
            run.velocities, reference_half.velocities
        )
        crossing_counts[index] = np.sum(run.crossings_per_step, dtype=np.int64)
        durations[index] = run.duration_seconds

    position_slope = log_log_slope(step_sizes, position_errors)
    velocity_slope = log_log_slope(step_sizes, velocity_errors)
    reference_position_difference = pbc_position_rms(
        reference.positions, reference_half.positions, boundary
    )
    reference_velocity_difference = velocity_rms(
        reference.velocities, reference_half.velocities
    )
    raw_path = output_directory / "n1b_raw.npz"
    np.savez_compressed(
        raw_path,
        step_sizes=step_sizes,
        position_errors=position_errors,
        velocity_errors=velocity_errors,
        crossing_counts=crossing_counts,
        durations_seconds=durations,
        reference_crossings=reference.crossings_per_step,
        reference_half_crossings=reference_half.crossings_per_step,
        reference_position=reference.positions,
        reference_velocity=reference.velocities,
        reference_half_position=reference_half.positions,
        reference_half_velocity=reference_half.velocities,
    )
    return {
        "id": "N1b",
        "outcome": "PASS",
        "threshold": "diagnostic only; record crossings, errors, and observed slopes",
        "parameters": {
            "N": MULTIBODY_PARTICLES,
            "rho_star": MULTIBODY_DENSITY,
            "initial_temperature": MULTIBODY_TEMPERATURE,
            "seed": SEED,
            "boundary": "periodic",
            "L": box_length,
            "t_end": t_end,
            "dt_values": step_sizes.tolist(),
            "dt_ref": dt_reference,
            "dt_ref_half": dt_reference_half,
        },
        "measured": {
            "position_errors": position_errors.tolist(),
            "velocity_errors": velocity_errors.tolist(),
            "observed_position_slope": position_slope,
            "observed_velocity_slope": velocity_slope,
            "crossing_counts": crossing_counts.tolist(),
            "reference_crossing_count": int(np.sum(reference.crossings_per_step)),
            "reference_half_crossing_count": int(
                np.sum(reference_half.crossings_per_step)
            ),
            "reference_halving_position_rms": reference_position_difference,
            "reference_halving_velocity_rms": reference_velocity_difference,
            "reference_accuracy_caveat": (
                "The reference itself crosses the discontinuous cutoff; halving is "
                "reported but is weaker evidence than in N1a."
            ),
        },
        "duration_seconds": perf_counter() - started,
        "raw_data": raw_path.name,
    }


def run_velocity_verlet_series(
    positions: np.ndarray,
    velocities: np.ndarray,
    boundary: PeriodicBoundary,
    *,
    dt: float,
    steps: int,
    track_crossings: bool,
) -> dict[str, np.ndarray | float]:
    simulation = NVESimulation(positions, velocities, boundary)
    times = np.arange(steps + 1, dtype=np.float64) * dt
    potential = np.empty(steps + 1, dtype=np.float64)
    kinetic = np.empty(steps + 1, dtype=np.float64)
    total = np.empty(steps + 1, dtype=np.float64)
    temperatures = np.empty(steps + 1, dtype=np.float64)
    momentum = np.empty((steps + 1, 3), dtype=np.float64)
    crossings = np.zeros(steps + 1, dtype=np.int64)

    def store(index: int) -> None:
        state = simulation.state
        potential[index] = state.potential_energy
        kinetic[index] = state.kinetic_energy
        total[index] = state.potential_energy + state.kinetic_energy
        temperatures[index] = temperature(state.kinetic_energy, state.particle_count)
        momentum[index] = total_momentum(state.velocities)

    store(0)
    previous_active = (
        pair_active_mask(simulation.state.positions, boundary)
        if track_crossings
        else None
    )
    started = perf_counter()
    for index in range(1, steps + 1):
        simulation.step(dt)
        if previous_active is not None:
            current_active = pair_active_mask(simulation.state.positions, boundary)
            crossings[index] = np.count_nonzero(current_active != previous_active)
            previous_active = current_active
        store(index)
    duration = perf_counter() - started
    return {
        "time": times,
        "potential_energy": potential,
        "kinetic_energy": kinetic,
        "total_energy": total,
        "temperature": temperatures,
        "momentum": momentum,
        "crossings": crossings,
        "duration_seconds": duration,
    }


def analyze_long_nve(
    output_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray | float]]:
    positions, velocities, boundary, box_length = multibody_initial_state()
    steps = 10_000
    series = run_velocity_verlet_series(
        positions,
        velocities,
        boundary,
        dt=DT_CANDIDATE,
        steps=steps,
        track_crossings=True,
    )
    energy = np.asarray(series["total_energy"])
    time = np.asarray(series["time"])
    momentum = np.asarray(series["momentum"])
    crossings = np.asarray(series["crossings"])
    initial_energy = energy[0]
    relative = (energy - initial_energy) / abs(initial_energy)
    maximum_relative = float(np.max(np.abs(relative)))
    slope = float(np.polyfit(time, relative, 1)[0])
    trend_over_run = slope * float(time[-1] - time[0])
    differences = np.diff(energy)
    monotonic = bool(np.all(differences >= 0.0) or np.all(differences <= 0.0))
    momentum_drift = np.linalg.norm(momentum - momentum[0], axis=1)
    maximum_momentum_drift = float(np.max(momentum_drift))
    momentum_threshold = 1.0e-12 * math.sqrt(MULTIBODY_PARTICLES)

    crossing_steps = crossings[1:] > 0
    absolute_energy_steps = np.abs(differences)
    mean_jump_crossing = (
        float(np.mean(absolute_energy_steps[crossing_steps]))
        if np.any(crossing_steps)
        else None
    )
    mean_jump_no_crossing = (
        float(np.mean(absolute_energy_steps[~crossing_steps]))
        if np.any(~crossing_steps)
        else None
    )
    short_end = min(1000, steps)
    short_maximum = float(np.max(np.abs(relative[: short_end + 1])))
    block = max(1, steps // 10)
    first_block_mean = float(np.mean(relative[:block]))
    last_block_mean = float(np.mean(relative[-block:]))

    raw_path = output_directory / "nve_long_raw.npz"
    np.savez_compressed(
        raw_path,
        time=time,
        potential_energy=series["potential_energy"],
        kinetic_energy=series["kinetic_energy"],
        total_energy=energy,
        temperature=series["temperature"],
        momentum=momentum,
        relative_energy_change=relative,
        cutoff_crossings=crossings,
    )

    common_parameters = {
        "N": MULTIBODY_PARTICLES,
        "rho_star": MULTIBODY_DENSITY,
        "initial_temperature": MULTIBODY_TEMPERATURE,
        "seed": SEED,
        "dt": DT_CANDIDATE,
        "steps": steps,
        "t_end": float(time[-1]),
        "L": box_length,
        "boundary": "periodic",
        "potential_variant": "A",
    }
    c1_passed = bool(maximum_relative < 1.0e-4 and not monotonic)
    c1 = {
        "id": "C1",
        "outcome": "PASS" if c1_passed else "FAIL",
        "threshold": "max |E(t)-E0|/|E0| < 1e-4 and no monotonic drift",
        "parameters": common_parameters,
        "measured": {
            "initial_energy": float(initial_energy),
            "maximum_relative_energy_change": maximum_relative,
            "linear_relative_drift_per_tau": slope,
            "linear_trend_over_run": trend_over_run,
            "strictly_monotonic": monotonic,
            "first_10_percent_mean_relative_change": first_block_mean,
            "last_10_percent_mean_relative_change": last_block_mean,
        },
        "duration_seconds": float(series["duration_seconds"]),
        "raw_data": raw_path.name,
    }
    c2 = {
        "id": "C2",
        "outcome": "PASS" if maximum_momentum_drift < momentum_threshold else "FAIL",
        "threshold": f"|P(t)-P(0)| < {momentum_threshold:.17g}",
        "parameters": common_parameters,
        "measured": {
            "initial_momentum": momentum[0].tolist(),
            "maximum_momentum_drift": maximum_momentum_drift,
        },
        "duration_seconds": float(series["duration_seconds"]),
        "raw_data": raw_path.name,
    }
    n5 = {
        "id": "N5",
        "outcome": "PASS",
        "threshold": "diagnostic only; C1 threshold is reported without fitting convergence order",
        "parameters": common_parameters,
        "measured": {
            "short_first_1000_step_max_relative_change": short_maximum,
            "long_max_relative_change": maximum_relative,
            "linear_relative_drift_per_tau": slope,
            "linear_trend_over_run": trend_over_run,
            "total_cutoff_crossings": int(np.sum(crossings, dtype=np.int64)),
            "steps_with_crossings": int(np.count_nonzero(crossing_steps)),
            "mean_absolute_energy_step_on_crossing": mean_jump_crossing,
            "mean_absolute_energy_step_without_crossing": mean_jump_no_crossing,
            "note": "Energy convergence order is not inferred from this series.",
        },
        "duration_seconds": float(series["duration_seconds"]),
        "raw_data": raw_path.name,
    }
    return n5, c1, c2, series


def run_c3(output_directory: Path) -> dict[str, Any]:
    started = perf_counter()
    box_length = 6.0
    boundary = PeriodicBoundary(box_length)
    initial_positions = two_particle_positions(1.2, box_length=box_length)
    initial_velocities = np.array(
        [[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]], dtype=np.float64
    )
    simulation = NVESimulation(initial_positions, initial_velocities, boundary)
    steps_each_way = 200
    dt = DT_CANDIDATE
    unwrapped_displacement = np.zeros_like(initial_positions)
    previous_positions = simulation.state.positions.copy()
    maximum_component_displacement = 0.0
    for _ in range(steps_each_way):
        simulation.step(dt)
        increment = boundary.displacement(simulation.state.positions - previous_positions)
        unwrapped_displacement += increment
        previous_positions = simulation.state.positions.copy()
        maximum_component_displacement = max(
            maximum_component_displacement,
            float(np.max(np.abs(unwrapped_displacement))),
        )
    forward_positions = simulation.state.positions.copy()
    forward_velocities = simulation.state.velocities.copy()
    simulation.state.velocities *= -1.0
    for _ in range(steps_each_way):
        simulation.step(dt)
    final_positions = simulation.state.positions.copy()
    final_velocities = simulation.state.velocities.copy()
    position_error = pbc_position_rms(final_positions, initial_positions, boundary)
    velocity_error = velocity_rms(final_velocities, -initial_velocities)
    net_displacement_valid = bool(maximum_component_displacement < box_length / 2.0)
    passed = bool(
        net_displacement_valid
        and position_error < 1.0e-10
        and velocity_error < 1.0e-10
    )
    raw_path = output_directory / "c3_raw.npz"
    np.savez_compressed(
        raw_path,
        initial_positions=initial_positions,
        initial_velocities=initial_velocities,
        forward_positions=forward_positions,
        forward_velocities=forward_velocities,
        final_positions=final_positions,
        final_velocities=final_velocities,
        unwrapped_forward_displacement=unwrapped_displacement,
    )
    return {
        "id": "C3",
        "outcome": "PASS" if passed else "FAIL",
        "threshold": {
            "position_rms": "< 1e-10 using minimum-image difference",
            "velocity_rms": "< 1e-10 against -v0",
            "net_displacement": "< L/2",
        },
        "parameters": {
            "N": 2,
            "seed": SEED,
            "dt": dt,
            "steps_each_way": steps_each_way,
            "T_short": dt * steps_each_way,
            "L": box_length,
            "initial_separation": 1.2,
        },
        "measured": {
            "position_rms": position_error,
            "velocity_rms": velocity_error,
            "maximum_forward_unwrapped_component_displacement": maximum_component_displacement,
            "net_displacement_valid": net_displacement_valid,
        },
        "duration_seconds": perf_counter() - started,
        "raw_data": raw_path.name,
    }


def energy_fluctuation(energy: np.ndarray) -> dict[str, float]:
    segment = energy[energy.size // 2 :]
    mean = float(np.mean(segment))
    standard_deviation = float(np.std(segment))
    midpoint = segment.size // 2
    first_mean = float(np.mean(segment[:midpoint]))
    second_mean = float(np.mean(segment[midpoint:]))
    return {
        "mean": mean,
        "relative_standard_deviation": standard_deviation / abs(mean),
        "half_mean_relative_shift": abs(second_mean - first_mean) / abs(mean),
    }


def run_c4(
    output_directory: Path,
    existing_dt_series: dict[str, np.ndarray | float],
) -> dict[str, Any]:
    started = perf_counter()
    positions, velocities, boundary, box_length = multibody_initial_state()
    step_sizes = np.array([0.001, 0.002, 0.004], dtype=np.float64)
    t_end = 20.0
    fluctuations = np.empty(step_sizes.size, dtype=np.float64)
    half_mean_shifts = np.empty(step_sizes.size, dtype=np.float64)
    raw: dict[str, np.ndarray] = {"step_sizes": step_sizes}
    durations: dict[str, float] = {}
    for index, dt in enumerate(step_sizes):
        if dt == DT_CANDIDATE:
            series = existing_dt_series
        else:
            series = run_velocity_verlet_series(
                positions,
                velocities,
                boundary,
                dt=float(dt),
                steps=round(t_end / float(dt)),
                track_crossings=False,
            )
        energy = np.asarray(series["total_energy"])
        analysis = energy_fluctuation(energy)
        fluctuations[index] = analysis["relative_standard_deviation"]
        half_mean_shifts[index] = analysis["half_mean_relative_shift"]
        key = f"dt_{str(float(dt)).replace('.', 'p')}"
        raw[f"{key}_time"] = np.asarray(series["time"])
        raw[f"{key}_total_energy"] = energy
        durations[str(float(dt))] = float(series["duration_seconds"])
    scaling_slope = log_log_slope(step_sizes, fluctuations)
    bounded = bool(np.all(np.isfinite(fluctuations)))
    scaling_consistent = bool(1.7 <= scaling_slope <= 2.3)
    outcome = "PASS" if bounded and scaling_consistent else "FAIL"
    raw_path = output_directory / "c4_raw.npz"
    np.savez_compressed(raw_path, **raw)
    return {
        "id": "C4",
        "outcome": outcome,
        "threshold": "bounded/stationary energy variance with O(dt^2) magnitude",
        "analysis_interpretation": (
            "O(dt^2) is evaluated with the same 2.0 +/- 0.3 slope band used "
            "for N1a; half-segment mean shifts are reported without a new threshold."
        ),
        "parameters": {
            "N": MULTIBODY_PARTICLES,
            "rho_star": MULTIBODY_DENSITY,
            "initial_temperature": MULTIBODY_TEMPERATURE,
            "seed": SEED,
            "dt_values": step_sizes.tolist(),
            "t_end": t_end,
            "analysis_segment": "second half of each trajectory",
            "L": box_length,
        },
        "measured": {
            "relative_standard_deviation": fluctuations.tolist(),
            "half_mean_relative_shifts": half_mean_shifts.tolist(),
            "observed_scaling_slope": scaling_slope,
            "bounded_finite": bounded,
            "scaling_consistent": scaling_consistent,
        },
        "per_run_duration_seconds": durations,
        "duration_seconds": perf_counter() - started,
        "raw_data": raw_path.name,
    }


def run_forward_euler_series(
    positions: np.ndarray,
    velocities: np.ndarray,
    boundary: PeriodicBoundary,
    *,
    dt: float,
    steps: int,
) -> tuple[dict[str, np.ndarray], str | None, float]:
    potential = ShiftedLennardJones()
    initial = potential.evaluate(positions, boundary)
    state = ParticleState(
        positions=boundary.wrap(positions),
        velocities=velocities,
        forces=initial.forces,
        time=0.0,
        potential_energy=initial.potential_energy,
        kinetic_energy=kinetic_energy(velocities),
    )
    time_values = np.empty(steps + 1, dtype=np.float64)
    energy_values = np.empty(steps + 1, dtype=np.float64)
    time_values[0] = 0.0
    energy_values[0] = state.potential_energy + state.kinetic_energy
    completed = 0
    failure: str | None = None
    started = perf_counter()
    for index in range(1, steps + 1):
        try:
            # Teaching-only explicit Forward Euler; never part of production integration.
            new_positions = boundary.wrap(state.positions + dt * state.velocities)
            new_velocities = state.velocities + dt * state.forces
            evaluated = potential.evaluate(new_positions, boundary)
            state = ParticleState(
                positions=new_positions,
                velocities=new_velocities,
                forces=evaluated.forces,
                time=index * dt,
                potential_energy=evaluated.potential_energy,
                kinetic_energy=kinetic_energy(new_velocities),
            )
        except (ValueError, FloatingPointError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break
        completed = index
        time_values[index] = index * dt
        energy_values[index] = state.potential_energy + state.kinetic_energy
    duration = perf_counter() - started
    return (
        {
            "time": time_values[: completed + 1],
            "total_energy": energy_values[: completed + 1],
        },
        failure,
        duration,
    )


def run_c5(output_directory: Path, c1_maximum_drift: float) -> dict[str, Any]:
    started = perf_counter()
    positions, velocities, boundary, box_length = multibody_initial_state()
    requested_steps = 10_000
    series, failure, run_duration = run_forward_euler_series(
        positions,
        velocities,
        boundary,
        dt=DT_CANDIDATE,
        steps=requested_steps,
    )
    energy = series["total_energy"]
    relative = (energy - energy[0]) / abs(energy[0])
    maximum_drift = float(np.max(np.abs(relative)))
    ratio = (
        math.inf if c1_maximum_drift == 0.0 else maximum_drift / c1_maximum_drift
    )
    orders = math.inf if math.isinf(ratio) else math.log10(ratio)
    if ratio >= 100.0:
        outcome = "PASS"
    elif ratio > 1.0:
        outcome = "INCONCLUSIVE"
    else:
        outcome = "FAIL"
    raw_path = output_directory / "c5_raw.npz"
    np.savez_compressed(
        raw_path,
        time=series["time"],
        total_energy=energy,
        relative_energy_change=relative,
    )
    return {
        "id": "C5",
        "outcome": outcome,
        "threshold": "Forward Euler energy drift >> Velocity-Verlet C1 drift",
        "analysis_interpretation": (
            "At least two orders of magnitude is treated as unambiguous '>>'; "
            "between one and one hundred times is INCONCLUSIVE, not a relaxed pass."
        ),
        "parameters": {
            "N": MULTIBODY_PARTICLES,
            "rho_star": MULTIBODY_DENSITY,
            "initial_temperature": MULTIBODY_TEMPERATURE,
            "seed": SEED,
            "dt": DT_CANDIDATE,
            "requested_steps": requested_steps,
            "L": box_length,
            "integrator": "Forward Euler teaching control",
        },
        "measured": {
            "completed_steps": int(energy.size - 1),
            "maximum_relative_energy_change": maximum_drift,
            "velocity_verlet_c1_maximum_relative_energy_change": c1_maximum_drift,
            "drift_ratio": ratio,
            "orders_of_magnitude_ratio": orders,
            "numerical_failure": failure,
        },
        "duration_seconds": perf_counter() - started,
        "integrator_run_duration_seconds": run_duration,
        "raw_data": raw_path.name,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(output_directory: Path, summary: dict[str, Any]) -> None:
    raw_files = sorted(output_directory.glob("*_raw.npz"))
    summary["raw_data_sha256"] = {
        path.name: sha256(path) for path in raw_files
    }
    summary_path = output_directory / "summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("n1a", "full"), default="full"
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    command = (
        "python scripts/run_pu001b_validation.py "
        f"--output-dir {arguments.output_dir.as_posix()} --phase {arguments.phase}"
    )
    summary: dict[str, Any] = {
        "task": "PU-001B",
        "solver_commit": git_output("rev-parse", "HEAD"),
        "solver_tree": git_output("rev-parse", "HEAD^{tree}"),
        "authorized_baseline": "ed4faad4bd571f2a5ba8c7fabe2a45961792d326",
        "command": command,
        "executed_at": datetime.now(CHINA_STANDARD_TIME).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "frozen_constants": {
            "mass": 1.0,
            "sigma": 1.0,
            "epsilon": 1.0,
            "k_B": 1.0,
            "rc": float(CUTOFF),
            "potential_variant": "A",
        },
        "results": {},
        "gate_decision": "INCOMPLETE",
    }
    overall_started = perf_counter()

    n1a = run_n1a(arguments.output_dir)
    summary["results"]["N1a"] = n1a
    write_summary(arguments.output_dir, summary)
    if n1a["outcome"] != "PASS" or arguments.phase == "n1a":
        summary["gate_decision"] = (
            "PASS" if n1a["outcome"] == "PASS" else "STOPPED_ON_N1A_FAILURE"
        )
        summary["total_duration_seconds"] = perf_counter() - overall_started
        write_summary(arguments.output_dir, summary)
        raise SystemExit(0 if n1a["outcome"] == "PASS" else 1)

    n1b = run_n1b(arguments.output_dir)
    summary["results"]["N1b"] = n1b
    write_summary(arguments.output_dir, summary)

    n5, c1, c2, long_series = analyze_long_nve(arguments.output_dir)
    summary["results"]["N5"] = n5
    summary["results"]["C1"] = c1
    if c1["outcome"] != "PASS":
        c2["outcome"] = "INCONCLUSIVE"
        c2["gate_note"] = "Measured in the same trajectory but not accepted because C1 failed."
        summary["results"]["C2"] = c2
        summary["gate_decision"] = "STOPPED_ON_C1_FAILURE"
        summary["not_executed_due_to_gate"] = ["C3", "C4", "C5"]
        summary["total_duration_seconds"] = perf_counter() - overall_started
        write_summary(arguments.output_dir, summary)
        raise SystemExit(1)

    summary["results"]["C2"] = c2
    write_summary(arguments.output_dir, summary)
    if c2["outcome"] != "PASS":
        summary["gate_decision"] = "STOPPED_ON_C2_FAILURE"
        summary["not_executed_due_to_gate"] = ["C3", "C4", "C5"]
        summary["total_duration_seconds"] = perf_counter() - overall_started
        write_summary(arguments.output_dir, summary)
        raise SystemExit(1)

    c3 = run_c3(arguments.output_dir)
    summary["results"]["C3"] = c3
    write_summary(arguments.output_dir, summary)
    if c3["outcome"] != "PASS":
        summary["gate_decision"] = "STOPPED_ON_C3_FAILURE"
        summary["not_executed_due_to_gate"] = ["C4", "C5"]
        summary["total_duration_seconds"] = perf_counter() - overall_started
        write_summary(arguments.output_dir, summary)
        raise SystemExit(1)

    c4 = run_c4(arguments.output_dir, long_series)
    summary["results"]["C4"] = c4
    write_summary(arguments.output_dir, summary)
    if c4["outcome"] != "PASS":
        summary["gate_decision"] = "STOPPED_ON_C4_FAILURE"
        summary["not_executed_due_to_gate"] = ["C5"]
        summary["total_duration_seconds"] = perf_counter() - overall_started
        write_summary(arguments.output_dir, summary)
        raise SystemExit(1)

    c1_maximum = c1["measured"]["maximum_relative_energy_change"]
    c5 = run_c5(arguments.output_dir, c1_maximum)
    summary["results"]["C5"] = c5
    summary["gate_decision"] = (
        "PASS_NUMERICAL_VALIDATION"
        if c5["outcome"] == "PASS"
        else "STOPPED_ON_C5_NONPASS"
    )
    summary["total_duration_seconds"] = perf_counter() - overall_started
    write_summary(arguments.output_dir, summary)
    if c5["outcome"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
