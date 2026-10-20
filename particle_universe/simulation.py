"""NVE orchestration, force-call accounting, trajectories, and metadata."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .boundaries import Boundary
from .constants import (
    BOLTZMANN,
    CUTOFF,
    DIMENSIONS,
    EPSILON,
    MASS,
    POTENTIAL_VARIANT,
    SIGMA,
)
from .diagnostics import ThermodynamicSample, kinetic_energy, sample, total_momentum
from .integrator import VelocityVerlet
from .potential import ForceResult, ShiftedLennardJones
from .state import ParticleState
from .validation import integer, nonnegative_scalar, particle_matrix, positive_scalar, seed_value


@dataclass(slots=True)
class ForceEvaluationCounts:
    initial: int = 0
    step: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"initial": self.initial, "step": self.step}


@dataclass(frozen=True, slots=True)
class RunResult:
    metadata: dict[str, Any]
    samples: tuple[ThermodynamicSample, ...]
    positions: tuple[np.ndarray, ...]
    velocities: tuple[np.ndarray, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "samples": [item.to_dict() for item in self.samples],
            "trajectory": {
                "positions": [item.tolist() for item in self.positions],
                "velocities": [item.tolist() for item in self.velocities],
            },
        }


class NVESimulation:
    """Own an NVE state while keeping all physical components explicit."""

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        boundary: Boundary,
        *,
        potential: ShiftedLennardJones | None = None,
    ) -> None:
        self.boundary = boundary
        self.potential = potential if potential is not None else ShiftedLennardJones()
        self.integrator = VelocityVerlet()
        self.force_evaluations = ForceEvaluationCounts()

        initial_positions = boundary.wrap(
            particle_matrix(positions, "positions", minimum_particles=2, copy=True)
        )
        initial_velocities = particle_matrix(
            velocities, "velocities", minimum_particles=2, copy=True
        )
        if initial_positions.shape != initial_velocities.shape:
            raise ValueError("positions and velocities must have the same shape")

        initial_force = self.potential.evaluate(initial_positions, boundary)
        self.force_evaluations.initial = 1
        self.state = ParticleState(
            positions=initial_positions,
            velocities=initial_velocities,
            forces=initial_force.forces,
            time=np.float64(0.0),
            potential_energy=initial_force.potential_energy,
            kinetic_energy=kinetic_energy(initial_velocities),
        )

    def _evaluate_step_force(self, positions: np.ndarray) -> ForceResult:
        result = self.potential.evaluate(positions, self.boundary)
        self.force_evaluations.step += 1
        return result

    def step(self, dt: float) -> ParticleState:
        self.state = self.integrator.step(
            self.state, dt, self.boundary, self._evaluate_step_force
        )
        return self.state

    def run(
        self,
        *,
        steps: int,
        dt: float,
        seed: int,
        density: float | None,
        initial_temperature: float,
        sample_interval: int = 1,
        solver_commit: str | None = None,
    ) -> RunResult:
        step_count = integer(steps, "steps", minimum=0)
        interval = integer(sample_interval, "sample_interval", minimum=1)
        time_step = positive_scalar(dt, "dt")
        deterministic_seed = seed_value(seed)
        reduced_temperature = nonnegative_scalar(
            initial_temperature, "initial_temperature"
        )
        reduced_density = (
            None if density is None else float(positive_scalar(density, "density"))
        )

        started_at = datetime.now(timezone.utc)
        started_clock = perf_counter()
        initial_energy = np.float64(
            self.state.potential_energy + self.state.kinetic_energy
        )
        initial_momentum = total_momentum(self.state.velocities)

        samples: list[ThermodynamicSample] = [sample(self.state, 0)]
        positions: list[np.ndarray] = [self.state.positions.copy()]
        velocities: list[np.ndarray] = [self.state.velocities.copy()]

        for current_step in range(1, step_count + 1):
            self.step(time_step)
            if current_step % interval == 0 or current_step == step_count:
                samples.append(sample(self.state, current_step))
                positions.append(self.state.positions.copy())
                velocities.append(self.state.velocities.copy())

        wall_time = perf_counter() - started_clock
        finished_at = datetime.now(timezone.utc)
        energy_values = np.asarray(
            [item.total_energy for item in samples], dtype=np.float64
        )
        momentum_values = np.asarray(
            [item.total_momentum for item in samples], dtype=np.float64
        )
        absolute_energy_drift = np.abs(energy_values - initial_energy)
        if initial_energy == 0.0:
            relative_energy_drift: float | None = None
        else:
            relative_energy_drift = float(
                np.max(absolute_energy_drift) / abs(initial_energy)
            )
        momentum_drift = np.linalg.norm(
            momentum_values - initial_momentum, axis=1
        )

        metadata: dict[str, Any] = {
            "solver_commit": solver_commit or detect_solver_commit(),
            "reference_commit": None,
            "seed": deterministic_seed,
            "N": self.state.particle_count,
            "dim": DIMENSIONS,
            "rho_star": reduced_density,
            "initial_temperature": float(reduced_temperature),
            "temperature_dof": 3 * self.state.particle_count - 3,
            "sigma": float(SIGMA),
            "epsilon": float(EPSILON),
            "mass": float(MASS),
            "k_B": float(BOLTZMANN),
            "rc": float(CUTOFF),
            "potential_variant": POTENTIAL_VARIANT,
            "integrator": self.integrator.name,
            "dt": float(time_step),
            "steps": step_count,
            "t_end": float(time_step * step_count),
            "dt_ref": None,
            "crossing_count": None,
            "boundary": self.boundary.name,
            "L": (
                None
                if self.boundary.box_length is None
                else float(self.boundary.box_length)
            ),
            "minimum_image_valid": (
                None
                if self.boundary.box_length is None
                else bool(self.boundary.box_length / 2.0 > CUTOFF)
            ),
            "energy_drift": {
                "max_absolute": float(np.max(absolute_energy_drift)),
                "max_relative": relative_energy_drift,
            },
            "momentum_drift": float(np.max(momentum_drift)),
            "force_evaluations": self.force_evaluations.to_dict(),
            "timestamps": {
                "started_utc": started_at.isoformat(),
                "finished_utc": finished_at.isoformat(),
            },
            "wall_time_seconds": float(wall_time),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        }
        return RunResult(
            metadata=metadata,
            samples=tuple(samples),
            positions=tuple(positions),
            velocities=tuple(velocities),
        )


def detect_solver_commit() -> str:
    override = os.environ.get("PARTICLE_UNIVERSE_COMMIT")
    if override:
        return override
    repository = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if commit else "unknown"
