"""The sole PU-001 production integrator: kick-drift-kick Velocity-Verlet."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .boundaries import Boundary
from .constants import MASS
from .diagnostics import kinetic_energy
from .potential import ForceResult
from .state import ParticleState
from .validation import finite_result, positive_scalar

ForceEvaluator = Callable[[np.ndarray], ForceResult]


class VelocityVerlet:
    name = "velocity-verlet"

    def step(
        self,
        state: ParticleState,
        dt: float,
        boundary: Boundary,
        evaluate_new_force: ForceEvaluator,
    ) -> ParticleState:
        time_step = positive_scalar(dt, "dt")
        half_velocity = state.velocities + 0.5 * time_step * state.forces / MASS
        new_positions = boundary.wrap(state.positions + time_step * half_velocity)

        # Exactly one force evaluation occurs per completed Velocity-Verlet step.
        force_result = evaluate_new_force(new_positions)
        new_velocities = (
            half_velocity + 0.5 * time_step * force_result.forces / MASS
        )
        finite_result(new_positions, "integrated positions")
        finite_result(new_velocities, "integrated velocities")

        return ParticleState(
            positions=new_positions,
            velocities=new_velocities,
            forces=force_result.forces,
            time=np.float64(state.time + time_step),
            potential_energy=force_result.potential_energy,
            kinetic_energy=kinetic_energy(new_velocities),
        )
