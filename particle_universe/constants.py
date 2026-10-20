"""Frozen physical constants for the PU-001 reduced-unit model."""

from __future__ import annotations

import numpy as np

DIMENSIONS = 3
MASS = np.float64(1.0)
SIGMA = np.float64(1.0)
EPSILON = np.float64(1.0)
BOLTZMANN = np.float64(1.0)
CUTOFF = np.float64(2.5)
POTENTIAL_VARIANT = "A"
INTEGRATOR_NAME = "velocity-verlet"
