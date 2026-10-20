"""Command-line entry point for a deterministic FCC NVE run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .boundaries import PeriodicBoundary
from .initialization import fcc_positions, maxwell_boltzmann_velocities
from .simulation import NVESimulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=3)
    parser.add_argument("--density", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    configuration = fcc_positions(arguments.cells, arguments.density)
    velocities = maxwell_boltzmann_velocities(
        configuration.positions.shape[0],
        arguments.temperature,
        seed=arguments.seed,
    )
    simulation = NVESimulation(
        configuration.positions,
        velocities,
        PeriodicBoundary(configuration.box_length),
    )
    result = simulation.run(
        steps=arguments.steps,
        dt=arguments.dt,
        seed=arguments.seed,
        density=arguments.density,
        initial_temperature=arguments.temperature,
        sample_interval=arguments.sample_interval,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result.to_dict(), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
