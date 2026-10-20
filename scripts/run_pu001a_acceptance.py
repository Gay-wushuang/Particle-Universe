"""Execute PU-001A acceptance metrics and write an auditable JSON record."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from tests.acceptance_metrics import measure_all


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    results = measure_all()
    china_standard_time = timezone(timedelta(hours=8), name="Asia/Shanghai")
    record = {
        "task": "PU-001A",
        "scope": ["M1", "M2", "M3", "N3", "N4", "R1"],
        "explicitly_not_executed": ["N1a", "N1b", "N5", "C1", "C2", "C3", "C4", "C5"],
        "solver_commit": git_output("rev-parse", "HEAD"),
        "solver_tree": git_output("rev-parse", "HEAD^{tree}"),
        "command": "python scripts/run_pu001a_acceptance.py --output artifacts/pu-001a-acceptance.json",
        "executed_at": datetime.now(china_standard_time).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "results": results,
        "overall_passed": all(item["passed"] for item in results.values()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not record["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
