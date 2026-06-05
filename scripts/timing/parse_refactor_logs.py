#!/usr/bin/env python3

"""
Parse refactor benchmark logs for ReSolve issue 326.

Expected timing row format:

    TIMING,example,backend,ir_enabled,system,time_ms

The script combines timing rows with residuals already printed by
gpuRefactor, kluRefactor, gluRefactor, and sysRefactor then writes one CSV file for plotting.
"""

import argparse
import csv
import re
from pathlib import Path


FIELDNAMES = [
    "source_log",
    "N",
    "example",
    "backend",
    "method",
    "ir_enabled",
    "system",
    "time_ms",
    "residual",
]

METHOD_LABELS = {
    "kluRefactor": "klu",
    "gpuRefactor": "gpu_refactor",
    "gluRefactor": "glu_refactor",
    "sysRefactor": "sys_refactor",
}

SYSTEM_RE = re.compile(r"^System\s+(\d+):")

TIMING_RE = re.compile(
    r"^TIMING,"
    r"(?P<example>[^,]+),"
    r"(?P<backend>[^,]+),"
    r"(?P<ir_enabled>[01]),"
    r"(?P<system>\d+),"
    r"(?P<time_ms>[-+0-9.eE]+)"
)

RESIDUAL_RE = re.compile(
    r"2-Norm of the residual(?: \(before IR\))?:\s*([-+0-9.eE]+)"
)

FGMRES_RE = re.compile(
    r"FGMRES:\s+init nrm:\s*[-+0-9.eE]+\s+final nrm:\s*([-+0-9.eE]+)"
)

N_RE = re.compile(
    r"(?:^|[_/-])N(?P<N>125|250|500|1000|1k)(?:[_./-]|$)",
    re.IGNORECASE,
)


def infer_problem_size(path: Path) -> str:
    """Infer the GridKit N value from a log filename when possible."""
    match = N_RE.search(str(path))
    if not match:
        return ""

    value = match.group("N")
    if value.lower() == "1k":
        return "1000"

    return value


def method_name(example: str, ir_enabled: str) -> str:
    """Create a compact plotting label from the example and IR flag."""
    method = METHOD_LABELS.get(example, example)

    if ir_enabled == "1":
        return f"{method}_ir"

    return method


def parse_log(path: Path, problem_size: str = "") -> list[dict[str, str]]:
    """Parse one benchmark log into CSV-ready rows."""
    rows = []
    residual_by_system = {}
    current_system = ""

    if not problem_size:
        problem_size = infer_problem_size(path)

    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            line = line.strip()

            system_match = SYSTEM_RE.match(line)
            if system_match:
                current_system = system_match.group(1)
                continue

            residual_match = RESIDUAL_RE.search(line)
            if residual_match and current_system:
                residual_by_system[current_system] = residual_match.group(1)
                continue

            fgmres_match = FGMRES_RE.search(line)
            if fgmres_match and current_system:
                residual_by_system[current_system] = fgmres_match.group(1)
                continue

            timing_match = TIMING_RE.match(line)
            if not timing_match:
                continue

            row = timing_match.groupdict()
            row["source_log"] = str(path)
            row["N"] = problem_size
            row["method"] = method_name(row["example"], row["ir_enabled"])
            row["residual"] = residual_by_system.get(row["system"], "")
            rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Parse ReSolve refactor timing logs into CSV."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="Log files produced by refactor examples with -t.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("refactor_timings.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--N",
        dest="problem_size",
        default="",
        help="GridKit N value to use when it cannot be inferred from log names.",
    )

    return parser.parse_args()


def main() -> int:
    """Parse input logs and write one combined CSV file."""
    args = parse_args()

    rows = []
    for log_path in args.logs:
        rows.extend(parse_log(log_path, args.problem_size))

    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
