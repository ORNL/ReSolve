#!/usr/bin/env python3

"""
Plot refactor benchmark results for ReSolve issue 326.

Input CSV format is produced by parse_refactor_logs.py.

Expected columns:

    source_log,N,example,backend,method,ir_enabled,system,time_ms,residual

The script produces:

    - one solve-time graph per N
    - one residual graph per N
    - one average runtime scaling graph
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FIGURE_SIZE = (10, 6)
DPI = 200
GRID_ALPHA = 0.3

LEGEND_OPTIONS = {
    "loc": "center left",
    "bbox_to_anchor": (1.02, 0.5),
    "fontsize": "small",
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load benchmark rows from the parser output CSV."""
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def row_label(row: dict[str, str]) -> str:
    """Create a stable plot label from backend and method."""
    return f"{row['backend']} {row['method']}"


def group_by_key(rows: list[dict[str, str]],
                 key: str) -> dict[str, list[dict[str, str]]]:
    """Group CSV rows by a string-valued column."""
    grouped = defaultdict(list)

    for row in rows:
        grouped[row[key]].append(row)

    return grouped


def sorted_systems(rows: list[dict[str, str]]) -> list[int]:
    """Return sorted linear-system indices present in a row group."""
    return sorted({int(row["system"]) for row in rows})


def save_current_plot(output_path: Path) -> None:
    """Save the active matplotlib figure and close it."""
    plt.legend(**LEGEND_OPTIONS)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_problem_metric(problem_size: str,
                        rows: list[dict[str, str]],
                        metric_key: str,
                        y_label: str,
                        title: str,
                        output_path: Path,
                        log_scale: bool = False) -> None:
    """Plot one per-system metric for one GridKit problem size."""
    grouped = defaultdict(list)

    for row in rows:
        grouped[row_label(row)].append(row)

    plt.figure(figsize=FIGURE_SIZE)

    for label, label_rows in sorted(grouped.items()):
        sorted_rows = sorted(label_rows, key=lambda row: int(row["system"]))

        systems = []
        values = []
        for row in sorted_rows:
            if not row[metric_key]:
                continue

            systems.append(int(row["system"]))
            values.append(float(row[metric_key]))

        if systems:
            plt.plot(systems, values, marker="o", label=label)

    plt.xlabel("Linear system")
    plt.ylabel(y_label)
    plt.title(title)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))

    if log_scale:
        # Residuals can vary by orders of magnitude, especially after
        # iterative refinement.
        plt.yscale("log")
        plt.grid(True, axis="y", alpha=GRID_ALPHA, which="both")
    else:
        plt.grid(True, axis="y", alpha=GRID_ALPHA)

    save_current_plot(output_path)


def plot_time_for_problem(problem_size: str,
                          rows: list[dict[str, str]],
                          output_dir: Path) -> None:
    """Plot solve time per linear system for one GridKit problem size."""
    plot_problem_metric(
        problem_size=problem_size,
        rows=rows,
        metric_key="time_ms",
        y_label="Solve time (ms)",
        title=f"GridKit N={problem_size}: solve time per system",
        output_path=output_dir / f"N{problem_size}_solve_time.png",
    )


def plot_residual_for_problem(problem_size: str,
                              rows: list[dict[str, str]],
                              output_dir: Path) -> None:
    """Plot relative residual per linear system for one GridKit problem size."""
    plot_problem_metric(
        problem_size=problem_size,
        rows=rows,
        metric_key="residual",
        y_label="Relative residual",
        title=f"GridKit N={problem_size}: residual per system",
        output_path=output_dir / f"N{problem_size}_residual.png",
        log_scale=True,
    )


def average_times_by_problem(rows: list[dict[str, str]]) -> list[tuple[int, float]]:
    """Average solve time over all systems for each GridKit N value."""
    times_by_problem = defaultdict(list)

    for row in rows:
        if row["N"]:
            times_by_problem[row["N"]].append(float(row["time_ms"]))

    averages = []
    for problem_size, times in times_by_problem.items():
        averages.append((int(problem_size), sum(times) / len(times)))

    return sorted(averages)


def plot_average_runtime_scaling(rows: list[dict[str, str]],
                                 output_dir: Path) -> None:
    """Plot average solve time over all systems for each N."""
    grouped = defaultdict(list)
    problem_sizes = sorted({int(row["N"]) for row in rows if row["N"]})

    for row in rows:
        grouped[row_label(row)].append(row)

    plt.figure(figsize=FIGURE_SIZE)

    for label, label_rows in sorted(grouped.items()):
        points = average_times_by_problem(label_rows)

        if not points:
            continue

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        plt.plot(x_values, y_values, marker="o", label=label)

    plt.xlabel("GridKit N")
    plt.ylabel("Average solve time (ms)")
    plt.title("Average solve time scaling")
    plt.xticks(problem_sizes)
    plt.grid(True, alpha=GRID_ALPHA)

    if len(problem_sizes) == 1:
        # Local smoke tests often use only one synthetic N value. Add x-axis
        # padding so the points are not drawn directly on the plot boundary.
        n_value = problem_sizes[0]
        padding = max(1, int(0.05 * n_value))
        plt.xlim(n_value - padding, n_value + padding)

    save_current_plot(output_dir / "average_solve_time_scaling.png")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot ReSolve refactor benchmark results."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV file produced by parse_refactor_logs.py.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("refactor_plots"),
        help="Directory where plot images will be written.",
    )

    return parser.parse_args()


def main() -> int:
    """Create issue 326 plots from parsed benchmark CSV data."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.csv_path)
    rows_by_problem = group_by_key(rows, "N")

    for problem_size, problem_rows in sorted(
        rows_by_problem.items(),
        key=lambda item: int(item[0]) if item[0] else -1,
    ):
        if not problem_size:
            continue

        plot_time_for_problem(problem_size, problem_rows, args.output_dir)
        plot_residual_for_problem(problem_size, problem_rows, args.output_dir)

    plot_average_runtime_scaling(rows, args.output_dir)

    print(f"Wrote plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
