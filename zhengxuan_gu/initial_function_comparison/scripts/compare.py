"""Score every run at the same interior coordinates, then summarize across PINN seeds."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(RESULTS.parent / "src"))
from profiles import CASES, EVAL_TIMES as TIMES

FD_METHODS = ("explicit_euler", "implicit_euler", "crank_nicolson")
SEEDS = (7, 17, 27)
GRID = np.linspace(0.0, 1.0, 65)
NEGATIVE_TOLERANCE = 1e-8


def check_grid(group):
    if not np.array_equal(group["x"].to_numpy(), GRID):
        raise ValueError("Each time slice must contain exactly the 65 common nodes.")


def measure(prediction, reference):
    interior = prediction[1:-1]
    error = interior - reference[1:-1]
    rmse = np.sqrt(np.mean(error**2))
    reference_rms = np.sqrt(np.mean(reference[1:-1]**2))
    return {
        "rmse": rmse,
        "relative_rmse": rmse / reference_rms,
        "max_abs_error": np.max(np.abs(error)),
        "boundary_max": np.max(np.abs(prediction[[0, -1]])),
        "interior_min": np.min(interior),
        "negative_nodes": np.count_nonzero(interior < -NEGATIVE_TOLERANCE),
    }


def main():
    reference = pd.read_csv(RESULTS / "reference/profiles.csv", float_precision="round_trip")
    fd = pd.read_csv(RESULTS / "fd/profiles.csv", float_precision="round_trip")
    pinn = pd.read_csv(RESULTS / "pinn/profiles.csv", float_precision="round_trip")
    fields = ["case", "method", "N", "seed", "t"]
    # For grouping only: PINN has no solver grid; FD has no random seed.
    predictions = pd.concat([fd.assign(seed=0), pinn.assign(N=0)], ignore_index=True)
    if not np.isfinite(predictions["prediction"]).all() or not np.isfinite(reference["reference"]).all():
        raise ValueError("A prediction or reference value is nonfinite.")
    expected = {(c, m, n, 0, t) for c in CASES for m in FD_METHODS
                for n in (64, 128) for t in TIMES}
    expected.update((c, "pinn", 0, s, t) for c in CASES for s in SEEDS for t in TIMES)
    grouped = predictions.groupby(fields)
    if set(grouped.groups) != expected:
        raise ValueError("Missing or unexpected combinations of cases, settings or times.")

    ref = {}
    for key, group in reference.groupby(["case", "t"]):
        group = group.sort_values("x")
        check_grid(group)
        ref[key] = group["reference"].to_numpy()
    if set(ref) != {(c, t) for c in CASES for t in TIMES}:
        raise ValueError("Unexpected case/time combinations in the reference.")
    metrics = []
    for (case, method, n, seed, t), group in grouped:
        group = group.sort_values("x")
        check_grid(group)
        values = measure(group["prediction"].to_numpy(), ref[case, t])
        if values["boundary_max"] > 1e-12 or (t == 0 and values["max_abs_error"] > 1e-12):
            raise ValueError(f"Initial or boundary check failed for {case}, {method}.")
        metrics.append(dict(case=case, method=method, N=n, seed=seed, t=t, **values))

    per_run = pd.DataFrame(metrics)
    summary = per_run.groupby(["case", "method", "N", "t"], as_index=False).agg(
        runs=("rmse", "count"), rmse=("rmse", "median"),
        rmse_min=("rmse", "min"), rmse_max=("rmse", "max"),
        relative_rmse=("relative_rmse", "median"),
        max_abs_error=("max_abs_error", "median"), boundary_max=("boundary_max", "max"),
        interior_min=("interior_min", "min"),
        negative_runs=("negative_nodes", lambda counts: counts.gt(0).sum()),
        negative_nodes_min=("negative_nodes", "min"), negative_nodes_max=("negative_nodes", "max"),
    )
    for table, filename in [(per_run, "metrics_per_run.csv"), (summary, "summary.csv")]:
        for key in ("N", "seed"):
            if key in table:
                table[key] = table[key].replace(0, np.nan).astype("Int64")
        table.to_csv(RESULTS / filename, index=False)

    terminal = summary.loc[summary.t.eq(TIMES[-1])].copy()
    fd_costs = pd.read_csv(RESULTS / "fd/timings.csv")
    pinn_costs = pd.read_csv(RESULTS / "pinn/timings.csv").groupby("case", as_index=False)[
        ["train_seconds", "query_seconds"]].median()
    fd_table = terminal[terminal.N.notna()].merge(
        fd_costs[["case", "method", "N", "solve_seconds"]],
        on=["case", "method", "N"], validate="one_to_one")
    pinn_table = terminal[terminal.N.isna()].merge(pinn_costs, on="case", validate="one_to_one")
    if (len(fd_table), len(pinn_table)) != (24, 4):
        raise ValueError("Timings do not match the accuracy configurations.")
    pd.concat([fd_table, pinn_table], ignore_index=True).to_csv(
        RESULTS / "terminal_comparison.csv", index=False)
    print(f"Scored {len(per_run)} time slices on 63 common interior nodes.")


if __name__ == "__main__":
    main()
